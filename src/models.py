"""Forecasting models behind a common interface.

Every forecaster implements::

    fit(train: pd.DataFrame) -> self
    predict(future: pd.DataFrame) -> np.ndarray

where ``train``/``future`` are date-indexed frames with a ``sales`` column
(``future.sales`` is ignored; its index defines the horizon and it carries the
known-in-advance exogenous flags).

Models: seasonal-naive baseline, Holt-Winters (ETS), SARIMA, LightGBM, and an
optional Prophet wrapper.
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from .features import EXOG_COLS, LAGS, ROLL_WINDOWS, calendar_features, make_supervised

warnings.filterwarnings("ignore")  # silence statsmodels convergence chatter


class Forecaster(ABC):
    name: str = "forecaster"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, future: pd.DataFrame) -> np.ndarray:
        ...


class SeasonalNaiveForecaster(Forecaster):
    """Predict each day with the value from ``period`` days earlier."""

    name = "SeasonalNaive"

    def __init__(self, period: int = 7) -> None:
        self.period = period

    def fit(self, train: pd.DataFrame) -> "SeasonalNaiveForecaster":
        self._history = train["sales"].to_numpy(dtype=float)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        h = len(future)
        hist = list(self._history)
        preds = []
        for _ in range(h):
            preds.append(hist[-self.period])
            hist.append(hist[-self.period])
        return np.array(preds)


class ETSForecaster(Forecaster):
    """Holt-Winters exponential smoothing with weekly seasonality."""

    name = "ETS(Holt-Winters)"

    def __init__(self, seasonal_periods: int = 7) -> None:
        self.seasonal_periods = seasonal_periods

    def fit(self, train: pd.DataFrame) -> "ETSForecaster":
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        y = train["sales"].astype(float)
        self._model = ExponentialSmoothing(
            y,
            trend="add",
            seasonal="add",
            seasonal_periods=self.seasonal_periods,
            initialization_method="estimated",
        ).fit()
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._model.forecast(len(future)))


class SARIMAForecaster(Forecaster):
    """SARIMAX with weekly seasonal terms and exogenous flags."""

    name = "SARIMA"

    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7)) -> None:
        self.order = order
        self.seasonal_order = seasonal_order

    def _exog(self, frame: pd.DataFrame) -> np.ndarray | None:
        cols = [c for c in EXOG_COLS if c in frame.columns]
        return frame[cols].astype(float).to_numpy() if cols else None

    def fit(self, train: pd.DataFrame) -> "SARIMAForecaster":
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y = train["sales"].astype(float)
        self._model = SARIMAX(
            y,
            exog=self._exog(train),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        fc = self._model.get_forecast(len(future), exog=self._exog(future))
        return np.asarray(fc.predicted_mean)


class LightGBMForecaster(Forecaster):
    """Gradient-boosted trees on lag + calendar features, forecast recursively."""

    name = "LightGBM"

    def __init__(self, **params) -> None:
        self.params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "verbose": -1,
            **params,
        }

    def fit(self, train: pd.DataFrame) -> "LightGBMForecaster":
        from lightgbm import LGBMRegressor

        sup = make_supervised(train)
        self._feature_cols = [c for c in sup.columns if c != "sales"]
        self._model = LGBMRegressor(**self.params)
        self._model.fit(sup[self._feature_cols], sup["sales"])
        self._history = train.copy()
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        history = self._history.copy()
        preds: list[float] = []
        exog_cols = [c for c in EXOG_COLS if c in future.columns]

        for ts, row in future.iterrows():
            series = history["sales"]
            cal = calendar_features(pd.DatetimeIndex([ts])).iloc[0]
            feat = dict(cal)
            for col in exog_cols:
                feat[col] = row[col]
            for lag in LAGS:
                feat[f"lag_{lag}"] = series.iloc[-lag]
            for window in ROLL_WINDOWS:
                feat[f"rollmean_{window}"] = series.iloc[-window:].mean()
                feat[f"rollstd_{window}"] = series.iloc[-window:].std()

            x = pd.DataFrame([feat])[self._feature_cols]
            yhat = float(self._model.predict(x)[0])
            preds.append(yhat)

            new_row = {"sales": yhat}
            for col in exog_cols:
                new_row[col] = row[col]
            history = pd.concat(
                [history, pd.DataFrame([new_row], index=[ts])]
            )
        return np.array(preds)


class ProphetForecaster(Forecaster):
    """Facebook Prophet wrapper (optional dependency)."""

    name = "Prophet"

    def fit(self, train: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet

        df = pd.DataFrame({"ds": train.index, "y": train["sales"].astype(float).to_numpy()})
        self._model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        for col in EXOG_COLS:
            if col in train.columns:
                df[col] = train[col].to_numpy()
                self._model.add_regressor(col)
        self._exog_cols = [c for c in EXOG_COLS if c in train.columns]
        self._model.fit(df)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame({"ds": future.index})
        for col in self._exog_cols:
            df[col] = future[col].to_numpy()
        forecast = self._model.predict(df)
        return forecast["yhat"].clip(lower=0).to_numpy()


class EnsembleForecaster(Forecaster):
    """Average several forecasters — a cheap, robust hedge against any one model.

    Ensembling rarely wins every period, but it reliably reduces the *variance*
    of forecast error: when SARIMA over-reacts and ETS under-reacts, their mean
    is usually closer than either alone.
    """

    name = "Ensemble(mean)"

    def __init__(self, members: list[Forecaster] | None = None) -> None:
        self.members = members or [ETSForecaster(), SARIMAForecaster()]

    def fit(self, train: pd.DataFrame) -> "EnsembleForecaster":
        for m in self.members:
            m.fit(train)
        return self

    def predict(self, future: pd.DataFrame) -> np.ndarray:
        preds = np.vstack([m.predict(future) for m in self.members])
        return preds.mean(axis=0)


def default_models(include_prophet: bool = True) -> list[Forecaster]:
    models: list[Forecaster] = [
        SeasonalNaiveForecaster(),
        ETSForecaster(),
        SARIMAForecaster(),
        LightGBMForecaster(),
        EnsembleForecaster(),
    ]
    if include_prophet and prophet_available():
        models.append(ProphetForecaster())
    return models


def prophet_available() -> bool:
    """True only if Prophet imports *and* its Stan backend initializes."""
    try:
        from prophet import Prophet

        Prophet()  # constructing loads the stan backend; fails fast if broken
        return True
    except Exception:  # noqa: BLE001
        return False
