"""Evaluation metrics and a hold-out backtest for forecasters."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import Forecaster


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom != 0
    return float(np.mean(2 * np.abs(y_pred - y_true)[mask] / denom[mask]) * 100)


def all_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": round(mae(y_true, y_pred), 2),
        "RMSE": round(rmse(y_true, y_pred), 2),
        "MAPE": round(mape(y_true, y_pred), 2),
        "sMAPE": round(smape(y_true, y_pred), 2),
    }


@dataclass
class BacktestResult:
    model: str
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    test_index: pd.DatetimeIndex


def train_test_split(series: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(series) <= horizon:
        raise ValueError("series shorter than horizon")
    return series.iloc[:-horizon], series.iloc[-horizon:]


def backtest(
    series: pd.DataFrame,
    model: Forecaster,
    horizon: int = 28,
) -> BacktestResult:
    """Fit ``model`` on all but the last ``horizon`` days and score the holdout."""
    train, test = train_test_split(series, horizon)
    model.fit(train)
    preds = np.clip(model.predict(test), 0, None)
    return BacktestResult(
        model=model.name,
        metrics=all_metrics(test["sales"].to_numpy(), preds),
        y_true=test["sales"].to_numpy(),
        y_pred=preds,
        test_index=test.index,
    )
