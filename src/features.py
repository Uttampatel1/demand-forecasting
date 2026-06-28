"""Feature engineering for the ML (gradient-boosting) forecaster.

Turns a univariate daily series (plus known-in-advance exogenous flags like
promotions/holidays) into a supervised-learning table with lag and calendar
features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 7, 14, 28)
ROLL_WINDOWS = (7, 28)
EXOG_COLS = ("promo", "holiday")


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    doy = dates.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "dayofweek": dates.dayofweek,
            "month": dates.month,
            "day": dates.day,
            "is_weekend": (dates.dayofweek >= 5).astype(int),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=dates,
    )


def make_supervised(
    df: pd.DataFrame,
    target: str = "sales",
    exog: tuple[str, ...] = EXOG_COLS,
) -> pd.DataFrame:
    """Build a feature frame from a single series.

    ``df`` must be indexed by date and contain ``target`` and any ``exog`` cols.
    Rows with NaN lags (the warm-up period) are dropped.
    """
    df = df.sort_index()
    feats = calendar_features(df.index)
    feats[target] = df[target].to_numpy()

    for col in exog:
        if col in df.columns:
            feats[col] = df[col].to_numpy()

    for lag in LAGS:
        feats[f"lag_{lag}"] = feats[target].shift(lag)
    for window in ROLL_WINDOWS:
        feats[f"rollmean_{window}"] = feats[target].shift(1).rolling(window).mean()
        feats[f"rollstd_{window}"] = feats[target].shift(1).rolling(window).std()

    return feats.dropna()


def feature_columns(frame: pd.DataFrame, target: str = "sales") -> list[str]:
    return [c for c in frame.columns if c != target]
