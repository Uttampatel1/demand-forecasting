"""Distribution-free prediction intervals via split-conformal calibration.

A point forecast answers *"how much?"*; inventory, staffing and budget decisions
need *"within what error?"*. Split conformal prediction wraps a calibrated
interval around **any** forecaster's point prediction, with a finite-sample
coverage guarantee and no distributional assumptions:

1. Hold out the tail of the training series as a calibration set.
2. Fit on the earlier part, measure the model's absolute errors on the held-out tail.
3. Pad future point forecasts by the appropriate residual quantile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .models import Forecaster


@dataclass
class IntervalForecast:
    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    nominal_coverage: float   # target 1 - alpha


def conformal_interval(
    model_factory: Callable[[], Forecaster],
    train: pd.DataFrame,
    future: pd.DataFrame,
    alpha: float = 0.1,
    calib_size: int = 14,
) -> IntervalForecast:
    """Split-conformal prediction interval around a forecaster's point forecast.

    ``model_factory`` is a zero-argument callable returning a *fresh, unfitted*
    forecaster (we fit it twice — once to calibrate, once on all data). The last
    ``calib_size`` rows of ``train`` are used to estimate residual quantiles.
    """
    n = len(train)
    calib_size = int(min(calib_size, max(1, n // 5)))
    fit_part = train.iloc[: n - calib_size]
    calib_part = train.iloc[n - calib_size :]

    # Calibrate on a clean held-out tail the model never trained on.
    cal_pred = model_factory().fit(fit_part).predict(calib_part)
    residuals = np.abs(calib_part["sales"].to_numpy(dtype=float) - cal_pred)

    # Conformal quantile with the finite-sample correction (capped at 1.0).
    q_level = min(1.0, np.ceil((calib_size + 1) * (1 - alpha)) / calib_size)
    q = float(np.quantile(residuals, q_level))

    # Refit on the full training window for the production forecast.
    point = model_factory().fit(train).predict(future)
    return IntervalForecast(
        point=np.asarray(point, dtype=float),
        lower=np.clip(np.asarray(point, dtype=float) - q, 0.0, None),
        upper=np.asarray(point, dtype=float) + q,
        nominal_coverage=1.0 - alpha,
    )


def empirical_coverage(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Fraction of actuals that fell inside the interval (for validation)."""
    actual = np.asarray(actual, dtype=float)
    inside = (actual >= np.asarray(lower)) & (actual <= np.asarray(upper))
    return float(inside.mean())
