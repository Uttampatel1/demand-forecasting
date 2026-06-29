"""Rolling-origin (walk-forward) backtesting.

A single hold-out (``evaluate.backtest``) reports the error on *one* slice of
history — which can be lucky or unlucky. Rolling-origin evaluation re-fits the
model at several cut-off points and forecasts the next ``horizon`` days from each,
producing a *distribution* of errors instead of a single number. That distribution
is what tells you whether a model is genuinely better or just won one draw, and
its spread is an honest estimate of how much next month's error might vary.

By default the training window *expands* (every origin trains on all history up to
its cut-off), which matches how a forecast is actually re-run in production. Pass
``window="sliding"`` for a fixed-length training window instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluate import all_metrics
from .models import Forecaster


@dataclass
class OriginResult:
    """One walk-forward fold: forecast the ``horizon`` days after a cut-off."""

    cutoff: pd.Timestamp
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    test_index: pd.DatetimeIndex


@dataclass
class RollingBacktestResult:
    """Aggregate of several origins for one model."""

    model: str
    origins: list[OriginResult]
    # mean / std of each metric across origins (the error *distribution*)
    mean_metrics: dict[str, float]
    std_metrics: dict[str, float]
    # metrics computed on all folds pooled together
    pooled_metrics: dict[str, float]

    @property
    def n_origins(self) -> int:
        return len(self.origins)


def rolling_origin_splits(
    series: pd.DataFrame,
    horizon: int,
    n_origins: int = 3,
    step: int | None = None,
    window: str = "expanding",
    min_train: int | None = None,
):
    """Yield ``(train, test)`` frames for each origin, earliest cut-off first.

    The most recent test window is always the final ``horizon`` rows; each earlier
    origin steps back by ``step`` days (default: ``horizon``, i.e. non-overlapping
    test windows). With ``window="sliding"`` the training window keeps a fixed
    length equal to the first origin's training size.
    """
    if step is None:
        step = horizon
    if step <= 0 or horizon <= 0 or n_origins <= 0:
        raise ValueError("horizon, step and n_origins must be positive")

    n = len(series)
    # Cut-offs measured from the end: the k-th origin's test ends at n - k*step.
    ends = [n - k * step for k in range(n_origins)]
    ends = [e for e in ends if e - horizon >= 1]  # need at least 1 training row
    if not ends:
        raise ValueError("series too short for the requested horizon/origins")
    ends.sort()  # chronological order

    train_len = min(e - horizon for e in ends) if window == "sliding" else None
    for end in ends:
        test = series.iloc[end - horizon : end]
        train_full = series.iloc[: end - horizon]
        if window == "sliding":
            train = train_full.iloc[-train_len:]
        elif window == "expanding":
            train = train_full
        else:
            raise ValueError("window must be 'expanding' or 'sliding'")
        if min_train is not None and len(train) < min_train:
            continue
        yield train, test


def rolling_origin_backtest(
    series: pd.DataFrame,
    model: Forecaster,
    horizon: int = 28,
    n_origins: int = 3,
    step: int | None = None,
    window: str = "expanding",
    min_train: int | None = None,
) -> RollingBacktestResult:
    """Walk-forward backtest of ``model`` across ``n_origins`` cut-offs.

    A fresh model is fit at every origin (we deep-copy so callers can reuse the
    instance). Returns per-origin results plus the mean/std of each metric and the
    pooled metrics over all folds.
    """
    from copy import deepcopy

    origins: list[OriginResult] = []
    for train, test in rolling_origin_splits(
        series, horizon, n_origins, step, window, min_train
    ):
        fitted = deepcopy(model).fit(train)
        preds = np.clip(fitted.predict(test), 0, None)
        y_true = test["sales"].to_numpy()
        origins.append(
            OriginResult(
                cutoff=train.index[-1],
                metrics=all_metrics(y_true, preds),
                y_true=y_true,
                y_pred=preds,
                test_index=test.index,
            )
        )

    if not origins:
        raise ValueError("no valid origins produced (check horizon/min_train)")

    metric_names = list(origins[0].metrics)
    mean_metrics = {
        m: round(float(np.mean([o.metrics[m] for o in origins])), 2)
        for m in metric_names
    }
    std_metrics = {
        m: round(float(np.std([o.metrics[m] for o in origins])), 2)
        for m in metric_names
    }
    pooled_true = np.concatenate([o.y_true for o in origins])
    pooled_pred = np.concatenate([o.y_pred for o in origins])
    return RollingBacktestResult(
        model=model.name,
        origins=origins,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        pooled_metrics=all_metrics(pooled_true, pooled_pred),
    )
