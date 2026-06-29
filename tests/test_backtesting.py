import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    RollingBacktestResult,
    rolling_origin_backtest,
    rolling_origin_splits,
)
from src.models import SeasonalNaiveForecaster


def _series(n=180):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    sales = 100 + 10 * np.asarray(idx.dayofweek >= 5, dtype=int)
    return pd.DataFrame({"sales": sales}, index=idx)


def test_splits_are_chronological_and_sized():
    df = _series(120)
    splits = list(rolling_origin_splits(df, horizon=14, n_origins=3))
    assert len(splits) == 3
    cutoffs = [tr.index[-1] for tr, _ in splits]
    assert cutoffs == sorted(cutoffs)  # earliest origin first
    for train, test in splits:
        assert len(test) == 14
        assert train.index[-1] < test.index[0]  # no leakage


def test_expanding_window_grows_sliding_is_constant():
    df = _series(150)
    exp = list(rolling_origin_splits(df, horizon=10, n_origins=3, window="expanding"))
    exp_sizes = [len(tr) for tr, _ in exp]
    assert exp_sizes == sorted(exp_sizes) and exp_sizes[0] < exp_sizes[-1]

    sli = list(rolling_origin_splits(df, horizon=10, n_origins=3, window="sliding"))
    sli_sizes = {len(tr) for tr, _ in sli}
    assert len(sli_sizes) == 1  # fixed-length training window


def test_non_overlapping_test_windows_by_default():
    df = _series(150)
    tests = [test.index for _, test in rolling_origin_splits(df, horizon=10, n_origins=3)]
    for earlier, later in zip(tests, tests[1:]):
        assert earlier[-1] < later[0]


def test_backtest_aggregates_distribution():
    df = _series(180)
    res = rolling_origin_backtest(df, SeasonalNaiveForecaster(period=7), horizon=14, n_origins=4)
    assert isinstance(res, RollingBacktestResult)
    assert res.n_origins == 4
    assert set(res.mean_metrics) == {"MAE", "RMSE", "MAPE", "sMAPE"}
    # std is a real distribution measure -> non-negative
    assert all(v >= 0 for v in res.std_metrics.values())
    # pooled predictions cover every fold
    assert sum(len(o.y_pred) for o in res.origins) == 14 * 4


def test_seasonal_naive_is_near_perfect_on_seasonal_data():
    # period-7 seasonal-naive should nail a pure weekly pattern across all origins
    df = _series(180)
    res = rolling_origin_backtest(df, SeasonalNaiveForecaster(period=7), horizon=7, n_origins=3)
    assert res.pooled_metrics["MAE"] < 1e-6


def test_model_instance_is_reusable_across_calls():
    df = _series(120)
    model = SeasonalNaiveForecaster(period=7)
    r1 = rolling_origin_backtest(df, model, horizon=7, n_origins=2)
    r2 = rolling_origin_backtest(df, model, horizon=7, n_origins=2)
    assert r1.pooled_metrics == r2.pooled_metrics  # deepcopy -> no fit() bleed


def test_raises_when_series_too_short():
    df = _series(10)
    with pytest.raises(ValueError):
        rolling_origin_backtest(df, SeasonalNaiveForecaster(), horizon=20, n_origins=3)
