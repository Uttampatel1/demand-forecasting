import numpy as np
import pandas as pd

from src.intervals import conformal_interval, empirical_coverage
from src.models import EnsembleForecaster, ETSForecaster, SeasonalNaiveForecaster


def _series(n=200):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    base = 100 + 15 * np.asarray((idx.dayofweek >= 5), dtype=int)
    noise = np.random.default_rng(0).normal(0, 3, n)
    return pd.DataFrame({"sales": base + noise, "promo": 0, "holiday": 0}, index=idx)


def _future(series, h=14):
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    return pd.DataFrame({"sales": 0.0, "promo": 0, "holiday": 0}, index=idx)


def test_ensemble_averages_member_predictions():
    s = _series()
    fut = _future(s, 14)
    members = [ETSForecaster(), SeasonalNaiveForecaster(7)]
    ens = EnsembleForecaster(members).fit(s)
    expected = np.vstack([m.predict(fut) for m in members]).mean(axis=0)
    np.testing.assert_allclose(ens.predict(fut), expected)


def test_conformal_interval_brackets_point_and_orders_bounds():
    s = _series()
    fut = _future(s, 14)
    iv = conformal_interval(lambda: SeasonalNaiveForecaster(7), s, fut, alpha=0.1)
    assert iv.point.shape == (14,)
    assert np.all(iv.lower <= iv.point)
    assert np.all(iv.point <= iv.upper)
    assert iv.nominal_coverage == 0.9
    assert np.all(iv.lower >= 0.0)        # demand can't be negative


def test_conformal_interval_widens_for_higher_confidence():
    s = _series()
    fut = _future(s, 14)
    narrow = conformal_interval(lambda: SeasonalNaiveForecaster(7), s, fut, alpha=0.2)
    wide = conformal_interval(lambda: SeasonalNaiveForecaster(7), s, fut, alpha=0.05)
    assert (wide.upper - wide.lower).mean() >= (narrow.upper - narrow.lower).mean()


def test_empirical_coverage_counts_hits():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 0.0, 0.0, 10.0])
    upper = np.array([2.0, 2.0, 2.0, 12.0])
    # first two inside, third above upper, fourth below lower => 0.5
    assert empirical_coverage(actual, lower, upper) == 0.5
