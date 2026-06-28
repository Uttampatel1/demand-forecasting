import numpy as np
import pandas as pd

from src.features import calendar_features, make_supervised


def _series(n=120):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {"sales": np.arange(n) + 100, "promo": 0, "holiday": 0}, index=idx
    )
    return df


def test_calendar_features_columns():
    cal = calendar_features(pd.date_range("2022-01-01", periods=10, freq="D"))
    assert {"dayofweek", "month", "is_weekend", "doy_sin", "doy_cos"} <= set(cal.columns)
    assert cal["doy_sin"].between(-1, 1).all()


def test_make_supervised_creates_lags_and_drops_warmup():
    sup = make_supervised(_series(120))
    assert "lag_28" in sup.columns
    assert "rollmean_7" in sup.columns
    assert not sup.isna().any().any()
    # 28-day max lag + 28-day rolling window warm-up is dropped.
    assert len(sup) == 120 - 28


def test_make_supervised_target_present():
    sup = make_supervised(_series())
    assert "sales" in sup.columns
