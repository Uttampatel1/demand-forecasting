import numpy as np
import pandas as pd

from src.models import (
    ETSForecaster,
    LightGBMForecaster,
    SeasonalNaiveForecaster,
    default_models,
)


def _series(n=200):
    idx = pd.date_range("2022-01-01", periods=n, freq="D")
    base = 100 + 15 * np.asarray((idx.dayofweek >= 5), dtype=int)
    noise = np.random.default_rng(0).normal(0, 3, n)
    return pd.DataFrame(
        {"sales": base + noise, "promo": 0, "holiday": 0}, index=idx
    )


def _future(series, h=14):
    idx = pd.date_range(series.index[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    return pd.DataFrame({"sales": 0.0, "promo": 0, "holiday": 0}, index=idx)


def test_seasonal_naive_repeats_last_period():
    s = _series(40)
    m = SeasonalNaiveForecaster(period=7).fit(s)
    preds = m.predict(_future(s, 7))
    assert len(preds) == 7
    np.testing.assert_allclose(preds, s["sales"].to_numpy()[-7:])


def test_ets_predict_shape():
    s = _series()
    preds = ETSForecaster().fit(s).predict(_future(s, 14))
    assert len(preds) == 14
    assert np.isfinite(preds).all()


def test_lightgbm_recursive_forecast_shape():
    s = _series()
    preds = LightGBMForecaster(n_estimators=50).fit(s).predict(_future(s, 14))
    assert len(preds) == 14
    assert np.isfinite(preds).all()


def test_default_models_nonempty():
    assert len(default_models(include_prophet=False)) == 4
