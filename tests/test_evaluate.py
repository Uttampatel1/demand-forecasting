import numpy as np
import pandas as pd

from src.evaluate import all_metrics, backtest, mape, rmse, train_test_split
from src.models import SeasonalNaiveForecaster


def test_mape_zero_for_perfect_prediction():
    y = np.array([10, 20, 30], dtype=float)
    assert mape(y, y) == 0.0
    assert all_metrics(y, y)["RMSE"] == 0.0


def test_rmse_known_value():
    # constant error of 2 -> RMSE is 2
    assert rmse([0, 0, 0, 0], [2, 2, 2, 2]) == 2.0


def test_metrics_keys():
    m = all_metrics([1, 2, 3], [1, 2, 4])
    assert set(m) == {"MAE", "RMSE", "MAPE", "sMAPE"}


def test_train_test_split_sizes():
    idx = pd.date_range("2022-01-01", periods=50, freq="D")
    df = pd.DataFrame({"sales": range(50)}, index=idx)
    train, test = train_test_split(df, horizon=10)
    assert len(train) == 40 and len(test) == 10


def test_backtest_seasonal_naive_runs():
    idx = pd.date_range("2022-01-01", periods=120, freq="D")
    sales = 100 + 10 * np.asarray((idx.dayofweek >= 5), dtype=int)
    df = pd.DataFrame({"sales": sales}, index=idx)
    res = backtest(df, SeasonalNaiveForecaster(period=7), horizon=14)
    assert len(res.y_pred) == 14
    assert res.metrics["MAPE"] >= 0
