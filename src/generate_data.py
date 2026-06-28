"""Generate a realistic synthetic retail sales dataset.

The series is built from interpretable components so the forecasting task is
genuine but reproducible:

* a slow upward **trend** (business growth),
* **weekly seasonality** (weekends differ from weekdays),
* **yearly seasonality** (festive/seasonal peaks),
* **holiday** uplifts,
* **promotions** (random multiplicative boosts),
* multiplicative **noise**.

Produces daily sales for several store/product combinations.

Run:  python -m src.generate_data
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFAULT_START = "2021-01-01"
DEFAULT_END = "2023-12-31"
STORES = ["Bangalore", "Mumbai", "Delhi"]
PRODUCTS = ["Tea", "Coffee"]


def _holiday_dates(index: pd.DatetimeIndex) -> pd.Series:
    """Approximate fixed-date holiday flags (synthetic, region-agnostic)."""
    md = index.strftime("%m-%d")
    holidays = {"01-01", "01-26", "08-15", "10-02", "12-25", "11-12"}
    return pd.Series(md.isin(holidays).astype(float), index=index)


def generate_series(
    store: str,
    product: str,
    start: str,
    end: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    index = pd.date_range(start, end, freq="D")
    n = len(index)
    t = np.arange(n)

    base = {"Tea": 220.0, "Coffee": 160.0}[product]
    store_mult = {"Bangalore": 1.2, "Mumbai": 1.0, "Delhi": 0.9}[store]

    trend = base * store_mult * (1.0 + 0.00035 * t)

    dow = index.dayofweek.to_numpy()
    weekly = 1.0 + 0.18 * (dow >= 5) - 0.05 * (dow == 2)  # weekend up, mid-week dip

    doy = index.dayofyear.to_numpy()
    yearly = 1.0 + 0.20 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    holidays = _holiday_dates(index).to_numpy()
    holiday_uplift = 1.0 + 0.45 * holidays

    promo = (rng.random(n) < 0.06).astype(float)
    promo_uplift = 1.0 + 0.30 * promo

    noise = rng.normal(1.0, 0.06, n)

    sales = trend * weekly * yearly * holiday_uplift * promo_uplift * noise
    sales = np.clip(np.round(sales), 0, None)

    return pd.DataFrame(
        {
            "date": index,
            "store": store,
            "product": product,
            "sales": sales.astype(int),
            "promo": promo.astype(int),
            "holiday": holidays.astype(int),
        }
    )


def generate_dataset(
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = [
        generate_series(store, product, start, end, rng)
        for store in STORES
        for product in PRODUCTS
    ]
    return pd.concat(frames, ignore_index=True)


def main(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    df = generate_dataset()
    path = os.path.join(data_dir, "sales.csv")
    df.to_csv(path, index=False)
    print(f"Wrote {len(df):,} rows to {path}")
    print(f"  Stores: {df['store'].nunique()} | Products: {df['product'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    return path


if __name__ == "__main__":
    main()
