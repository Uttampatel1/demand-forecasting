"""Run the full model comparison and write results + a forecast plot.

Run:  python -m src.run_experiment
"""
from __future__ import annotations

import os

import pandas as pd

from .evaluate import BacktestResult, backtest
from .models import default_models

DATA_PATH = os.path.join("data", "sales.csv")
RESULTS_PATH = os.path.join("data", "results.csv")
PLOT_PATH = os.path.join("data", "forecast_vs_actual.png")


def load_sales(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def build_series(
    df: pd.DataFrame,
    store: str | None = None,
    product: str | None = None,
) -> pd.DataFrame:
    """Return a date-indexed frame (sales + exog) for a slice or the total."""
    sub = df
    if store is not None:
        sub = sub[sub["store"] == store]
    if product is not None:
        sub = sub[sub["product"] == product]
    agg = (
        sub.groupby("date")
        .agg(sales=("sales", "sum"), promo=("promo", "max"), holiday=("holiday", "max"))
        .sort_index()
    )
    agg = agg.asfreq("D")
    return agg


def run(horizon: int = 28) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    df = load_sales()
    series = build_series(df)  # company-wide daily demand

    results: dict[str, BacktestResult] = {}
    rows = []
    for model in default_models():
        try:
            res = backtest(series, model, horizon=horizon)
        except Exception as exc:  # noqa: BLE001 - skip a misbehaving model, keep the rest
            print(f"(skipped {model.name}: {exc})")
            continue
        results[model.name] = res
        rows.append({"model": model.name, **res.metrics})

    table = (
        pd.DataFrame(rows)
        .sort_values("MAPE")
        .reset_index(drop=True)
    )
    return table, results


def save_outputs(table: pd.DataFrame, results: dict[str, BacktestResult]) -> None:
    os.makedirs("data", exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    best_name = table.iloc[0]["model"]
    best = results[best_name]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(best.test_index, best.y_true, label="Actual", color="#222", linewidth=2)
        ax.plot(
            best.test_index, best.y_pred, label=f"Forecast ({best_name})",
            color="#d6336c", linewidth=2, linestyle="--",
        )
        ax.set_title(
            f"Forecast vs Actual — {best_name} "
            f"(MAPE {best.metrics['MAPE']}%, RMSE {best.metrics['RMSE']})"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Units sold / day")
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_PATH, dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"(plot skipped: {exc})")


def main() -> None:
    table, results = run()
    print("\n=== Model comparison (28-day holdout, company-wide demand) ===\n")
    print(table.to_string(index=False))
    save_outputs(table, results)
    best = table.iloc[0]
    print(f"\nBest model: {best['model']}  (MAPE {best['MAPE']}%, RMSE {best['RMSE']})")
    print(f"Results -> {RESULTS_PATH}\nPlot    -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
