# 📈 Demand / Sales Forecasting

**Business question:** *How much will we sell over the next month — and within what error?* Demand forecasts drive inventory, staffing, and cash-flow decisions; a few points of accuracy translate directly into fewer stock-outs and less dead stock.

This project forecasts daily retail demand and **rigorously compares** a naive baseline, classical statistical models, and machine learning — scored on a held-out month.

---

<!-- portfolio-visuals -->

## 🔧 How it works

![Demand / Sales Forecasting — architecture & data flow](./architecture.svg)

*End-to-end flow from input to output — see [`architecture.svg`](./architecture.svg).*

## 📊 Live dashboard

![Demand / Sales Forecasting — interactive Streamlit dashboard](./dashboard.png)

*Real screenshot of the Streamlit app on synthetic data — no API key needed. Run it with `streamlit run dashboard.py`.*

---


## Key results (28-day hold-out, company-wide daily demand)

| Model | MAPE | RMSE | MAE |
|-------|-----:|-----:|----:|
| **SARIMA** ✅ best | **3.27%** | 53.72 | 44.67 |
| Seasonal-Naive (baseline) | 3.92% | 126.24 | 58.96 |
| ETS (Holt-Winters) | 5.55% | 137.67 | 81.90 |
| LightGBM | 7.50% | 160.74 | 107.74 |

*(Numbers are reproducible from `python -m src.run_experiment`. Prophet is also wired in and is auto-included in the comparison when a CmdStan backend is available.)*

**What this says for the business:** monthly demand can be predicted to **~3% error**. At that accuracy, safety-stock buffers can be tightened materially without raising stock-out risk.

**Data-scientist's read on the result:**
- On a **smooth, aggregated** series, classical models (SARIMA/ETS) capture trend + weekly seasonality cleanly and **beat ML** — a common, honest finding.
- Always report the **seasonal-naive baseline**: here it's a strong, near-free benchmark that any model must beat to earn its keep.
- **LightGBM** isn't the winner here, but it's the right tool when you have **many SKUs, noisy series, and rich features** (price, weather, promotions) — it scales where fitting one SARIMA per series doesn't.

## Rolling-origin (walk-forward) validation — don't trust a single hold-out

One hold-out reports the error on *one* slice of history, which can be lucky. `src/backtesting.py`
re-fits each model at several cut-offs and forecasts the next horizon from each, producing an
**error distribution** instead of a single number — and the ranking changes:

| Model | MAPE (mean ± std) over 4 origins | RMSE (mean) |
|-------|:--------------------------------:|------------:|
| **SARIMA** ✅ robust best | **3.58% ± 0.99** | 64.5 |
| Ensemble (mean) | 4.36% ± 0.87 | 89.7 |
| ETS (Holt-Winters) | 5.87% ± 1.17 | 138.0 |
| Seasonal-Naive (baseline) | 7.69% ± 3.40 | 193.7 |
| LightGBM | 7.75% ± 2.13 | 158.2 |

*(Reproducible from `python -m src.run_experiment`. Expanding window by default; `window="sliding"` for fixed-length training.)*

**Why this matters:** on the single 28-day hold-out, **Seasonal-Naive looked 2nd-best (3.92%)** — but across
four re-fits it's actually one of the **worst and by far the most volatile** (±3.40). Walk-forward catches
that; a single split hides it. SARIMA wins *and* is the steadiest, which is the model you'd actually deploy.

## Demo

![Forecast vs Actual](data/forecast_vs_actual.png)

*Generated at `data/forecast_vs_actual.png` after running the experiment. Launch the interactive dashboard to explore any store/product slice and model.*

## How it works

```
synthetic daily sales ─► train/test split (last 28 days = hold-out)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  SeasonalNaive          ETS / SARIMA            LightGBM
  (baseline)        (statsmodels)        (lag + calendar features,
                                          recursive multi-step)
        └─────────────────────┼─────────────────────┘
                              ▼
                 MAPE · RMSE · MAE · sMAPE  ─►  ranked comparison + plot
```

All models implement one interface (`fit` / `predict`), so adding a model or swapping the evaluation target is trivial.

## "Within what error?" — prediction intervals

A point forecast isn't enough to set safety stock. `src/intervals.py` adds
**split-conformal prediction intervals** that wrap *any* model's point forecast in
a calibrated band with a finite-sample coverage guarantee and no distributional
assumptions — hold out the tail of history, measure residuals, pad the forecast:

```python
from src.intervals import conformal_interval
from src.models import SARIMAForecaster
iv = conformal_interval(lambda: SARIMAForecaster(), train, future, alpha=0.1)  # 90% band
iv.point, iv.lower, iv.upper
```

There's also an **`EnsembleForecaster`** that averages members (ETS + SARIMA by
default) to cut forecast-error variance.

## Tech stack

- **Data/ML:** pandas, NumPy, scikit-learn, statsmodels (ETS, SARIMA), LightGBM, Prophet (optional)
- **Models:** SeasonalNaive, ETS, SARIMA, LightGBM, **Ensemble**, Prophet (opt)
- **Uncertainty:** split-conformal prediction intervals (`src/intervals.py`)
- **Viz:** matplotlib, Plotly
- **App:** Streamlit dashboard
- **Observability:** structured logging via `src/logging_utils.py` (`LOG_LEVEL` env, per-model timing)
- **Deploy:** `Dockerfile` + `docker-compose.yml`; GitHub Actions CI runs the suite
- **Tests:** pytest (26 tests)

## Setup & run

```bash
cd 06-demand-forecasting
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data        # create synthetic data/sales.csv
python -m src.run_experiment       # train all models, print table, save plot
streamlit run dashboard.py         # interactive dashboard
pytest -q                          # run tests
```

**Optional — enable Prophet:**

```bash
pip install prophet==1.1.6
python -m cmdstanpy.install_cmdstan --compiler
# Prophet is then auto-detected and added to the comparison.
```

## Project structure

```
06-demand-forecasting/
├── dashboard.py             # Streamlit forecast explorer
├── src/
│   ├── generate_data.py     # synthetic retail demand generator
│   ├── features.py          # lag + calendar feature engineering
│   ├── models.py            # SeasonalNaive, ETS, SARIMA, LightGBM, Ensemble, Prophet
│   ├── intervals.py         # split-conformal prediction intervals
│   ├── backtesting.py       # rolling-origin (walk-forward) validation
│   ├── evaluate.py          # MAPE/RMSE/MAE/sMAPE + hold-out backtest
│   ├── logging_utils.py     # structured logging + timing
│   └── run_experiment.py    # full comparison + plot
├── notebooks/
│   └── forecasting_story.ipynb   # the analysis narrative
├── tests/                   # 26 pytest tests
├── Dockerfile               # containerised Streamlit app
├── docker-compose.yml
├── .github/workflows/ci.yml
├── requirements.txt
└── .gitignore
```

## Possible extensions

- **Hierarchical forecasting** with reconciliation across store × product.
- **Prediction intervals** (quantile LightGBM / SARIMA CIs) for safety-stock sizing.
- **Exogenous drivers**: price, weather, marketing spend.
- Automated **model selection per series** (let each SKU pick its best model).
