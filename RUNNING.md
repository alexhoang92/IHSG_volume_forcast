# IHSG Volume & User Acquisition Forecast — Running Instructions

## Overview

This pipeline forecasts **weekly IHSG (IDX Composite) market trading volume 6–8 weeks ahead** using a SARIMAX model, and **new brokerage account registrations** using a Negative Binomial model. A 3-cycle rolling backtest validates both models over the last 6 months.

---

## 1. Prerequisites

### Python Version
Python 3.9 or higher is required.

### Install Dependencies

```bash
cd ihsg_forecast
pip install -r requirements.txt
```

Packages installed (pinned versions):

| Package | Version | Purpose |
|---|---|---|
| yfinance | 0.2.40 | IHSG OHLCV data from Yahoo Finance |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computation |
| statsmodels | 0.14.2 | SARIMAX + Negative Binomial models |
| scipy | 1.13.0 | Statistical utilities |
| matplotlib | 3.9.0 | Chart generation |
| seaborn | 0.13.2 | Chart styling |
| scikit-learn | 1.5.0 | Evaluation utilities |
| python-dateutil | 2.9.0 | Date handling |
| openpyxl | 3.1.2 | Excel output support |
| tabulate | 0.9.0 | Backtest summary table formatting |

---

## 2. User Input Required — Data Files

**These steps are required before running the full pipeline.**

### Step 2.0 — Provide IDR volume data (required)

The pipeline no longer uses yfinance for volume. It reads daily IDR-value trading volume from a user-provided file:

```
data/raw/ihsg_volume.csv
```

**Required columns:**

| Column | Type | Description |
|---|---|---|
| `Date` | Date (YYYY-MM-DD) | Trading date |
| `volume` | Integer | Daily total trading value in IDR (e.g. `13612384062538`) |

The file must cover at least from `FETCH_START` (default `2023-01-01`) to the current date. Rows with missing dates are ignored (they will not appear in the merged output). The pipeline will **fail with a FileNotFoundError** if this file is absent.

> **Volume units:** IDR (Indonesian Rupiah) total daily transaction value — not share count. All forecast outputs (`forecast_volume_idr_bn`) are derived directly from this source.

---

### Step 2.1 — Macro Shock File

**This step is required before running the full pipeline.**

The pipeline uses a user-provided macro shock file to capture qualitative events (BI rate decisions, geopolitical events, trade shocks) that affect market volume.

### Step 2.1a — Copy the template

```bash
cp data/macro/macro_shocks_TEMPLATE.csv data/macro/macro_shocks.csv
```

### Step 2.1b — Fill in the CSV

Open `data/macro/macro_shocks.csv` and fill in **one row per week** from `2023-01-06` to today.

**Column reference:**

| Column | Type | Required | Description |
|---|---|---|---|
| `week_end_date` | Date (YYYY-MM-DD) | Yes | Friday date of the week |
| `event_description` | Text | Yes | Brief note (use "No major event" for quiet weeks) |
| `shock_score` | Float | Yes | Macro sentiment score (see guide below) |
| `event_type` | Text | No | `geopolitical` / `monetary_policy` / `trade` / `corporate` / blank |
| `policy_rate` | Float | Yes | Bank Indonesia benchmark rate that week (e.g. `6.00`) |
| `new_accounts` | Integer | No | Actual weekly new brokerage registrations (enables real Model 2) |

**Shock Score Guide:**

| Score | Meaning | Example |
|---|---|---|
| `+5.0` | Structural break — major positive | MSCI index confirmation / upgrade, IMF bailout |
| `+3.0` | Large positive | BI surprise rate cut, sovereign upgrade |
| `+2.0` | Major positive shock | Large FDI announcement, current account surplus beat |
| `+1.5` | Moderate positive | Better-than-expected GDP, election win with continuity |
| `+1.0` | Mild positive | Stable inflation print, positive trade balance |
| `+0.5` | Slight positive | Minor reform announcement |
| `0.0` | Neutral / no event | Most weeks |
| `-0.5` | Slight negative | Minor USD/IDR pressure |
| `-1.0` | Mild negative | Tariff threat, FII outflow week |
| `-1.5` | Moderate negative | Political uncertainty, BI surprise hold |
| `-2.0` | Major negative shock | Sudden rate hike, geopolitical escalation, crisis |
| `-3.0` | Large negative | Sovereign downgrade, flash crash |
| `-5.0` | Structural break — major negative | MSCI downgrade to Frontier, global financial crisis |

> **How the model uses shock scores — asymmetric lag structure:**
>
> The model decomposes `shock_score` into three distinct effects rather than using the raw signed value:
>
> | Variable | Formula | Effect |
> |---|---|---|
> | `macro_shock_abs` | `\|score\|` | **Announcement week:** both positive and negative events drive volume up (euphoric buying or panic selling) |
> | `macro_neg_lag1/2` | `max(0, −score)` lagged 1–2 wks | **Negative aftermath:** fear and uncertainty suppress volume in the 1–2 weeks following a negative event |
> | `macro_pos_lag1/2` | `max(0, +score)` lagged 1–2 wks | **Positive momentum:** fund rebalancing and continued buying sustain elevated volume 1–2 weeks after a positive event |
>
> This means: a score of `+5.0` (MSCI confirmation) drives a ~63% volume spike on announcement week and ~70% sustained uplift the following week. A score of `−5.0` (MSCI downgrade) drives an equivalent spike on announcement week but then depresses volume ~13% for the next two weeks. Use ±3–5 for structural break events where the market impact is visibly larger than typical events — the model will linearly scale the effect.

> **Note:** The pipeline runs end-to-end **even without this file** — all macro variables default to zero with a warning message. However, forecast quality improves significantly with accurate macro input.

---

## 3. Running the Pipeline

All commands must be run from inside the `ihsg_forecast/` directory:

```bash
cd ihsg_forecast
```

### Full Pipeline (recommended first run)

```bash
python main.py
```

Runs all 10 steps:
1. Fetch IHSG daily **price** (OHLC) from Yahoo Finance (incremental — appends new days only) and merge with `data/raw/ihsg_volume.csv` for IDR volume
2. Load macro shock input
3. Compute 10 weekly input variables
4. Fit Model 1 (SARIMAX)
5. Fit Model 2 (Negative Binomial)
6. Generate 8-week forward forecast
7. Save forecast CSV
8. Run 3-cycle rolling backtest
9. Generate 4 backtest charts
10. Save backtest summary report

**Expected runtime:** 3–8 minutes (mostly SARIMAX fitting × 4 cycles)

---

### Skip Data Fetch (use cached raw data)

```bash
python main.py --skip-fetch
```

Use this when you have already fetched data and want to re-run models without a new Yahoo Finance download.

---

### Backtest Only (re-run backtest from existing processed data)

```bash
python main.py --backtest-only
```

Loads `data/processed/weekly_variables.csv` directly and re-runs the 3-cycle backtest. Useful for quickly regenerating charts and reports after parameter changes in `config.py`.

> Requires a prior full pipeline run to have generated `weekly_variables.csv`.

---

### Forecast Only (skip backtest)

```bash
python main.py --forecast-only
```

Runs the full pipeline through model fitting and forward forecast, but skips the backtest. Faster if you only need the 8-week forward forecast.

---

## 4. Output Files

After a full pipeline run, the following files are generated:

```
ihsg_forecast/
├── data/
│   ├── raw/
│   │   ├── ihsg_daily_ohlcv.csv          ← merged daily price+volume (auto-generated)
│   │   └── ihsg_volume.csv               ← USER-PROVIDED daily IDR volume (required)
│   ├── macro/
│   │   ├── macro_shocks.csv              ← USER-PROVIDED macro shock input
│   │   ├── macro_shocks_TEMPLATE.csv     ← blank template
│   │   └── scenarios.csv                 ← USER-PROVIDED scenario definitions (BASE/BULL/BEAR)
│   ├── ipo/
│   │   └── ipo_calendar.csv              ← USER-PROVIDED IPO announcement dates
│   └── processed/
│       ├── weekly_variables.csv          ← computed weekly input variables
│       └── formula_notes.csv             ← formula reference for each variable
└── outputs/
    ├── csv/
    │   ├── forecast_forward.csv          ← BASE scenario 8-week forecast (backward compat)
    │   ├── backtest_results.csv          ← week-by-week backtest detail (3 cycles)
    │   └── scenarios/
    │       ├── forecast_BASE.csv         ← BASE scenario forecast
    │       ├── forecast_BULL.csv         ← BULL scenario forecast
    │       ├── forecast_BEAR.csv         ← BEAR scenario forecast
    │       ├── forecast_all_scenarios.csv ← all scenarios combined (one row per week/scenario)
    │       └── forecast_summary_table.csv ← wide-format pivot: all scenarios side-by-side + model notes
    ├── reports/
    │   └── backtest_summary.txt          ← metrics table per cycle
    └── charts/
        ├── scenario_fan_chart.png        ← 8-week forecast fan chart (all scenarios + CI)
        ├── backtest_combined.png         ← combined backtest chart (volume + new accounts, 3 cycles)
        ├── backtest_volume_forecast.png  ← log volume: actual vs forecast per cycle
        ├── backtest_volume_levels.png    ← volume in IDR billion per cycle
        ├── backtest_new_accounts.png     ← new accounts: actual vs forecast per cycle
        ├── backtest_error_distribution.png ← MAPE distribution per cycle
        └── ipo_effect_analysis.png       ← IPO uplift analysis chart
```

### Forward Forecast CSV columns

| Column | Description |
|---|---|
| `week_end_date` | Friday of the forecast week |
| `forecast_log_volume` | Log of avg daily IDR volume forecast (trading-day-adjusted) |
| `lower_ci` | 95% CI lower bound (log scale — clipped to ±1.5 log-units; treat as indicative) |
| `upper_ci` | 95% CI upper bound (log scale — clipped to ±1.5 log-units; treat as indicative) |
| `forecast_volume_idr_bn` | Forecast total weekly volume in IDR billion (`exp(forecast_log_volume) × 5 / 1e9`) |
| `forecast_new_accounts` | Forecast weekly new brokerage registrations |

> The **wide summary table** (`forecast_summary_table.csv`) shows all three scenarios side by side with model diagnostics appended as `# NOTE:` comment lines at the bottom.

---

## 5. Backtest Design

The backtest uses a **rolling expanding-window** approach over the last 6 months:

```
Full history:  |═══════════════════════════════════════|
                                       ↑ today - 6 months

Cycle 1:  Train: all data up to t-6
          Test:  t-6 → t-4  (~8–9 weeks)

Cycle 2:  Train: all data up to t-4
          Test:  t-4 → t-2  (~8–9 weeks)

Cycle 3:  Train: all data up to t-2
          Test:  t-2 → t    (~8–9 weeks)
```

**Metrics computed per cycle:**

| Metric | Applied to |
|---|---|
| MAE (log vol) | `log_volume_adj` (trading-day-adjusted) |
| RMSE (log vol) | `log_volume_adj` |
| MAPE (volume %) | total weekly volume (IDR) — recovered as `exp(log_volume_adj) × trading_days` |
| Direction accuracy | weekly volume changes |
| 95% CI coverage | `log_volume_adj` |
| MAE (new accounts) | new_accounts |

---

## 6. Configuration

All parameters are in `config.py`. Edit these to adjust model behaviour:

```python
TICKER               = "^JKSE"           # Yahoo Finance ticker
FETCH_START          = "2023-01-01"      # data history start
FORECAST_WEEKS       = 8                 # forward forecast horizon
SARIMAX_ORDER        = (1, 1, 1)         # ARIMA (p, d, q)
SARIMAX_SEASONAL_ORDER = (1, 0, 1, 52)  # Seasonal (P, D, Q, S)
BACKTEST_MONTHS      = 6                 # total backtest window
CYCLE_MONTHS         = 2                 # each cycle length
MACRO_SHOCK_MAX      = 5.0               # clip ceiling for shock scores (±2 typical; ±5 structural breaks)
```

After changing `config.py`, re-run with:
```bash
python main.py --skip-fetch
```

---

## 7. Troubleshooting

| Issue | Solution |
|---|---|
| `FileNotFoundError: ihsg_volume.csv` | The IDR volume file is missing. Place `ihsg_volume.csv` with `Date` and `volume` columns in `data/raw/`. |
| `ValueError: Insufficient data` | Check internet connection; Yahoo Finance may be throttling. Try again in a few minutes. |
| `WARNING: macro_shocks.csv not found` | Expected if you haven't created the macro file. All macro variables will be 0. |
| `WARNING: SARIMAX default optimizer failed` | Pipeline automatically retries with `method='nm'`. No action needed. |
| `WARNING: NegativeBinomial failed` | Pipeline falls back to Poisson GLM. No action needed. |
| Forecast volume values look too small (~2 IDR Bn) | Volume source has reverted to yfinance share counts. Ensure `ihsg_volume.csv` is present and covers the full date range. Expected values are ~100,000–300,000 IDR Bn/week. |
| Charts not generated | Check `outputs/charts/` directory exists. Run `python main.py --backtest-only` to regenerate. |
| `ModuleNotFoundError` | Ensure you are running from the `ihsg_forecast/` directory and have run `pip install -r requirements.txt`. |

---

## 8. Model Notes

### Model 1 — SARIMAX Volume Forecast
- **Target:** `log_volume_adj` — log of *average daily IDR volume* (`log(weekly_IDR_volume / trading_days)`), which removes the calendar swing caused by weeks with fewer trading days (public holidays)
- **Exogenous variables:**

| Variable | Description |
|---|---|
| `weekly_return` | Log return of IHSG index |
| `realized_volatility` | Std dev of daily returns × √n_days |
| `macro_shock_abs` | `\|shock_score\|` — announcement-week spike (both positive and negative events drive volume up) |
| `macro_neg_lag1` | `max(0, −score)` lagged 1 wk — residual elevated volume 1 week after negative event |
| `macro_neg_lag2` | `max(0, −score)` lagged 2 wks — volume suppression 2 weeks after negative event |
| `macro_pos_lag1` | `max(0, +score)` lagged 1 wk — sustained buying 1 week after positive event |
| `macro_pos_lag2` | `max(0, +score)` lagged 2 wks — continued uplift 2 weeks after positive event |
| `interest_rate_direction` | 4-week change in BI policy rate |
| `d_geo`, `d_mp`, `d_trade` | Event-type dummies |
| `log_trading_days` | Residual calendar effects |

- **Shock decomposition rationale:** Both positive and negative macro events cause a volume spike on the announcement week (euphoric buying or fire-selling). The direction of the event only matters in subsequent weeks: negative shocks depress volume (fear/uncertainty) while positive shocks sustain elevated volume (fund rebalancing, continued buying). The raw signed `macro_shock_score` is retained in `weekly_variables.csv` for reference but is not used directly as a model input.
- **Forward forecast** assumes 5 trading days/week by default; back-transforms to total IDR weekly volume via `exp(forecast) × 5 / 1e9` = IDR Billion
- **Output is back-transformed** to IDR billion for display; expected range ~100,000–400,000 IDR Bn/week
- **Minimum data:** 104 weeks (2 years) for reliable seasonal coefficient estimation; 56 weeks absolute minimum

### Model 2 — New Account Forecast
- **Target:** `new_accounts` (weekly new brokerage registrations)
- **Inputs:** Model 1 forecast volume (`log_volume_adj`, mean-centred) + `log_trading_days`, lagged volume, return, momentum signals, IPO dummies
- **Mean-centring:** all `lag_lv_*` and `log_volume_adj` features are centred on the training mean before fitting and forecasting. This is required for numerical stability (log of IDR values is ~31, causing near-singular design matrix without centring).
- **Demo mode:** If no `new_accounts` column is provided, a synthetic series is generated (clearly labelled in all outputs)
- **Minimum data:** 160 rows for Negative Binomial convergence (10 events-per-parameter rule); falls back to Poisson at ~130 rows

### Trading Day Adjustment
Raw weekly volume sums over however many days the exchange was open that week. A 2-day holiday week will show ~60% of a normal week's volume — not because markets were quiet, but because of the calendar. The adjustment `log_volume_adj = log(volume / trading_days)` normalises all weeks to a per-day basis so the model compares like-for-like. The AR features (`volume_momentum`, `lag_lv_1..4`) also use `log_volume_adj` so holiday weeks don't propagate distortions through the autoregressive structure.
