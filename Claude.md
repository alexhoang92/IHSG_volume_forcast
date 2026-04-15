# IHSG Market Volume & New User Forecasting — Claude Code Instructions

## Project Overview

Build a two-model econometric forecasting pipeline for the Indonesian equity market (IHSG / IDX Composite). The system forecasts **weekly market trading volume 6–8 weeks ahead** (Model 1 — SARIMAX) and **new brokerage account registrations** driven by that volume (Model 2 — Negative Binomial). A rolling backtest validates both models over the last 6 months using 3 cycles of 2-month windows.

---

## Project Structure

Create the following directory layout before writing any code:

```
ihsg_forecast/
├── data/
│   ├── raw/                  # fetched OHLCV data saved here
│   ├── macro/                # user-uploaded qualitative macro input CSV
│   └── processed/            # computed input variables saved here
├── models/
│   ├── model1_volume.py      # SARIMAX volume forecast
│   └── model2_users.py       # Negative Binomial new user forecast
├── backtest/
│   └── backtest_engine.py    # rolling 2-month window backtest
├── outputs/
│   ├── csv/                  # all result CSVs
│   ├── charts/               # backtest visualisation plots
│   └── reports/              # summary backtest report
├── config.py                 # all parameters in one place
├── fetch_data.py             # data fetching module
├── compute_variables.py      # variable computation module
├── main.py                   # single entry point to run everything
└── requirements.txt          # pinned dependencies
```

---

## Step 0 — Dependencies

### `requirements.txt`
```
yfinance==0.2.40
pandas==2.2.2
numpy==1.26.4
statsmodels==0.14.2
scipy==1.13.0
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.5.0
python-dateutil==2.9.0
openpyxl==3.1.2
tabulate==0.9.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Step 1 — Configuration

### `config.py`

Define all parameters here. No magic numbers anywhere else in the codebase.

```python
# ── Data fetch ────────────────────────────────────────────────────────────────
TICKER          = "^JKSE"          # Yahoo Finance ticker for IHSG
FETCH_START     = "2023-01-01"     # historical fetch start
FETCH_END       = None             # None = today

# ── Modelling ─────────────────────────────────────────────────────────────────
FORECAST_WEEKS  = 8                # Model 1 horizon
SARIMAX_ORDER   = (1, 1, 1)        # (p, d, q)
SARIMAX_SEASONAL_ORDER = (1, 0, 1, 52)   # (P, D, Q, S)
MAX_LAGS        = 4                # max lags for volume in Model 2

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_MONTHS = 6                # total backtest window
CYCLE_MONTHS    = 2                # each cycle length in months
# This yields 3 cycles:
#   Cycle 1: months t-6 to t-4  (train on all prior, test on these 2 months)
#   Cycle 2: months t-4 to t-2
#   Cycle 3: months t-2 to t

# ── File paths ────────────────────────────────────────────────────────────────
RAW_DATA_PATH        = "data/raw/ihsg_daily_ohlcv.csv"
MACRO_INPUT_PATH     = "data/macro/macro_shocks.csv"   # user uploads this
PROCESSED_DATA_PATH  = "data/processed/weekly_variables.csv"
BACKTEST_RESULTS_PATH = "outputs/csv/backtest_results.csv"
REPORT_PATH          = "outputs/reports/backtest_summary.txt"
```

---

## Step 2 — Data Fetching

### `fetch_data.py`

Fetch daily OHLCV for IHSG from Yahoo Finance using `yfinance`. Save raw data to CSV immediately after fetch so it can be inspected and re-used without re-fetching.

**Requirements:**
- Use `yfinance.download(TICKER, start=FETCH_START, auto_adjust=True)`
- Drop rows where `Close` is NaN
- Reset index so `Date` is a column, not the index
- Rename columns to lowercase: `date, open, high, low, close, volume`
- Save to `data/raw/ihsg_daily_ohlcv.csv` with `index=False`
- Print a confirmation: total rows fetched, date range, any missing dates detected

**Function signature:**
```python
def fetch_ihsg_daily() -> pd.DataFrame:
    """
    Fetches IHSG daily OHLCV from Yahoo Finance.
    Returns cleaned daily DataFrame and saves to RAW_DATA_PATH.
    """
```

---

## Step 3 — Compute Input Variables

### `compute_variables.py`

Take the raw daily OHLCV DataFrame and produce a **weekly** DataFrame with all computed input variables. This is the core data preparation module.

### 3A — Weekly Aggregation

Resample daily OHLCV to weekly (week ending Friday) using the following rules:

| Field | Aggregation rule |
|---|---|
| `date` | Last trading day of the week (Friday or last available) |
| `open` | First open of the week |
| `high` | Max high of the week |
| `low` | Min low of the week |
| `close` | Last close of the week |
| `volume` | Sum of all daily volumes in the week |
| `trading_days` | Count of trading days in the week |

Use `pd.Grouper(freq="W-FRI")` on the date-indexed daily DataFrame.

### 3B — Compute Each Variable

Implement each variable as its own function. Compute them in order and attach to the weekly DataFrame.

---

#### Variable 1 — `index_level`
```
index_level = weekly close price (P_t)
```
- This is just the `close` column after weekly aggregation.
- No transformation needed.
- Formula note: `P_t = Friday closing price of IHSG`

---

#### Variable 2 — `weekly_return`
```
weekly_return = ln(P_t / P_{t-1})
```
- Use `numpy.log` for log returns.
- First row will be NaN — leave it as NaN, do not fill.
- Formula note: `r_t = ln(P_t / P_{t-1})` — log return ensures stationarity

---

#### Variable 3 — `log_volume`
```
log_volume = ln(weekly_volume)
```
- Apply `numpy.log` to the summed weekly volume.
- This is the **target variable** for Model 1.
- Formula note: `lv_t = ln(V_t)` — log transform stabilises variance

---

#### Variable 4 — `realized_volatility`
```
realized_volatility = std(daily_returns_within_week) × sqrt(trading_days)
```
- Compute daily returns on the raw daily DataFrame first: `r_d = ln(close_d / close_{d-1})`
- Group by ISO week, take `std()` of daily returns within each week
- Multiply by `sqrt(trading_days_that_week)` to scale to weekly
- Merge back onto the weekly DataFrame by week key
- Formula note: `σ_t = std(r_daily in week t) × √n_days` — realised weekly volatility proxy

---

#### Variable 5 — `volume_momentum`
```
volume_momentum = log_volume_t - log_volume_{t-1}
```
- First-difference of `log_volume`.
- First row will be NaN.
- Formula note: `ΔV_t = lv_t − lv_{t-1}` — captures acceleration in market activity

---

#### Variable 6 — `lagged_log_volume` (lags 1 to 4)
```
lag_lv_1 = log_volume shifted 1 week
lag_lv_2 = log_volume shifted 2 weeks
lag_lv_3 = log_volume shifted 3 weeks
lag_lv_4 = log_volume shifted 4 weeks
```
- Use `pd.DataFrame.shift(k)` for each lag k ∈ {1, 2, 3, 4}.
- These are autoregressive inputs for SARIMAX and Model 2.
- Formula note: `lv_{t-k}` for k = 1…4 — used as AR terms and for CCF analysis

---

#### Variable 7 — `cumulative_4w_return`
```
cumulative_4w_return = sum(weekly_return_t, weekly_return_{t-1}, weekly_return_{t-2}, weekly_return_{t-3})
```
- Rolling 4-week sum of `weekly_return`.
- Use `pd.Series.rolling(4).sum()`.
- Formula note: `CR_t = Σ r_{t-k}, k=0..3` — captures sustained bull/bear momentum signal for new user acquisition

---

#### Variable 8 — `interest_rate_direction`
```
interest_rate_direction = policy_rate_t - policy_rate_{t-4}
```
- **This variable requires manual input** — Bank Indonesia policy rate is not available via yfinance.
- Read from the user-uploaded `macro_shocks.csv` if a `policy_rate` column is present.
- If column is absent, fill with `0` and print a warning: `"WARNING: policy_rate column not found in macro input. interest_rate_direction set to 0. Please add policy_rate column to macro_shocks.csv."`
- Formula note: `IR_t = rate_t − rate_{t-4}` — 4-week change; positive = tightening, negative = easing

---

#### Variable 9 — `macro_shock_score` (from user-uploaded file)
```
macro_shock_score ∈ {-2, -1.5, -1, -0.5, 0, +0.5, +1, +1.5, +2}
```
- Read from `data/macro/macro_shocks.csv` — this is the **user-uploaded qualitative input**.
- Expected columns in the upload: `week_end_date, event_description, shock_score, event_type`
- `week_end_date` must be parseable as a date (format: YYYY-MM-DD).
- Merge onto the weekly DataFrame by `week_end_date`.
- Fill weeks with no event as `0` (neutral).
- Validate: `shock_score` must be numeric and within [−2, +2]. Raise a clear error if not.
- Formula note: `S_t ∈ {-2…+2}` — analyst-assigned weekly macro sentiment score

---

#### Variable 10 — `event_type_dummies`
```
D_geo, D_mp, D_trade, D_corporate ∈ {0, 1}
```
- Derive from the `event_type` column in `macro_shocks.csv`.
- Expected values: `"geopolitical"`, `"monetary_policy"`, `"trade"`, `"corporate"`, `""` or NaN for no event.
- Use `pd.get_dummies()` on `event_type`, then rename columns to `d_geo`, `d_mp`, `d_trade`, `d_corporate`.
- Fill NaN as 0.
- Formula note: Binary dummies per event category — avoid dummy trap (omit one baseline category in model)

---

### 3C — Save Processed Data

After computing all variables, save two files:

**1. `data/processed/weekly_variables.csv`** — full computed variable table with columns:
```
week_end_date, index_level, weekly_return, log_volume, realized_volatility,
volume_momentum, lag_lv_1, lag_lv_2, lag_lv_3, lag_lv_4,
cumulative_4w_return, interest_rate_direction, macro_shock_score,
d_geo, d_mp, d_trade, d_corporate, trading_days
```

**2. `data/processed/formula_notes.csv`** — one row per variable with columns:
```
variable_name, formula, inputs_required, source, notes
```
Populate this programmatically by maintaining a `FORMULA_REGISTRY` dict in `compute_variables.py`:
```python
FORMULA_REGISTRY = {
    "index_level":            {"formula": "P_t = Friday close", ...},
    "weekly_return":          {"formula": "r_t = ln(P_t / P_{t-1})", ...},
    # etc.
}
```

**Function signature for the main computation pipeline:**
```python
def compute_all_variables(daily_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw daily OHLCV and macro shock input.
    Returns weekly DataFrame with all computed variables.
    Saves weekly_variables.csv and formula_notes.csv.
    """
```

---

## Step 4 — Macro Input File Format

### `data/macro/macro_shocks.csv` — user uploads this file

Provide a **template CSV** with the following exact structure. Claude Code must generate this blank template and save it to `data/macro/macro_shocks_TEMPLATE.csv` so users know exactly what to fill in:

```csv
week_end_date,event_description,shock_score,event_type,policy_rate
2024-01-05,No major event,0,,6.00
2024-01-12,BI holds rate at 6%,0.5,monetary_policy,6.00
2024-01-19,US tariff threat on EV exports,-1.0,trade,6.00
2024-01-26,No major event,0,,6.00
2024-02-02,Pemilu uncertainty spike,-1.5,geopolitical,6.00
```

**Scoring guide to include as a comment block at the top of the template:**
```
# MACRO SHOCK SCORE GUIDE
# +2.0 : Major positive shock  (e.g. BI rate cut, large FDI announcement, IMF upgrade)
# +1.5 : Moderate positive     (e.g. better-than-expected GDP, current account surplus)
# +1.0 : Mild positive         (e.g. stable inflation print, positive trade balance)
# +0.5 : Slight positive       (e.g. minor reform announcement)
#  0.0 : Neutral / no event
# -0.5 : Slight negative       (e.g. minor USD/IDR pressure)
# -1.0 : Mild negative         (e.g. tariff threat, FII outflow week)
# -1.5 : Moderate negative     (e.g. political uncertainty, BI surprise hold)
# -2.0 : Major negative shock  (e.g. sudden rate hike, geopolitical escalation, crisis)
#
# EVENT TYPE values: geopolitical | monetary_policy | trade | corporate | (leave blank if none)
# policy_rate: Bank Indonesia benchmark rate in % (e.g. 6.00) — fill every row
```

---

## Step 5 — Model 1: SARIMAX Volume Forecast

### `models/model1_volume.py`

Fit a SARIMAX model to `log_volume` with exogenous variables.

**Model specification:**
```
SARIMAX(
    endog  = log_volume,
    exog   = [weekly_return, realized_volatility, macro_shock_score,
              interest_rate_direction, d_geo, d_mp, d_trade],
    order  = SARIMAX_ORDER,          # from config.py
    seasonal_order = SARIMAX_SEASONAL_ORDER,
    enforce_stationarity  = False,
    enforce_invertibility = False
)
```

**Requirements:**
- Use `statsmodels.tsa.statespace.sarimax.SARIMAX`
- Drop rows with any NaN in endog or exog before fitting
- Store fitted model object so it can be used in backtesting
- Forecast method must accept a `steps` parameter and a future `exog` array
- For future exog in forecast: carry forward the last observed values for rate/dummies; use `0` for macro shock score (neutral assumption); use last `weekly_return` and `realized_volatility` as naive forward values
- Return forecast as a DataFrame with columns: `week_end_date, forecast_log_volume, lower_ci, upper_ci, forecast_volume` (exponentiate the log forecast for `forecast_volume`)

**Function signatures:**
```python
def fit_model1(weekly_df: pd.DataFrame) -> SARIMAXResults:
    """Fits SARIMAX on full weekly_df. Returns fitted model."""

def forecast_model1(fitted_model, weekly_df: pd.DataFrame, steps: int = 8) -> pd.DataFrame:
    """Generates steps-ahead volume forecast with 95% CI."""
```

---

## Step 6 — Model 2: New User Acquisition Forecast

### `models/model2_users.py`

Fit a Negative Binomial regression where new weekly account registrations are a function of volume and market conditions.

**Important note on data:** Real new account registration counts are not publicly available via any API. The module must:

1. **Check if `data/macro/macro_shocks.csv` contains a `new_accounts` column** — if present, use it as the Model 2 target.
2. **If absent**, generate a synthetic `new_accounts` series for demonstration purposes using the formula below, and print a clear warning:
   ```
   WARNING: 'new_accounts' column not found in macro_shocks.csv.
   Using synthetic new_accounts series for demonstration.
   Add a 'new_accounts' column with weekly registration counts to enable real Model 2 fitting.
   ```

**Synthetic new_accounts generation (for demo only):**
```python
# Base: ~5000 new accounts/week with volume and return drivers
base = 5000
new_accounts_synthetic = (
    base
    * np.exp(0.4 * (log_volume - log_volume.mean()) / log_volume.std())
    * np.exp(0.3 * cumulative_4w_return.fillna(0))
    * np.random.lognormal(0, 0.15, len(weekly_df))
).astype(int)
```

**Model specification:**
```
Negative Binomial GLM:
    endog  = new_accounts
    exog   = [const, log_volume, volume_momentum, cumulative_4w_return,
              lag_lv_1, lag_lv_2, macro_shock_score, weekly_return]
```

Use `statsmodels.discrete.discrete_model.NegativeBinomial`.

**Requirements:**
- Run a cross-correlation function (CCF) between `new_accounts` and `lag_lv_1..4` before fitting — print the lag with highest absolute correlation as the "dominant lag"
- Include only lags with |correlation| > 0.2 in the model
- Return fitted model and in-sample predictions
- Forecast: given Model 1's volume forecast, project `new_accounts` forward

**Function signatures:**
```python
def fit_model2(weekly_df: pd.DataFrame) -> NegativeBinomialResults:
    """Fits Negative Binomial on weekly_df. Returns fitted model."""

def forecast_model2(fitted_model2, volume_forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Takes Model 1 volume forecast, returns weekly new_accounts forecast."""
```

---

## Step 7 — Backtest Engine

### `backtest/backtest_engine.py`

Implement a **rolling expanding-window backtest** over the last 6 months, with 3 cycles of 2 months each.

### Backtest Design

```
Full data:  |═══════════════════════════════════|
                                  ↑
                          6 months back from today

Cycle 1:  Train: all data up to month t-6
          Test:  months t-6 to t-4  (≈ 8–9 weeks)

Cycle 2:  Train: all data up to month t-4
          Test:  months t-4 to t-2  (≈ 8–9 weeks)

Cycle 3:  Train: all data up to month t-2
          Test:  months t-2 to t    (≈ 8–9 weeks)
```

Each cycle: fit Model 1 on training window → forecast test window → compare to actuals.

### Metrics to compute per cycle

| Metric | Formula | Applied to |
|---|---|---|
| MAE | `mean(|actual - forecast|)` | log_volume, new_accounts |
| RMSE | `sqrt(mean((actual - forecast)²))` | log_volume, new_accounts |
| MAPE | `mean(|actual - forecast| / actual) × 100` | volume (levels) |
| Direction accuracy | `% weeks where sign(Δforecast) == sign(Δactual)` | volume |
| Coverage | `% actuals falling within 95% CI` | volume |

### Output per cycle

Save to `outputs/csv/backtest_results.csv` with columns:
```
cycle, week_end_date, actual_log_volume, forecast_log_volume,
lower_ci, upper_ci, actual_volume, forecast_volume,
actual_new_accounts, forecast_new_accounts,
error_log_vol, error_vol_pct, in_ci
```

### Backtest summary table

Print and save to `outputs/reports/backtest_summary.txt`:
```
╔══════════════════════════════════════════════════════════════╗
║           IHSG FORECAST MODEL — BACKTEST SUMMARY            ║
╠══════════════════════╦═══════════╦═══════════╦══════════════╣
║ Metric               ║  Cycle 1  ║  Cycle 2  ║   Cycle 3   ║
╠══════════════════════╬═══════════╬═══════════╬══════════════╣
║ MAE (log vol)        ║   x.xxx   ║   x.xxx   ║    x.xxx    ║
║ RMSE (log vol)       ║   x.xxx   ║   x.xxx   ║    x.xxx    ║
║ MAPE (volume %)      ║   x.x%    ║   x.x%    ║    x.x%     ║
║ Direction accuracy   ║   xx%     ║   xx%     ║    xx%      ║
║ 95% CI coverage      ║   xx%     ║   xx%     ║    xx%      ║
╠══════════════════════╬═══════════╬═══════════╬══════════════╣
║ MAE (new accounts)   ║   xxx     ║   xxx     ║    xxx      ║
╚══════════════════════╩═══════════╩═══════════╩══════════════╝
```

**Function signatures:**
```python
def run_backtest(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs 3-cycle rolling backtest.
    Returns combined results DataFrame.
    Saves backtest_results.csv and backtest_summary.txt.
    """

def compute_cycle_metrics(actuals: pd.Series, forecasts: pd.Series,
                           lower_ci: pd.Series, upper_ci: pd.Series) -> dict:
    """Returns dict of MAE, RMSE, MAPE, direction_acc, coverage."""
```

---

## Step 8 — Visualisations

### `backtest/backtest_engine.py` (add plotting functions) or separate `outputs/charts/`

Generate the following 4 charts and save as PNG to `outputs/charts/`:

### Chart 1 — `backtest_volume_forecast.png`
- Line chart: actual `log_volume` (solid dark line) vs forecast (dashed coloured line per cycle)
- Shaded 95% CI band for each cycle
- Vertical dashed lines separating the 3 backtest cycles
- Title: "IHSG Weekly Volume — Backtest Forecast vs Actual (3 cycles × 2 months)"
- X-axis: week_end_date | Y-axis: Log volume

### Chart 2 — `backtest_volume_levels.png`
- Same as Chart 1 but in IDR trillion (exponentiated back from log)
- Easier for business stakeholders to read

### Chart 3 — `backtest_new_accounts.png`
- Actual vs forecast new accounts per week across 3 cycles
- Bar chart for actuals, line overlay for forecast

### Chart 4 — `backtest_error_distribution.png`
- Histogram of percentage forecast errors (MAPE per week) across all 3 cycles
- Separate panels for each cycle
- Mark the mean error with a vertical line

**Use `matplotlib` and `seaborn`. Apply a clean business style:**
```python
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"actual": "#2C2C2A", "cycle1": "#185FA5", "cycle2": "#0F6E56", "cycle3": "#993C1D"}
```

---

## Step 9 — Main Entry Point

### `main.py`

Single script to run the entire pipeline. Must be runnable with:
```bash
python main.py
```

Or with arguments:
```bash
python main.py --skip-fetch        # use existing raw data (skip yfinance call)
python main.py --backtest-only     # skip model fitting, only re-run backtest
python main.py --forecast-only     # fit and forecast, skip backtest
```

**Pipeline sequence in `main.py`:**
```
1. Parse CLI arguments
2. Print banner: "IHSG VOLUME & USER ACQUISITION FORECAST PIPELINE"
3. Create all output directories if not exist
4. FETCH: fetch_ihsg_daily() → save raw CSV
5. MACRO: load macro_shocks.csv → validate columns → warn if missing
6. COMPUTE: compute_all_variables(daily_df, macro_df) → save processed CSV + formula notes
7. MODEL 1: fit_model1(weekly_df) → print AIC/BIC summary
8. MODEL 2: fit_model2(weekly_df) → print dominant lag + pseudo R²
9. FORECAST: forecast_model1 + forecast_model2 → save forecast CSV
10. BACKTEST: run_backtest(weekly_df) → save results + summary
11. CHARTS: generate all 4 charts → save PNGs
12. Print: "Pipeline complete. Outputs saved to /outputs/"
```

**Forecast output CSV** (`outputs/csv/forecast_forward.csv`):
```
week_end_date, forecast_log_volume, lower_ci, upper_ci,
forecast_volume_idr_bn, forecast_new_accounts
```

---

## Step 10 — Error Handling & Validation

Apply these rules throughout all modules:

| Situation | Behaviour |
|---|---|
| `^JKSE` fetch returns < 100 rows | Raise `ValueError` with clear message |
| `macro_shocks.csv` not found | Print warning, continue with zeros for all macro vars |
| `shock_score` outside [−2, +2] | Print warning row-by-row, clip to bounds |
| Weekly volume = 0 (market closed week) | Drop the row before modelling |
| SARIMAX fails to converge | Try with `method='nm'` fallback, print warning if used |
| Any NaN in final forecast | Raise error — do not silently output NaN forecasts |

---

## Output File Summary

When the pipeline completes, the following files must exist:

```
data/raw/ihsg_daily_ohlcv.csv          ← raw fetched OHLCV (daily)
data/macro/macro_shocks_TEMPLATE.csv   ← blank template for user input
data/processed/weekly_variables.csv   ← all 13 computed input variables (weekly)
data/processed/formula_notes.csv      ← formula reference for each variable
outputs/csv/forecast_forward.csv      ← 8-week forward forecast
outputs/csv/backtest_results.csv      ← week-by-week backtest detail (3 cycles)
outputs/reports/backtest_summary.txt  ← metrics table per cycle
outputs/charts/backtest_volume_forecast.png
outputs/charts/backtest_volume_levels.png
outputs/charts/backtest_new_accounts.png
outputs/charts/backtest_error_distribution.png
```

---

## Notes for Claude Code

- **Do not hard-code any dates** — derive all backtest window boundaries dynamically from `pd.Timestamp.today()` and `config.py` parameters.
- **All monetary values** must be in **IDR billion** in output CSVs and chart labels. Convert from raw volume (shares × price) if needed.
- **Log transform** all volume series before modelling; always back-transform for output display.
- **Macro input is optional** — the pipeline must run end-to-end even with no `macro_shocks.csv` uploaded, using zeros for all qualitative variables.
- **Print progress** at each major step using a consistent format: `[STEP N/10] Description...`
- **Model 2 is clearly labelled as demonstration** if no real `new_accounts` data is provided.
- Keep all modules independently importable — `fetch_data.py` should work without importing `models/`.
