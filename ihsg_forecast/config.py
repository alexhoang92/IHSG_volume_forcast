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
RAW_DATA_PATH         = "data/raw/ihsg_daily_ohlcv.csv"
MACRO_INPUT_PATH      = "data/macro/macro_shocks.csv"      # user uploads this
MACRO_TEMPLATE_PATH   = "data/macro/macro_shocks_TEMPLATE.csv"
PROCESSED_DATA_PATH   = "data/processed/weekly_variables.csv"
FORMULA_NOTES_PATH    = "data/processed/formula_notes.csv"
FORECAST_PATH         = "outputs/csv/forecast_forward.csv"
BACKTEST_RESULTS_PATH = "outputs/csv/backtest_results.csv"
REPORT_PATH           = "outputs/reports/backtest_summary.txt"
CHARTS_DIR            = "outputs/charts"

# ── IPO calendar ──────────────────────────────────────────────────────────────
IPO_INPUT_PATH           = "data/ipo/ipo_calendar.csv"
IPO_TEMPLATE_PATH        = "data/ipo/ipo_calendar_TEMPLATE.csv"
# Weeks after announcement date to flag as follow-on subscription/allotment effect
IPO_EFFECT_WEEKS_AFTER   = 2
# Min market cap (IDR trillion) to qualify as a large IPO; set 0 to treat all equally
IPO_MIN_MARKET_CAP_IDR_T = 1.0

# ── Scenario forecasts ────────────────────────────────────────────────────────
SCENARIOS_INPUT_PATH  = "data/macro/scenarios.csv"
SCENARIOS_OUTPUT_DIR  = "outputs/csv/scenarios"
IPO_IMPACT_PATH       = "outputs/csv/ipo_impact_analysis.csv"
