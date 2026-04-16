# IHSG Volume & New Account Forecast

An econometric forecasting pipeline for the Indonesian equity market (IDX Composite / IHSG). Forecasts **weekly market trading volume 6–8 weeks ahead** using SARIMAX and **new brokerage account registrations** using a Negative Binomial model. A 3-cycle rolling backtest validates both models over the last 6 months.

## Models

| Model | Type | Target | Key Inputs |
|---|---|---|---|
| Model 1 | SARIMAX(1,1,1)(1,0,1,52) | `log_volume_adj` (per-day avg) | weekly return, realised volatility, macro shock decomposition (abs + neg/pos lags ×3), BI rate direction, event dummies, `log_trading_days` |
| Model 2 | Negative Binomial GLM | `new_accounts` | Model 1 volume forecast, volume momentum, lagged volumes, return signals |

### Macro Shock Decomposition

Both positive and negative macro events spike volume on the announcement week (euphoric buying or panic selling). The model decomposes `shock_score` into seven distinct exogenous terms rather than using the raw signed value:

| Variable | Formula | Effect |
|---|---|---|
| `macro_shock_abs` | `\|score\|` | Announcement-week spike — any event type drives volume up |
| `macro_neg_lag1` | `max(0, −score)` lag 1 wk | Residual panic selling week +1 |
| `macro_neg_lag2` | `max(0, −score)` lag 2 wks | Fear/uncertainty suppresses volume week +2 |
| `macro_neg_lag3` | `max(0, −score)` lag 3 wks | Structural break tail (MSCI-scale events show −7% at week +3) |
| `macro_pos_lag1` | `max(0, +score)` lag 1 wk | Fund rebalancing first wave |
| `macro_pos_lag2` | `max(0, +score)` lag 2 wks | Sustained buying week +2 |
| `macro_pos_lag3` | `max(0, +score)` lag 3 wks | MSCI effective-date second wave (passive funds execute ~2 wks after announcement) |

Lag window calibrated from event study on 19 historical shock events; see [RUNNING.md](RUNNING.md) for details.

### Trading Day Adjustment

Weekly volume is normalised to a per-trading-day basis (`log_volume_adj = log(volume / trading_days)`) so holiday-shortened weeks (e.g. Eid al-Fitr) are not misread as low-activity weeks. `log_trading_days` is included as an exogenous regressor in both models. All AR features (`volume_momentum`, `lag_lv_1..4`) are derived from the adjusted series. Forecast output is converted back to total weekly IDR volume.

## Quick Start

```bash
cd ihsg_forecast
pip install -r requirements.txt
python main.py
```

See [RUNNING.md](RUNNING.md) for full instructions, macro input format, CLI options, and output file reference.

## Example Results (run: 2026-04-16)

### Backtest Performance — 3-cycle rolling expanding window

| Metric | Cycle 1 (Oct–Dec 25) | Cycle 2 (Dec 25–Feb 26) | Cycle 3 (Feb–Apr 26) |
|---|---|---|---|
| MAPE (volume) | **13.8%** | 23.0% | 33.3% |
| MAE (log vol) | 0.131 | 0.220 | 0.272 |
| Direction accuracy | 62% | 62% | 50% |
| 95% CI coverage | 100% | 100% | 78% |

Cycle 3 MAPE is elevated because it covers the Jan–Mar 2026 MSCI re-ranking period — an extreme structural break event with ±80% actual volume swings that no weekly model can fully anticipate out-of-sample.

### 8-Week Forward Forecast — BASE / BULL / BEAR Scenarios (from 2026-04-24)

BULL = MSCI confirms Indonesia as Emerging Market (+5 shock on 15-May)  
BEAR = MSCI downgrades Indonesia to Frontier Market (−5 shock on 15-May)

| Week end | BASE (IDR bn) | BULL (IDR bn) | BEAR (IDR bn) | BULL−BEAR spread |
|---|---|---|---|---|
| 24-Apr-26 | 209,258 | 209,258 | 209,258 | — |
| 01-May-26 | 215,276 | 215,276 | 215,276 | — |
| 08-May-26 | 202,947 | 213,442 | 202,947 | +5% |
| **15-May-26** | **237,034** | **324,072** | **319,861** | **+1%** ← both spike |
| **22-May-26** | **248,758** | **396,156** | **234,549** | **+65%** ← diverge |
| 29-May-26 | 192,443 | 209,018 | 165,800 | +23% |
| 05-Jun-26 | 206,374 | 220,031 | 175,623 | +22% |
| 12-Jun-26 | 197,310 | 202,300 | 194,396 | +4% |

Both scenarios spike on announcement week (abs shock effect). The spread opens the following week via the directional lag structure: BULL gets `pos_lag1 × 5 → +78%` uplift; BEAR gets `neg_lag1 × 5 → +7%` then `neg_lag2 × 5 → −17%` suppression.

Charts and CSV outputs from this run are committed to [`example_outputs/`](example_outputs/) as a reference snapshot. Live pipeline outputs are written to `ihsg_forecast/outputs/` (gitignored during normal runs).

## Output Files

After a pipeline run, files are written to `ihsg_forecast/outputs/` (gitignored). A reference snapshot from the 2026-04-16 run is committed to [`example_outputs/`](example_outputs/):

```
example_outputs/
├── csv/
│   ├── forecast_forward.csv               — BASE scenario 8-week forecast
│   ├── backtest_results.csv               — week-by-week backtest detail (3 cycles)
│   ├── ipo_impact_analysis.csv            — IPO uplift per event
│   └── scenarios/
│       ├── forecast_BASE.csv
│       ├── forecast_BULL.csv
│       ├── forecast_BEAR.csv
│       ├── forecast_all_scenarios.csv     — all scenarios combined
│       └── forecast_summary_table.csv     — wide pivot: all scenarios side-by-side
├── reports/
│   └── backtest_summary.txt               — metrics table per cycle
├── charts/
│   ├── scenario_fan_chart.png             — 8-week forecast fan (all scenarios + CI)
│   ├── backtest_combined.png              — volume + new accounts, 3 cycles
│   ├── backtest_volume_forecast.png       — log volume: actual vs forecast
│   ├── backtest_volume_levels.png         — volume in IDR billion per cycle
│   ├── backtest_new_accounts.png          — new accounts: actual vs forecast
│   ├── backtest_error_distribution.png    — MAPE distribution per cycle
│   └── ipo_effect_analysis.png            — IPO uplift analysis
└── processed/
    ├── weekly_variables.csv               — computed weekly input variables
    └── formula_notes.csv                  — formula reference for each variable
```

## Data Sources

- **IHSG OHLCV:** Yahoo Finance (`^JKSE`), fetched automatically
- **IDR Volume:** User-provided `data/raw/ihsg_volume.csv` (daily IDR transaction value)
- **Macro inputs:** User-provided `data/macro/macro_shocks.csv` (BI policy rate, event scores ±2 typical / ±5 structural breaks, optional new account counts)
- **Scenarios:** User-provided `data/macro/scenarios.csv` (BASE / BULL / BEAR paths)
