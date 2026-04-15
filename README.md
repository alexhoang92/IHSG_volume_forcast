# IHSG Volume & New Account Forecast

An econometric forecasting pipeline for the Indonesian equity market (IDX Composite / IHSG). Forecasts **weekly market trading volume 6–8 weeks ahead** using SARIMAX and **new brokerage account registrations** using a Negative Binomial model. A 3-cycle rolling backtest validates both models over the last 6 months.

## Models

| Model | Type | Target | Key Inputs |
|---|---|---|---|
| Model 1 | SARIMAX(1,1,1)(1,0,1,52) | `log_volume_adj` (per-day avg) | weekly return, realised volatility, macro shock score, BI rate direction, event dummies, `log_trading_days` |
| Model 2 | Negative Binomial GLM | `new_accounts` | Model 1 volume forecast, volume momentum, lagged volumes, return signals |

### Trading Day Adjustment

Weekly volume is normalised to a per-trading-day basis (`log_volume_adj = log(volume / trading_days)`) so holiday-shortened weeks (e.g. Eid al-Fitr) are not misread as low-activity weeks. `log_trading_days` is included as an exogenous regressor in both models. All AR features (`volume_momentum`, `lag_lv_1..4`) are derived from the adjusted series. Forecast output is converted back to total weekly IDR volume.

## Quick Start

```bash
cd ihsg_forecast
pip install -r requirements.txt
python main.py
```

See [RUNNING.md](RUNNING.md) for full instructions, macro input format, CLI options, and output file reference.

## Output

```
outputs/csv/forecast_forward.csv          — 8-week forward forecast
outputs/csv/backtest_results.csv          — week-by-week backtest (3 cycles)
outputs/reports/backtest_summary.txt      — MAE / RMSE / MAPE / direction acc / CI coverage
outputs/charts/backtest_volume_forecast.png
outputs/charts/backtest_volume_levels.png
outputs/charts/backtest_new_accounts.png
outputs/charts/backtest_error_distribution.png
```

## Data Sources

- **IHSG OHLCV:** Yahoo Finance (`^JKSE`), fetched automatically
- **Macro inputs:** User-provided `data/macro/macro_shocks.csv` (BI policy rate, event scores, optional new account counts) — pipeline runs with defaults if absent
