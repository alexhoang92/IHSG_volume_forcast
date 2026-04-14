#!/usr/bin/env python3
"""
IHSG Volume & User Acquisition Forecast Pipeline
Entry point — run from the ihsg_forecast/ directory.

Usage:
    python main.py                  # full pipeline
    python main.py --skip-fetch     # skip yfinance download, use existing raw CSV
    python main.py --backtest-only  # load processed data, re-run backtest only
    python main.py --forecast-only  # full pipeline except backtest
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# Ensure the ihsg_forecast/ directory is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    RAW_DATA_PATH, MACRO_INPUT_PATH, PROCESSED_DATA_PATH,
    FORECAST_PATH, FORECAST_WEEKS,
)

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       IHSG VOLUME & USER ACQUISITION FORECAST PIPELINE       ║
╚══════════════════════════════════════════════════════════════╝
"""

DIRS_TO_CREATE = [
    "data/raw",
    "data/macro",
    "data/processed",
    "models",
    "backtest",
    "outputs/csv",
    "outputs/charts",
    "outputs/reports",
]


def _bootstrap_dirs():
    for d in DIRS_TO_CREATE:
        os.makedirs(d, exist_ok=True)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="IHSG Volume & User Acquisition Forecast Pipeline"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip yfinance download; use existing raw CSV",
    )
    parser.add_argument(
        "--backtest-only",
        action="store_true",
        help="Skip fetch/compute/model; re-run backtest from existing processed data",
    )
    parser.add_argument(
        "--forecast-only",
        action="store_true",
        help="Run full pipeline (fetch → model → forecast) but skip backtest",
    )
    return parser.parse_args()


def main():
    print(BANNER)
    args = _parse_args()
    _bootstrap_dirs()

    # ── BACKTEST-ONLY shortcut ─────────────────────────────────────────────
    if args.backtest_only:
        print("[STEP 1/10] Loading existing processed data (--backtest-only mode) ...")
        if not os.path.exists(PROCESSED_DATA_PATH):
            print(f"  ERROR: {PROCESSED_DATA_PATH} not found. Run without --backtest-only first.")
            sys.exit(1)
        weekly_df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["week_end_date"])
        print(f"  Loaded {len(weekly_df)} rows from {PROCESSED_DATA_PATH}")

        print("\n[STEP 8/10] Running backtest ...")
        from backtest.backtest_engine import run_backtest
        run_backtest(weekly_df)

        print("\nPipeline complete. Outputs saved to outputs/")
        return

    # ── STEP 1: Fetch raw data ─────────────────────────────────────────────
    print("[STEP 1/10] Fetching IHSG daily OHLCV ...")
    if args.skip_fetch and os.path.exists(RAW_DATA_PATH):
        print(f"  --skip-fetch: loading existing {RAW_DATA_PATH}")
        daily_df = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    else:
        from fetch_data import fetch_ihsg_daily
        daily_df = fetch_ihsg_daily()

    # ── STEP 2: Load macro input ───────────────────────────────────────────
    print("\n[STEP 2/10] Loading macro shock input ...")
    from compute_variables import load_macro
    macro_df = load_macro(MACRO_INPUT_PATH)

    # ── STEP 3: Compute variables ──────────────────────────────────────────
    print("\n[STEP 3/10] Computing weekly input variables ...")
    from compute_variables import compute_all_variables
    weekly_df = compute_all_variables(daily_df, macro_df)

    # ── STEP 4: Fit Model 1 ────────────────────────────────────────────────
    print("\n[STEP 4/10] Fitting Model 1 (SARIMAX) ...")
    from models.model1_volume import fit_model1
    fitted1 = fit_model1(weekly_df)

    # ── STEP 5: Fit Model 2 ────────────────────────────────────────────────
    print("\n[STEP 5/10] Fitting Model 2 (Negative Binomial) ...")
    from models.model2_users import fit_model2
    fitted2, sig_lags, used_synthetic = fit_model2(weekly_df)

    # ── STEP 6: Generate forward forecast ─────────────────────────────────
    print(f"\n[STEP 6/10] Generating {FORECAST_WEEKS}-week forward forecast ...")
    from models.model1_volume import forecast_model1
    vol_forecast = forecast_model1(fitted1, weekly_df, steps=FORECAST_WEEKS)

    from models.model2_users import forecast_model2
    acct_forecast = forecast_model2(fitted2, sig_lags, vol_forecast, weekly_df)

    # ── STEP 7: Save forecast CSV ──────────────────────────────────────────
    print("\n[STEP 7/10] Saving forward forecast CSV ...")
    forecast_out = vol_forecast.merge(acct_forecast, on="week_end_date", how="left")
    forecast_out["forecast_volume_idr_bn"] = forecast_out["forecast_volume"] / 1e9
    forecast_out[
        [
            "week_end_date",
            "forecast_log_volume",
            "lower_ci",
            "upper_ci",
            "forecast_volume_idr_bn",
            "forecast_new_accounts",
        ]
    ].to_csv(FORECAST_PATH, index=False)
    print(f"  Forward forecast saved → {FORECAST_PATH}")
    print(forecast_out[["week_end_date", "forecast_volume_idr_bn", "forecast_new_accounts"]].to_string(index=False))

    # ── STEP 8–11: Backtest ────────────────────────────────────────────────
    if not args.forecast_only:
        print("\n[STEP 8/10] Running 3-cycle rolling backtest ...")
        from backtest.backtest_engine import run_backtest
        run_backtest(weekly_df)

        print("\n[STEP 9/10] Charts generated during backtest (see outputs/charts/).")
        print("\n[STEP 10/10] Summary report generated (see outputs/reports/).")
    else:
        print("\n[STEP 8-10/10] Skipped (--forecast-only mode).")

    if used_synthetic:
        print(
            "\n  NOTE: Model 2 used SYNTHETIC new_accounts data (demo mode).\n"
            "  To use real data, add a 'new_accounts' column to data/macro/macro_shocks.csv."
        )

    print("\nPipeline complete. Outputs saved to outputs/")


if __name__ == "__main__":
    main()
