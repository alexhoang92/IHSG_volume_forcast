#!/usr/bin/env python3
"""
IHSG Volume & User Acquisition Forecast Pipeline
Entry point — run from the ihsg_forecast/ directory.

Usage:
    python main.py                         # full pipeline
    python main.py --skip-fetch            # skip yfinance download, use existing raw CSV
    python main.py --backtest-only         # load processed data, re-run backtest only
    python main.py --forecast-only         # full pipeline except backtest
    python main.py --scenarios FILE        # scenario CSV (default: data/macro/scenarios.csv)
    python main.py --scenarios ""          # force neutral BASE forecast (ignore file)
    python main.py --no-ipo                # skip IPO calendar, set all IPO dummies to 0
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
    IPO_INPUT_PATH, TRAINING_START,
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
    "data/ipo",
    "models",
    "backtest",
    "scenarios",
    "outputs/csv",
    "outputs/csv/scenarios",
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
    parser.add_argument(
        "--scenarios",
        metavar="FILE",
        default=None,
        help=(
            "Path to scenario definitions CSV. "
            "Default: data/macro/scenarios.csv if present, else neutral BASE. "
            "Pass empty string '' to force neutral baseline."
        ),
    )
    parser.add_argument(
        "--no-ipo",
        action="store_true",
        help="Skip IPO calendar load; all IPO dummy columns set to 0.",
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
        from fetch_data import fetch_ihsg_incremental
        daily_df = fetch_ihsg_incremental()

    # ── STEP 2: Load macro input ───────────────────────────────────────────
    print("\n[STEP 2/10] Loading macro shock input ...")
    from compute_variables import load_macro
    macro_df = load_macro(MACRO_INPUT_PATH)

    # ── STEP 2b: Load IPO calendar ─────────────────────────────────────────
    if args.no_ipo:
        print("\n  [IPO] --no-ipo flag set; IPO dummies will be 0.")
        ipo_df = pd.DataFrame(
            columns=["announcement_date", "ticker", "company_name", "market_cap_idr_trillion"]
        )
    else:
        try:
            ipo_df = pd.read_csv(IPO_INPUT_PATH, comment="#", parse_dates=["announcement_date"])
            print(f"\n  [IPO] Loaded {len(ipo_df)} IPO records from {IPO_INPUT_PATH}")
        except FileNotFoundError:
            print(f"\n  WARNING: {IPO_INPUT_PATH} not found. IPO dummies will be 0.")
            ipo_df = pd.DataFrame(
                columns=["announcement_date", "ticker", "company_name", "market_cap_idr_trillion"]
            )

    # ── Apply training start cutoff ───────────────────────────────────────
    if TRAINING_START:
        cutoff = pd.Timestamp(TRAINING_START)
        before = len(daily_df)
        daily_df = daily_df[daily_df["date"] >= cutoff].copy()
        print(f"\n  [TRAINING_START={TRAINING_START}] Filtered {before} → {len(daily_df)} daily rows (dropped pre-{TRAINING_START} data)")

    # ── STEP 3: Compute variables ──────────────────────────────────────────
    print("\n[STEP 3/10] Computing weekly input variables ...")
    from compute_variables import compute_all_variables, compute_ipo_dummies
    weekly_df = compute_all_variables(daily_df, macro_df)
    weekly_df = compute_ipo_dummies(weekly_df, ipo_df)

    # ── STEP 4: Fit Model 1 ────────────────────────────────────────────────
    print("\n[STEP 4/10] Fitting Model 1 (SARIMAX) ...")
    from models.model1_volume import fit_model1
    fitted1 = fit_model1(weekly_df)
    _model_notes = {
        "m1_aic": getattr(fitted1, "aic", None),
        "m1_bic": getattr(fitted1, "bic", None),
    }

    # ── STEP 5: Fit Model 2 ────────────────────────────────────────────────
    print("\n[STEP 5/10] Fitting Model 2 (Negative Binomial) ...")
    from models.model2_users import fit_model2, compute_ipo_impact_analysis
    fitted2_dict, sig_lags, used_synthetic = fit_model2(weekly_df)

    # Collect Model 2 metadata for summary notes
    _result2 = fitted2_dict.get("result") if isinstance(fitted2_dict, dict) else fitted2_dict
    _pseudo_r2 = None
    if _result2 is not None and hasattr(_result2, "llf") and hasattr(_result2, "llnull") and _result2.llnull:
        _pseudo_r2 = 1 - (_result2.llf / _result2.llnull)
    _m2_type = "Poisson GLM (NegBinomial fallback)" if "poisson" in str(type(_result2)).lower() else "Negative Binomial"
    _ipo_warning = any(
        fitted2_dict.get("result").params.get(c, 0) < 0
        for c in ["ipo_announcement_week", "ipo_effect_week_1"]
        if hasattr(fitted2_dict.get("result", object()), "params")
        and c in getattr(fitted2_dict.get("result"), "params", {})
    ) if isinstance(fitted2_dict, dict) else False
    _na_count = int(weekly_df["new_accounts"].isna().sum()) if "new_accounts" in weekly_df.columns else 0
    _model_notes.update({
        "m2_model_type":        _m2_type,
        "m2_pseudo_r2":         _pseudo_r2,
        "ipo_warning":          _ipo_warning,
        "new_accounts_nan_count": _na_count,
    })

    # Run IPO impact analysis on the training period
    if not args.no_ipo and ipo_df is not None and len(ipo_df) > 0:
        compute_ipo_impact_analysis(fitted2_dict, weekly_df)

    # ── STEP 6: Generate forward forecasts (all scenarios) ─────────────────
    from scenarios.scenario_engine import get_all_scenarios
    scenario_exog_map = get_all_scenarios(
        args.scenarios, macro_df, weekly_df, steps=FORECAST_WEEKS
    )
    n_scenarios = len(scenario_exog_map)
    print(f"\n[STEP 6/10] Generating {FORECAST_WEEKS}-week forward forecast ({n_scenarios} scenario(s)) ...")

    from models.model1_volume import forecast_model1
    from models.model2_users import forecast_model2

    # IPO rows beyond today (future announcements for forecast window)
    today = pd.Timestamp.today().normalize()
    if ipo_df is not None and len(ipo_df) > 0:
        future_ipo_df = ipo_df[
            pd.to_datetime(ipo_df["announcement_date"], errors="coerce") > today
        ].copy()
    else:
        future_ipo_df = None

    all_vol_forecasts  = {}
    all_acct_forecasts = {}

    for scenario_name, future_exog in scenario_exog_map.items():
        print(f"  Forecasting scenario: {scenario_name} ...")
        vol_fc = forecast_model1(
            fitted1, weekly_df, steps=FORECAST_WEEKS, future_exog_df=future_exog
        )
        acct_fc = forecast_model2(
            fitted2_dict, sig_lags, vol_fc, weekly_df,
            future_exog_df=future_exog,
            future_ipo_df=future_ipo_df,
        )
        all_vol_forecasts[scenario_name]  = vol_fc
        all_acct_forecasts[scenario_name] = acct_fc

    # ── Contribution & sensitivity analysis for Model 1 ───────────────────
    from models.model1_volume import compute_contribution_analysis, compute_sensitivity_analysis
    contribution_by_scenario = {}
    for scenario_name, future_exog in scenario_exog_map.items():
        if future_exog is not None:
            vol_fc = all_vol_forecasts[scenario_name]
            contrib_df = compute_contribution_analysis(fitted1, future_exog, vol_fc)
            contribution_by_scenario[scenario_name] = contrib_df
    sensitivity_df = compute_sensitivity_analysis(fitted1, weekly_df)
    _model_notes["contribution_by_scenario"] = contribution_by_scenario
    _model_notes["sensitivity_df"] = sensitivity_df

    # ── STEP 7: Save forecast CSVs ─────────────────────────────────────────
    print("\n[STEP 7/10] Saving forward forecast CSV(s) ...")
    from scenarios.scenario_output import (
        save_scenario_forecasts, save_forecast_summary_table, save_contribution_analysis,
    )
    save_scenario_forecasts(all_vol_forecasts, all_acct_forecasts)
    save_forecast_summary_table(all_vol_forecasts, all_acct_forecasts, _model_notes)
    save_contribution_analysis(contribution_by_scenario, sensitivity_df)
    from scenarios.scenario_explanation import save_sensitivity_explanation
    save_sensitivity_explanation(sensitivity_df, contribution_by_scenario, _model_notes)

    # Print human-readable summary for BASE (or first) scenario
    base_name = "BASE" if "BASE" in all_vol_forecasts else list(all_vol_forecasts.keys())[0]
    vol_base  = all_vol_forecasts[base_name]
    acct_base = all_acct_forecasts[base_name]
    summary   = vol_base.merge(
        acct_base[["week_end_date", "forecast_new_accounts"]], on="week_end_date", how="left"
    )
    summary["forecast_volume_idr_bn"] = np.exp(summary["forecast_log_volume"]) * 5 / 1e9
    print(f"\n  {base_name} scenario summary:")
    print(summary[["week_end_date", "forecast_volume_idr_bn", "forecast_new_accounts"]].to_string(index=False))

    # ── Generate scenario fan chart ────────────────────────────────────────
    try:
        from scenarios.scenario_chart import plot_scenario_fan_chart
        plot_scenario_fan_chart(weekly_df, all_vol_forecasts, all_acct_forecasts)
    except Exception as e:
        print(f"  WARNING: Scenario fan chart failed: {e}")

    # ── STEP 8–11: Backtest ────────────────────────────────────────────────
    if not args.forecast_only:
        print("\n[STEP 8/10] Running 3-cycle rolling backtest ...")
        from backtest.backtest_engine import run_backtest
        bt_results = run_backtest(weekly_df)

        # Update summary table with backtest MAPE per cycle
        if bt_results is not None and len(bt_results) > 0:
            mapes = []
            for cyc in sorted(bt_results["cycle"].unique()):
                grp = bt_results[bt_results["cycle"] == cyc]
                errs = grp["error_vol_pct"].dropna()
                mapes.append(float(errs.abs().mean()) if len(errs) > 0 else None)
            _model_notes["backtest_mape"] = mapes
            save_forecast_summary_table(all_vol_forecasts, all_acct_forecasts, _model_notes)
            print("  Summary table updated with backtest MAPE.")

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
