"""
scenario_output.py — Save per-scenario forecast CSVs and a combined CSV.
"""

import os
import pandas as pd

from config import FORECAST_PATH, SCENARIOS_OUTPUT_DIR


def save_scenario_forecasts(
    vol_forecasts: dict,
    acct_forecasts: dict,
) -> None:
    """
    Saves:
      - outputs/csv/scenarios/forecast_{SCENARIO}.csv  (per scenario)
      - outputs/csv/scenarios/forecast_all_scenarios.csv  (combined)
      - outputs/csv/forecast_forward.csv  (BASE scenario for backward compat)
    """
    os.makedirs(SCENARIOS_OUTPUT_DIR, exist_ok=True)

    all_rows = []

    for scenario_name in vol_forecasts:
        vol_df  = vol_forecasts[scenario_name].copy()
        acct_df = acct_forecasts.get(scenario_name)

        # Merge volume + accounts
        if acct_df is not None:
            out = vol_df.merge(acct_df[["week_end_date", "forecast_new_accounts",
                                         "baseline_new_accounts", "ipo_contribution_accounts"]],
                               on="week_end_date", how="left")
        else:
            out = vol_df.copy()
            out["forecast_new_accounts"]     = 0
            out["baseline_new_accounts"]     = 0
            out["ipo_contribution_accounts"] = 0

        out["forecast_volume_idr_bn"] = out["forecast_volume"] / 1e9

        cols = [
            "week_end_date", "forecast_log_volume", "lower_ci", "upper_ci",
            "forecast_volume_idr_bn", "forecast_new_accounts",
            "baseline_new_accounts", "ipo_contribution_accounts",
        ]
        out_cols = [c for c in cols if c in out.columns]
        per_scenario = out[out_cols]

        # Save per-scenario CSV
        path = os.path.join(SCENARIOS_OUTPUT_DIR, f"forecast_{scenario_name}.csv")
        per_scenario.to_csv(path, index=False)
        print(f"  Scenario '{scenario_name}' saved → {path}")

        # Accumulate for combined CSV
        combined_row = per_scenario.copy()
        combined_row.insert(0, "scenario", scenario_name)
        all_rows.append(combined_row)

    # Save combined CSV
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined_path = os.path.join(SCENARIOS_OUTPUT_DIR, "forecast_all_scenarios.csv")
        combined.to_csv(combined_path, index=False)
        print(f"  Combined scenarios saved → {combined_path} ({len(combined)} rows)")

    # Backward-compatible forecast_forward.csv (always BASE, or first scenario)
    base_name = "BASE" if "BASE" in vol_forecasts else list(vol_forecasts.keys())[0]
    if base_name != "BASE":
        print(f"  NOTE: No BASE scenario found; forecast_forward.csv contains the '{base_name}' scenario.")

    vol_base  = vol_forecasts[base_name].copy()
    acct_base = acct_forecasts.get(base_name)
    if acct_base is not None:
        fwd = vol_base.merge(acct_base[["week_end_date", "forecast_new_accounts"]],
                             on="week_end_date", how="left")
    else:
        fwd = vol_base.copy()
        fwd["forecast_new_accounts"] = 0
    fwd["forecast_volume_idr_bn"] = fwd["forecast_volume"] / 1e9
    fwd[[
        "week_end_date", "forecast_log_volume", "lower_ci", "upper_ci",
        "forecast_volume_idr_bn", "forecast_new_accounts",
    ]].to_csv(FORECAST_PATH, index=False)
    print(f"  forecast_forward.csv (backward compat) saved → {FORECAST_PATH}")
