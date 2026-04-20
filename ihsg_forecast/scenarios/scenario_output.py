"""
scenario_output.py — Save per-scenario forecast CSVs and a combined CSV.
"""

import os
import numpy as np
import pandas as pd

from config import FORECAST_PATH, SCENARIOS_OUTPUT_DIR

# Maximum deviation (in log units) allowed for CI before clipping.
# Prevents exp() overflow from numerically unstable SARIMAX prediction intervals.
_CI_CLIP_DELTA = 1.5


def _clip_ci(forecast_lv: pd.Series, lower: pd.Series, upper: pd.Series):
    """
    Clip CI columns to [forecast ± _CI_CLIP_DELTA] in log space.
    Returns (lower_clipped, upper_clipped).
    """
    lo = np.maximum(lower, forecast_lv - _CI_CLIP_DELTA)
    hi = np.minimum(upper, forecast_lv + _CI_CLIP_DELTA)
    return lo, hi


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


def save_forecast_summary_table(
    vol_forecasts: dict,
    acct_forecasts: dict,
    model_notes: dict = None,
) -> None:
    """
    Saves a human-readable wide-format summary table:
        outputs/csv/forecast_summary_table.csv

    Columns (one row per forecast week):
        week_end_date,
        BASE_vol_idr_bn, BASE_vol_lower_95, BASE_vol_upper_95, BASE_new_accounts,
        BULL_vol_idr_bn, BULL_new_accounts,
        BEAR_vol_idr_bn, BEAR_new_accounts

    CI only for BASE — BULL/BEAR show point forecasts only.
    Clipped CI (±1.5 log-units) is used so values are meaningful.

    Appends # NOTE: comment lines at the bottom explaining model fit and data issues.

    Parameters
    ----------
    vol_forecasts   : dict  scenario_name → volume forecast DataFrame
    acct_forecasts  : dict  scenario_name → accounts forecast DataFrame
    model_notes     : dict  optional metadata keys:
                            m1_aic, m1_bic,
                            m2_model_type, m2_pseudo_r2,
                            ipo_warning (bool),
                            backtest_mape (list of 3 floats, or None per cycle),
                            new_accounts_nan_count (int)
    """
    if model_notes is None:
        model_notes = {}

    os.makedirs(SCENARIOS_OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(SCENARIOS_OUTPUT_DIR, "forecast_summary_table.csv")

    # ── Build BASE frame (with CI) ─────────────────────────────────────────────
    base_name = "BASE" if "BASE" in vol_forecasts else list(vol_forecasts.keys())[0]
    base_vol  = vol_forecasts[base_name].copy()
    base_acct = acct_forecasts.get(base_name)

    lo_clipped, hi_clipped = _clip_ci(
        base_vol["forecast_log_volume"],
        base_vol["lower_ci"],
        base_vol["upper_ci"],
    )
    table = pd.DataFrame({
        "week_end_date":     base_vol["week_end_date"],
        "BASE_vol_idr_bn":   (np.exp(base_vol["forecast_log_volume"]) * 5 / 1e9).round(2),
        "BASE_vol_lower_95": (np.exp(lo_clipped) * 5 / 1e9).round(2),
        "BASE_vol_upper_95": (np.exp(hi_clipped) * 5 / 1e9).round(2),
        "BASE_new_accounts": (base_acct["forecast_new_accounts"].astype(int)
                              if base_acct is not None else 0),
    })

    # ── BULL ──────────────────────────────────────────────────────────────────
    if "BULL" in vol_forecasts:
        bull_vol  = vol_forecasts["BULL"]
        bull_acct = acct_forecasts.get("BULL")
        table["BULL_vol_idr_bn"]   = (np.exp(bull_vol["forecast_log_volume"]) * 5 / 1e9).round(2).values
        table["BULL_new_accounts"] = (bull_acct["forecast_new_accounts"].astype(int).values
                                      if bull_acct is not None else 0)

    # ── BEAR ──────────────────────────────────────────────────────────────────
    if "BEAR" in vol_forecasts:
        bear_vol  = vol_forecasts["BEAR"]
        bear_acct = acct_forecasts.get("BEAR")
        table["BEAR_vol_idr_bn"]   = (np.exp(bear_vol["forecast_log_volume"]) * 5 / 1e9).round(2).values
        table["BEAR_new_accounts"] = (bear_acct["forecast_new_accounts"].astype(int).values
                                      if bear_acct is not None else 0)

    # ── Write CSV rows ─────────────────────────────────────────────────────────
    table.to_csv(summary_path, index=False)

    # ── Append # NOTE: lines ──────────────────────────────────────────────────
    notes = _build_notes(model_notes)
    with open(summary_path, "a") as f:
        f.write("\n")
        for line in notes:
            f.write(f"# {line}\n")

    print(f"  Forecast summary table saved → {summary_path} ({len(table)} forecast weeks)")


def save_contribution_analysis(
    contribution_by_scenario: dict,
    sensitivity_df,
) -> None:
    """
    Saves:
      - outputs/csv/scenarios/forecast_contribution_analysis.csv
          All scenarios stacked; one row per forecast week with per-variable
          contributions (in log-volume units) and ARMA residual component.
      - outputs/csv/scenarios/sensitivity_ranking.csv
          One row per exog variable: coefficient, historical std, and
          approximate % volume impact of a 1-std-dev change.
    """
    os.makedirs(SCENARIOS_OUTPUT_DIR, exist_ok=True)

    # ── Contribution analysis (stacked across scenarios) ──────────────────
    if contribution_by_scenario:
        stacked = []
        for scenario_name, contrib_df in contribution_by_scenario.items():
            row = contrib_df.copy()
            row.insert(0, "scenario", scenario_name)
            stacked.append(row)
        contrib_all = pd.concat(stacked, ignore_index=True)
        contrib_path = os.path.join(SCENARIOS_OUTPUT_DIR, "forecast_contribution_analysis.csv")
        contrib_all.to_csv(contrib_path, index=False)
        print(f"  Contribution analysis saved → {contrib_path} ({len(contrib_all)} rows)")

    # ── Sensitivity ranking ───────────────────────────────────────────────
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        sens_path = os.path.join(SCENARIOS_OUTPUT_DIR, "sensitivity_ranking.csv")
        sensitivity_df.to_csv(sens_path, index=False)
        print(f"  Sensitivity ranking saved → {sens_path} ({len(sensitivity_df)} variables)")


def _build_notes(mn: dict) -> list:
    """Build a list of plain note strings from the model_notes dict."""
    lines = []
    lines.append("─" * 60)
    lines.append("MODEL & DATA NOTES")
    lines.append("─" * 60)

    # Model 1
    aic = mn.get("m1_aic")
    bic = mn.get("m1_bic")
    if aic is not None:
        lines.append(
            f"Model 1 (Volume): SARIMAX(1,1,1)(1,0,1,52)  "
            f"AIC={aic:.1f}  BIC={bic:.1f}"
        )
    else:
        lines.append("Model 1 (Volume): SARIMAX(1,1,1)(1,0,1,52)")

    # Model 2
    m2_type = mn.get("m2_model_type", "Poisson GLM (NegBinomial fallback)")
    m2_r2   = mn.get("m2_pseudo_r2")
    if m2_r2 is not None:
        lines.append(
            f"Model 2 (New Accounts): {m2_type}  Pseudo R²={m2_r2:.4f}"
        )
    else:
        lines.append(f"Model 2 (New Accounts): {m2_type}")

    # CI caveat
    lines.append(
        "Confidence intervals (BASE_vol_lower_95 / BASE_vol_upper_95) are clipped "
        f"to forecast ±exp({_CI_CLIP_DELTA}) due to SARIMAX prediction-interval "
        "numerical instability over an 8-week horizon. Treat as indicative range only."
    )

    # IPO warning
    if mn.get("ipo_warning"):
        lines.append(
            "IPO WARNING: Negative IPO coefficients detected in Model 2. "
            "This suggests announcement_date values in ipo_calendar.csv may be "
            "IDX listing dates rather than book-open/subscription-start dates. "
            "Shift dates ~3–4 weeks earlier to the actual investor registration window "
            "for correct IPO uplift estimation."
        )

    # new_accounts NaN
    nan_cnt = mn.get("new_accounts_nan_count", 0)
    if nan_cnt and nan_cnt > 0:
        lines.append(
            f"Data note: {nan_cnt} week(s) in macro_shocks.csv had blank new_accounts "
            "and were excluded from Model 2 fitting."
        )

    # Backtest MAPE
    mapes = mn.get("backtest_mape")
    if mapes:
        parts = []
        for i, m in enumerate(mapes, 1):
            parts.append(f"Cycle {i}: {m:.1f}%" if m is not None else f"Cycle {i}: N/A")
        lines.append("Backtest MAPE (volume) — " + "  |  ".join(parts))

    # ── Sensitivity ranking (top 5 variables by 1-std impact) ─────────────
    sensitivity_df = mn.get("sensitivity_df")
    if sensitivity_df is not None and len(sensitivity_df) > 0:
        lines.append("─" * 60)
        lines.append("SENSITIVITY RANKING — impact of 1 std dev change on weekly volume")
        lines.append("  (log-volume units; vol_pct = approx % change in weekly IDR volume)")
        top5 = sensitivity_df.head(5)
        for rank, (_, r) in enumerate(top5.iterrows(), 1):
            sign = "+" if r["1std_log_impact"] >= 0 else ""
            lines.append(
                f"  {rank}. {r['variable']:<25} "
                f"coef={r['coefficient']:+.4f}  "
                f"1σ={r['historical_std']:.4f}  "
                f"→ {sign}{r['1std_log_impact']:.4f} log-units "
                f"({sign}{r['1std_vol_pct']:.1f}% vol)"
            )

    # ── Per-scenario contribution summary ─────────────────────────────────
    contrib_map = mn.get("contribution_by_scenario")
    if contrib_map:
        contrib_cols = [c for c in next(iter(contrib_map.values())).columns
                        if c.startswith("contrib_") and c not in ("contrib_intercept", "contrib_arma")]

        lines.append("─" * 60)
        lines.append("VARIABLE CONTRIBUTION SUMMARY (avg over 8-week forecast, log-volume units)")
        lines.append("  Positive = volume-boosting; Negative = volume-suppressing")

        base_avgs = None
        for scenario_name, contrib_df in contrib_map.items():
            avgs = {c: contrib_df[c].mean() for c in contrib_cols if c in contrib_df.columns}
            arma_avg = contrib_df["contrib_arma"].mean() if "contrib_arma" in contrib_df.columns else 0.0
            lines.append(f"  {scenario_name}:")

            # Top 3 positive and top 3 negative drivers
            sorted_avgs = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
            top_pos = [(k, v) for k, v in sorted_avgs if v > 0.0001][:3]
            top_neg = [(k, v) for k, v in reversed(sorted_avgs) if v < -0.0001][:3]

            if top_pos:
                pos_str = "  ".join(
                    f"{k.replace('contrib_', '')}={v:+.4f}" for k, v in top_pos
                )
                lines.append(f"    Boosters:    {pos_str}")
            if top_neg:
                neg_str = "  ".join(
                    f"{k.replace('contrib_', '')}={v:+.4f}" for k, v in top_neg
                )
                lines.append(f"    Suppressors: {neg_str}")
            lines.append(f"    ARMA component (avg): {arma_avg:+.4f}")

            if scenario_name == "BASE":
                base_avgs = avgs

        # ── Scenario differentiators vs BASE ──────────────────────────────
        if base_avgs is not None:
            for scenario_name, contrib_df in contrib_map.items():
                if scenario_name == "BASE":
                    continue
                avgs = {c: contrib_df[c].mean() for c in contrib_cols if c in contrib_df.columns}
                deltas = {k: avgs.get(k, 0) - base_avgs.get(k, 0) for k in contrib_cols}
                sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
                meaningful = [(k, v) for k, v in sorted_deltas if abs(v) > 0.0001][:5]
                if meaningful:
                    lines.append(f"  {scenario_name} vs BASE — key differentiators (avg delta):")
                    for k, v in meaningful:
                        sign = "+" if v >= 0 else ""
                        vol_pct = (np.exp(v) - 1) * 100
                        lines.append(
                            f"    {k.replace('contrib_', ''):<25} {sign}{v:.4f} log-units "
                            f"({sign}{vol_pct:.1f}% vol)"
                        )

    lines.append("─" * 60)
    return lines
