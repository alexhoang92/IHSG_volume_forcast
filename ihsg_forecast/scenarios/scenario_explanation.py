"""
scenario_explanation.py — Human-readable variable impact explanation notes.

Writes outputs/csv/scenarios/sensitivity_explanation.txt after each forecast run,
translating model coefficients and contributions into plain narrative.
"""

import os
import datetime
import numpy as np

from config import SCENARIOS_OUTPUT_DIR, SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER

_RULE = "─" * 60

# Plain-English descriptions for each exogenous variable.
# Each value: (label, direction_note, one_std_in_practice)
_VAR_DESCRIPTIONS = {
    "weekly_return": (
        "IHSG weekly log return",
        "Positive coefficient: rising market → higher participation → more volume",
        "1 std ≈ a ±3–5% weekly index move (typical IHSG weekly swing range)",
    ),
    "realized_volatility": (
        "Intraweek realized volatility (scaled by √trading days)",
        "Positive coefficient: higher volatility → forced rebalancing/hedging → more turnover",
        "1 std ≈ move from a calm week to an active week with ~2× daily price range",
    ),
    "macro_shock_abs": (
        "Absolute macro shock magnitude on announcement week (|shock_score|)",
        "Positive coefficient: any large event — positive OR negative — spikes volume on impact week",
        "1 std ≈ difference between a routine week and a BI rate-decision or MSCI review week",
    ),
    "macro_neg_lag1": (
        "Negative shock residual — 1 week after adverse event (panic-selling tail)",
        "Negative coefficient: fear/uncertainty from last week's adverse news suppresses volume",
        "1 std ≈ one week after a mid-sized adverse macro event (e.g., surprise rate hike)",
    ),
    "macro_neg_lag2": (
        "Negative shock residual — 2 weeks after adverse event (fear/uncertainty phase)",
        "Negative coefficient: two-week window of retail investor caution post-shock",
        "1 std ≈ lingering suppression two weeks after an MSCI downgrade warning",
    ),
    "macro_neg_lag3": (
        "Negative shock residual — 3 weeks after adverse event (structural break tail)",
        "Negative coefficient: persistent volume depression after major structural shocks",
        "1 std ≈ tail effect three weeks after a severe shock (e.g., full MSCI exclusion)",
    ),
    "macro_pos_lag1": (
        "Positive shock residual — 1 week after positive event (fund rebalancing wave)",
        "Positive coefficient: institutional rebalancing into IHSG one week after positive news",
        "1 std ≈ rebalancing week following a surprise BI rate cut or MSCI inclusion signal",
    ),
    "macro_pos_lag2": (
        "Positive shock residual — 2 weeks after positive event (sustained buying)",
        "Positive coefficient: sustained net inflows two weeks after a positive macro catalyst",
        "1 std ≈ ongoing buying pressure two weeks after an FTSE EM inclusion announcement",
    ),
    "macro_pos_lag3": (
        "Positive shock residual — 3 weeks after positive event (MSCI effective-date wave)",
        "Positive coefficient: index-tracking funds completing allocation three weeks out",
        "1 std ≈ MSCI effective-date settlement week after a positive review",
    ),
    "interest_rate_direction": (
        "BI policy rate direction (4-week net change in basis points)",
        "Positive coefficient: rate cuts ease liquidity → volume uplift; rate hikes suppress volume",
        "1 std ≈ a 25 bps BI policy step sustained over 4 weeks",
    ),
    "d_geo": (
        "Geopolitical event week dummy (1 = geopolitical shock week, 0 = otherwise)",
        "Positive coefficient: geopolitical shocks tend to spike short-term trading activity",
        "1 std ≈ proportion of event weeks in the sample; binary so std < 1",
    ),
    "d_mp": (
        "Monetary policy event week dummy (1 = BI/FOMC decision week, 0 = otherwise)",
        "Positive coefficient: policy announcement weeks generate elevated positioning activity",
        "1 std ≈ a BI Board of Governors meeting week vs. a non-event week",
    ),
    "d_trade": (
        "Trade policy event week dummy (1 = major trade policy announcement, 0 = otherwise)",
        "Mixed coefficient: tariff/trade announcements generate uncertainty-driven volume spikes",
        "1 std ≈ a week with a US tariff announcement or WTO ruling affecting Indonesia",
    ),
    "log_trading_days": (
        "Log of number of trading days in the week (calendar swing adjustment)",
        "Positive coefficient: more trading days → higher measured weekly volume (mechanical effect)",
        "1 std ≈ difference between a 4-day week (1 public holiday) and a full 5-day week",
    ),
}

_FALLBACK_DESC = ("(no description available)", "direction unknown", "N/A")


def _display_name(var_key: str) -> str:
    return _VAR_DESCRIPTIONS.get(var_key, _FALLBACK_DESC)[0]


def _vol_pct(log_impact: float) -> float:
    return (np.exp(log_impact) - 1) * 100


def save_sensitivity_explanation(
    sensitivity_df,
    contribution_by_scenario,
    model_notes,
    output_path: str = None,
) -> None:
    """
    Writes a plain-text explanation file summarising variable impacts.

    Parameters
    ----------
    sensitivity_df           : pd.DataFrame | None  (output of compute_sensitivity_analysis)
    contribution_by_scenario : dict | None          (scenario → contrib DataFrame)
    model_notes              : dict | None          (m1_aic, m1_bic, backtest_mape, ...)
    output_path              : str | None           (override default path; for testing)
    """
    os.makedirs(SCENARIOS_OUTPUT_DIR, exist_ok=True)
    path = output_path or os.path.join(SCENARIOS_OUTPUT_DIR, "sensitivity_explanation.txt")

    mn = model_notes or {}
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    run_date = datetime.date.today().isoformat()
    aic = mn.get("m1_aic")
    bic = mn.get("m1_bic")
    spec = f"SARIMAX{SARIMAX_ORDER}{SARIMAX_SEASONAL_ORDER}"
    fit_str = f"  AIC={aic:.1f}  BIC={bic:.1f}" if aic is not None else ""

    lines += [
        "IHSG Forecast — Variable Sensitivity & Impact Explanation",
        f"Generated : {run_date}",
        f"Model     : {spec}{fit_str}",
        f"Horizon   : 8 weeks forward",
        _RULE,
    ]

    # ── How to read ───────────────────────────────────────────────────────────
    lines += [
        "HOW TO READ THIS FILE",
        "",
        "  The SARIMAX model forecasts log(avg daily trading volume). A positive",
        "  coefficient means the variable boosts predicted log-volume — i.e. higher",
        "  weekly IDR trading value. A negative coefficient suppresses it.",
        "",
        "  Sensitivity metric:",
        "    1-std log impact = coefficient × historical std of the variable",
        "    % volume impact  = (exp(1-std log impact) − 1) × 100",
        "",
        "  Example: an impact of +0.05 log-units ≈ +5.1% more weekly trading volume.",
        "",
        "  Contribution (Section 3): average contribution of each variable over the",
        "  8-week forecast window, in log-volume units. Positive = volume-boosting.",
        _RULE,
    ]

    # ── Section 1: Variable Impact Dictionary ─────────────────────────────────
    lines.append("SECTION 1: VARIABLE IMPACT DICTIONARY")
    lines.append("  (ordered by sensitivity rank — most impactful first)")
    lines.append("")

    if sensitivity_df is None or len(sensitivity_df) == 0:
        lines.append("  Not available — sensitivity analysis was not run.")
    else:
        for rank, (_, row) in enumerate(sensitivity_df.iterrows(), 1):
            var = row["variable"]
            label, direction, in_practice = _VAR_DESCRIPTIONS.get(var, _FALLBACK_DESC)
            coef = row["coefficient"]
            std = row["historical_std"]
            log_imp = row["1std_log_impact"]
            pct_imp = row["1std_vol_pct"]
            sign = "+" if log_imp >= 0 else ""

            lines += [
                f"  [{rank:>2}] {var}",
                f"       Label      : {label}",
                f"       Direction  : {direction}",
                f"       Coefficient: {coef:+.4f}  |  Hist std: {std:.4f}",
                f"       1-std impact: {sign}{log_imp:.4f} log-units  ({sign}{pct_imp:.2f}% weekly volume)",
                f"       In practice : {in_practice}",
                "",
            ]

    lines.append(_RULE)

    # ── Section 2: Sensitivity Ranking Table ──────────────────────────────────
    lines.append("SECTION 2: SENSITIVITY RANKING")
    lines.append("  Variables sorted by absolute 1-std log impact (most sensitive first)")
    lines.append("")

    if sensitivity_df is None or len(sensitivity_df) == 0:
        lines.append("  Not available — sensitivity analysis was not run.")
    else:
        header = f"  {'Rank':<5} {'Variable':<26} {'Coef':>8} {'Hist Std':>10} {'1σ Impact':>11} {'% Vol':>9}"
        lines.append(header)
        lines.append("  " + "─" * (len(header) - 2))
        for rank, (_, row) in enumerate(sensitivity_df.iterrows(), 1):
            var = row["variable"]
            coef = row["coefficient"]
            std = row["historical_std"]
            log_imp = row["1std_log_impact"]
            pct = row["1std_vol_pct"]
            lines.append(
                f"  {rank:<5} {var:<26} {coef:>+8.4f} {std:>10.4f} "
                f"{log_imp:>+11.4f} {pct:>+9.2f}%"
            )

    lines.append(_RULE)

    # ── Section 3: Per-Scenario Narrative ─────────────────────────────────────
    lines.append("SECTION 3: PER-SCENARIO NARRATIVE")
    lines.append("")

    if not contribution_by_scenario:
        lines.append("  Not available — contribution analysis was not run.")
    else:
        contrib_cols = []
        first_df = next(iter(contribution_by_scenario.values()))
        contrib_cols = [
            c for c in first_df.columns
            if c.startswith("contrib_") and c not in ("contrib_intercept", "contrib_arma")
        ]

        # Compute averages per scenario
        scenario_avgs = {}
        scenario_arma = {}
        scenario_log_vol = {}
        for sname, cdf in contribution_by_scenario.items():
            scenario_avgs[sname] = {c: cdf[c].mean() for c in contrib_cols if c in cdf.columns}
            scenario_arma[sname] = cdf["contrib_arma"].mean() if "contrib_arma" in cdf.columns else 0.0
            scenario_log_vol[sname] = cdf["forecast_log_volume"].mean() if "forecast_log_volume" in cdf.columns else 0.0

        base_avgs = scenario_avgs.get("BASE")
        base_log_vol = scenario_log_vol.get("BASE", 0.0)

        for sname, avgs in scenario_avgs.items():
            arma_avg = scenario_arma[sname]
            avg_log_vol = scenario_log_vol[sname]
            avg_idr_bn = np.exp(avg_log_vol) * 5 / 1e9

            # Top positive and negative drivers by average contribution
            sorted_avgs = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
            top_pos = [(k, v) for k, v in sorted_avgs if v > 0.0001][:2]
            top_neg = [(k, v) for k, v in reversed(sorted_avgs) if v < -0.0001][:2]
            top_driver_key, top_driver_val = sorted_avgs[0] if sorted_avgs else ("(none)", 0.0)
            top_driver_name = _display_name(top_driver_key.replace("contrib_", ""))

            lines.append(f"  {sname} SCENARIO")
            lines.append(f"  Average weekly volume forecast: ~{avg_idr_bn:.0f} IDR Bn")
            lines.append("")

            if sname == "BASE":
                lines.append(
                    f"  This is the neutral reference forecast. The dominant exogenous driver"
                )
                lines.append(
                    f"  is {top_driver_name}, contributing {top_driver_val:+.4f} log-units on"
                )
                lines.append(
                    f"  average ({_vol_pct(top_driver_val):+.1f}% of weekly volume). The ARMA"
                )
                lines.append(
                    f"  component carries {arma_avg:+.4f} log-units, reflecting momentum from"
                )
                lines.append(f"  recent historical volume levels (mean-reversion dynamics).")

                if top_pos:
                    pos_parts = ", ".join(
                        f"{_display_name(k.replace('contrib_', ''))} ({v:+.4f})"
                        for k, v in top_pos
                    )
                    lines.append(f"  Volume boosters : {pos_parts}")
                if top_neg:
                    neg_parts = ", ".join(
                        f"{_display_name(k.replace('contrib_', ''))} ({v:+.4f})"
                        for k, v in top_neg
                    )
                    lines.append(f"  Volume suppressors: {neg_parts}")

            else:
                # Delta vs BASE
                if base_avgs is not None:
                    deltas = {k: avgs.get(k, 0.0) - base_avgs.get(k, 0.0) for k in contrib_cols}
                    sorted_deltas = sorted(deltas.items(), key=lambda x: abs(x[1]), reverse=True)
                    top_diffs = [(k, v) for k, v in sorted_deltas if abs(v) > 0.0001][:3]

                    vs_base_pct = (np.exp(avg_log_vol) - np.exp(base_log_vol)) / np.exp(base_log_vol) * 100
                    sign = "+" if vs_base_pct >= 0 else ""
                    lines.append(
                        f"  This scenario is {sign}{vs_base_pct:.1f}% vs BASE on average weekly volume."
                    )
                    lines.append(f"  Key differentiators vs BASE (avg delta over 8 weeks):")
                    for k, v in top_diffs:
                        vname = _display_name(k.replace("contrib_", ""))
                        sign2 = "+" if v >= 0 else ""
                        lines.append(
                            f"    • {vname}: {sign2}{v:.4f} log-units ({sign2}{_vol_pct(v):.1f}% vol)"
                        )
                else:
                    lines.append("  (No BASE scenario available for comparison.)")

                lines.append(
                    f"  The ARMA component contributes {arma_avg:+.4f} log-units (momentum/mean-reversion)."
                )
                if top_pos:
                    pos_parts = ", ".join(
                        f"{_display_name(k.replace('contrib_', ''))} ({v:+.4f})"
                        for k, v in top_pos
                    )
                    lines.append(f"  Volume boosters : {pos_parts}")
                if top_neg:
                    neg_parts = ", ".join(
                        f"{_display_name(k.replace('contrib_', ''))} ({v:+.4f})"
                        for k, v in top_neg
                    )
                    lines.append(f"  Volume suppressors: {neg_parts}")

            lines.append("")

    lines.append(_RULE)

    # ── Model Fit Notes ───────────────────────────────────────────────────────
    lines.append("MODEL FIT NOTES")
    lines.append("")
    if aic is not None:
        lines.append(f"  AIC: {aic:.1f}   BIC: {bic:.1f}")
        lines.append(
            "  (Lower AIC/BIC = better model fit. AIC penalises complexity less than BIC.)"
        )
    mapes = mn.get("backtest_mape")
    if mapes:
        mape_parts = []
        for i, m in enumerate(mapes, 1):
            mape_parts.append(f"Cycle {i}: {m:.1f}%" if m is not None else f"Cycle {i}: N/A")
        lines.append(f"  Backtest MAPE — " + "  |  ".join(mape_parts))
        lines.append(
            "  (MAPE > 20% over an 8-week horizon is common for high-volatility markets like IHSG.)"
        )
    lines.append("")
    lines += [
        _RULE,
        "For machine-readable data see:",
        "  sensitivity_ranking.csv           — full variable sensitivity table",
        "  forecast_contribution_analysis.csv — per-week contributions by scenario",
        "  forecast_summary_table.csv         — wide-format scenario comparison + notes",
        _RULE,
    ]

    # ── Write file ────────────────────────────────────────────────────────────
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Sensitivity explanation saved → {path}")
