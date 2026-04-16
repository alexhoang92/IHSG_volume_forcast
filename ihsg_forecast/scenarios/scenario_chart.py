"""
scenario_chart.py — Multi-scenario fan chart (volume + new accounts).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CHARTS_DIR

# Color scheme: BASE=gray, BULL=green, BEAR=amber, CRISIS=dark red, extras from palette
SCENARIO_COLORS = {
    "BASE":   "#555555",
    "BULL":   "#1a7d4c",
    "BEAR":   "#c06a1b",
    "CRISIS": "#a01515",
}
_EXTRA_COLORS = ["#185FA5", "#8B008B", "#008B8B", "#FF8C00", "#4B0082"]

# Scenario description shown in the legend text box
SCENARIO_LABELS = {
    "BASE": "BASE: FOMC hawkish, MSCI maintains EM status",
    "BULL": "BULL: FOMC dovish, MSCI maintains, FTSE follows",
    "BEAR": "BEAR: FOMC hawkish, MSCI downgrades Indonesia",
}

# Max log-unit deviation for CI clip — keeps exp() finite
_CI_CLIP_DELTA = 1.5


def _get_color(name: str, idx: int) -> str:
    return SCENARIO_COLORS.get(name, _EXTRA_COLORS[idx % len(_EXTRA_COLORS)])


def _clip_ci(forecast_lv, lower, upper):
    """Clip CI to [forecast ± _CI_CLIP_DELTA] in log space to avoid exp() overflow."""
    lo = np.maximum(lower, forecast_lv - _CI_CLIP_DELTA)
    hi = np.minimum(upper, forecast_lv + _CI_CLIP_DELTA)
    return lo, hi


def plot_scenario_fan_chart(
    weekly_df: pd.DataFrame,
    vol_forecasts: dict,
    acct_forecasts: dict,
    output_path: str = None,
    history_weeks: int = 16,
) -> None:
    """
    Two-panel scenario fan chart:
      - Panel A: volume (IDR Bn) with CI bands per scenario
      - Panel B: new accounts per scenario

    Degrades gracefully to single-line chart when only one scenario is present.
    CI bands are clipped to ±_CI_CLIP_DELTA log-units to prevent exp() overflow.
    """
    if output_path is None:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        output_path = os.path.join(CHARTS_DIR, "scenario_fan_chart.png")

    history = weekly_df.tail(history_weeks).copy()
    last_hist_date = pd.Timestamp(history["week_end_date"].max())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Panel A: Volume ─────────────────────────────────────────────────────
    hist_vol = np.exp(history["log_volume_adj"]) * history["trading_days"] / 1e9
    ax1.plot(history["week_end_date"], hist_vol,
             color="#2C2C2A", linewidth=2.5, label="Historical (adj.)", zorder=5)
    ax1.axvline(last_hist_date, color="#888888", linestyle="--", linewidth=1.2,
                label="Forecast start", zorder=6)

    for idx, (name, vol_df) in enumerate(vol_forecasts.items()):
        color = _get_color(name, idx)
        forecast_vol = np.exp(vol_df["forecast_log_volume"]) * 5 / 1e9
        ax1.plot(vol_df["week_end_date"], forecast_vol,
                 color=color, linewidth=2.2, linestyle="--", label=name, zorder=4)

        # Clip CI before exp() to avoid overflow
        lo_clipped, hi_clipped = _clip_ci(
            vol_df["forecast_log_volume"].values,
            vol_df["lower_ci"].values,
            vol_df["upper_ci"].values,
        )
        lower_vol = np.exp(lo_clipped) * 5 / 1e9
        upper_vol = np.exp(hi_clipped) * 5 / 1e9
        ax1.fill_between(vol_df["week_end_date"], lower_vol, upper_vol,
                         color=color, alpha=0.12, zorder=3)

    # Annotate MSCI decision date if it falls within forecast window
    msci_date = pd.Timestamp("2026-05-15")
    if vol_forecasts:
        first_df = next(iter(vol_forecasts.values()))
        fc_dates = pd.to_datetime(first_df["week_end_date"])
        if msci_date >= fc_dates.min() and msci_date <= fc_dates.max():
            ax1.axvline(msci_date, color="#9B0000", linestyle=":", linewidth=1.5, zorder=7)
            ax1.text(msci_date, ax1.get_ylim()[1] if ax1.get_ylim()[1] != 0 else 1,
                     "  MSCI\n  Review", fontsize=8, color="#9B0000",
                     va="top", ha="left", zorder=8)

    # Scenario assumption text box
    scenario_lines = [SCENARIO_LABELS.get(n, n) for n in vol_forecasts]
    if scenario_lines:
        box_text = "\n".join(scenario_lines)
        ax1.text(0.99, 0.04, box_text, transform=ax1.transAxes, fontsize=7.5,
                 verticalalignment="bottom", horizontalalignment="right",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="#CCCCCC"))

    ax1.set_title("IHSG Weekly Volume Forecast — Multi-Scenario Outlook (8-week horizon)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Volume (IDR Billion)", fontsize=10)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.tick_params(axis="x", rotation=30)

    # ── Panel B: New Accounts ────────────────────────────────────────────────
    if "new_accounts" in history.columns and not history["new_accounts"].isna().all():
        hist_accts = pd.to_numeric(history["new_accounts"], errors="coerce").fillna(0)
        ax2.bar(history["week_end_date"], hist_accts,
                color="#AAAAAA", alpha=0.55, width=4, label="Historical", zorder=2)
    ax2.axvline(last_hist_date, color="#888888", linestyle="--", linewidth=1.2, zorder=6)

    for idx, (name, acct_df) in enumerate(acct_forecasts.items()):
        if acct_df is None:
            continue
        color = _get_color(name, idx)
        ax2.plot(acct_df["week_end_date"], acct_df["forecast_new_accounts"],
                 color=color, linewidth=2.2, linestyle="--", label=name, zorder=4)

    ax2.set_title("New Brokerage Account Registrations — Scenario Forecast",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Week End Date", fontsize=10)
    ax2.set_ylabel("New Accounts", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scenario fan chart saved → {output_path}")
