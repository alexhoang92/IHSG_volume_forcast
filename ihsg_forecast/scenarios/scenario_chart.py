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


def _get_color(name: str, idx: int) -> str:
    return SCENARIO_COLORS.get(name, _EXTRA_COLORS[idx % len(_EXTRA_COLORS)])


def plot_scenario_fan_chart(
    weekly_df: pd.DataFrame,
    vol_forecasts: dict,
    acct_forecasts: dict,
    output_path: str = None,
    history_weeks: int = 12,
) -> None:
    """
    Two-panel scenario fan chart:
      - Panel A: volume (IDR Bn) with CI bands per scenario
      - Panel B: new accounts per scenario

    Degrades gracefully to single-line chart when only one scenario is present.
    """
    if output_path is None:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        output_path = os.path.join(CHARTS_DIR, "scenario_fan_chart.png")

    history = weekly_df.tail(history_weeks).copy()
    last_hist_date = pd.Timestamp(history["week_end_date"].max())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    plt.style.use("seaborn-v0_8-whitegrid")

    # ── Panel A: Volume ─────────────────────────────────────────────────────
    # Historical anchor
    hist_vol = np.exp(history["log_volume_adj"]) * history["trading_days"] / 1e9
    ax1.plot(history["week_end_date"], hist_vol,
             color="#2C2C2A", linewidth=2, label="Historical", zorder=5)
    ax1.axvline(last_hist_date, color="#999999", linestyle=":", linewidth=1)

    for idx, (name, vol_df) in enumerate(vol_forecasts.items()):
        color = _get_color(name, idx)
        forecast_vol = np.exp(vol_df["forecast_log_volume"]) * 5 / 1e9  # 5 trading days
        ax1.plot(vol_df["week_end_date"], forecast_vol,
                 color=color, linewidth=2, linestyle="--", label=name, zorder=4)
        lower_vol = np.exp(vol_df["lower_ci"]) * 5 / 1e9
        upper_vol = np.exp(vol_df["upper_ci"]) * 5 / 1e9
        ax1.fill_between(vol_df["week_end_date"], lower_vol, upper_vol,
                         color=color, alpha=0.08, zorder=3)

    ax1.set_title("IHSG Weekly Volume Forecast — Multi-Scenario Outlook (8-week horizon)")
    ax1.set_ylabel("Volume (IDR Billion)")
    ax1.legend(fontsize=9, loc="upper left")

    # ── Panel B: New Accounts ────────────────────────────────────────────────
    # Historical new_accounts if available
    if "new_accounts" in history.columns and not history["new_accounts"].isna().all():
        ax2.bar(history["week_end_date"], history["new_accounts"].fillna(0),
                color="#AAAAAA", alpha=0.5, width=4, label="Historical", zorder=2)
    ax2.axvline(last_hist_date, color="#999999", linestyle=":", linewidth=1)

    for idx, (name, acct_df) in enumerate(acct_forecasts.items()):
        if acct_df is None:
            continue
        color = _get_color(name, idx)
        ax2.plot(acct_df["week_end_date"], acct_df["forecast_new_accounts"],
                 color=color, linewidth=2, linestyle="--", label=name, zorder=4)

    ax2.set_title("New Brokerage Account Registrations — Scenario Forecast")
    ax2.set_xlabel("Week End Date")
    ax2.set_ylabel("New Accounts")
    ax2.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Scenario fan chart saved → {output_path}")
