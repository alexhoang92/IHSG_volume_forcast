import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe — must be set before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from config import (
    BACKTEST_MONTHS, CYCLE_MONTHS,
    BACKTEST_RESULTS_PATH, REPORT_PATH, CHARTS_DIR,
)
from models.model1_volume import fit_model1, forecast_model1, EXOG_COLS
from models.model2_users import fit_model2, forecast_model2

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "actual":  "#2C2C2A",
    "cycle1":  "#185FA5",
    "cycle2":  "#0F6E56",
    "cycle3":  "#993C1D",
}


# ── Metric computation ────────────────────────────────────────────────────────

def compute_cycle_metrics(
    actuals: pd.Series,
    forecasts: pd.Series,
    lower_ci: pd.Series,
    upper_ci: pd.Series,
) -> dict:
    """Returns dict of MAE, RMSE, MAPE, direction_acc, coverage for log_volume."""
    a = actuals.values
    f = forecasts.values
    lo = lower_ci.values
    hi = upper_ci.values

    mae  = float(np.mean(np.abs(a - f)))
    rmse = float(np.sqrt(np.mean((a - f) ** 2)))

    # MAPE on actual volume levels
    actual_vol   = np.exp(a)
    forecast_vol = np.exp(f)
    mape = float(np.mean(np.abs(actual_vol - forecast_vol) / actual_vol) * 100)

    # Direction accuracy (sign of first-difference matches)
    if len(a) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(a)) == np.sign(np.diff(f))) * 100)
    else:
        dir_acc = float("nan")

    coverage = float(np.mean((a >= lo) & (a <= hi)) * 100)

    return {
        "mae":          mae,
        "rmse":         rmse,
        "mape":         mape,
        "direction_acc": dir_acc,
        "coverage":     coverage,
    }


def compute_accounts_mae(actuals: pd.Series, forecasts: pd.Series) -> float:
    return float(np.mean(np.abs(actuals.values - forecasts.values)))


# ── Cycle boundary helpers ────────────────────────────────────────────────────

def _get_cycle_boundaries(weekly_df: pd.DataFrame):
    """
    Derive cycle start/end indices dynamically from today's date.
    Returns list of (train_end_idx, test_end_idx) tuples — 0-based exclusive upper.
    """
    today = pd.Timestamp.today().normalize()
    backtest_start = today - pd.DateOffset(months=BACKTEST_MONTHS)   # t-6
    cut1           = today - pd.DateOffset(months=4)                  # t-4
    cut2           = today - pd.DateOffset(months=2)                  # t-2

    dates = weekly_df["week_end_date"].reset_index(drop=True)

    bt_idx   = int(dates.searchsorted(backtest_start))
    cut1_idx = int(dates.searchsorted(cut1))
    cut2_idx = int(dates.searchsorted(cut2))
    end_idx  = len(dates)

    # Protect against degenerate windows
    bt_idx   = max(bt_idx, 1)
    cut1_idx = max(cut1_idx, bt_idx + 1)
    cut2_idx = max(cut2_idx, cut1_idx + 1)

    cycles = [
        (bt_idx,   cut1_idx),   # Cycle 1: train[:bt_idx], test[bt_idx:cut1_idx]
        (cut1_idx, cut2_idx),   # Cycle 2: train[:cut1_idx], test[cut1_idx:cut2_idx]
        (cut2_idx, end_idx),    # Cycle 3: train[:cut2_idx], test[cut2_idx:]
    ]
    return cycles


# ── Main backtest runner ──────────────────────────────────────────────────────

def run_backtest(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs 3-cycle rolling expanding-window backtest.
    Returns combined results DataFrame.
    Saves backtest_results.csv and backtest_summary.txt.
    """
    cycles = _get_cycle_boundaries(weekly_df)
    all_results = []
    cycle_metrics = []

    for cycle_num, (train_end, test_end) in enumerate(cycles, start=1):
        print(f"\n  --- Backtest Cycle {cycle_num} ---")
        train_df = weekly_df.iloc[:train_end].copy()
        test_df  = weekly_df.iloc[train_end:test_end].copy()

        if len(test_df) == 0:
            print(f"  Cycle {cycle_num}: empty test window — skipping.")
            continue

        print(f"  Train: {len(train_df)} weeks | Test: {len(test_df)} weeks")
        print(f"  Test period: {test_df['week_end_date'].min().date()} → {test_df['week_end_date'].max().date()}")

        # Fit Model 1 on training data
        fitted1 = fit_model1(train_df)

        # Forecast over test window using actual exog (proper backtest methodology)
        forecast_df = forecast_model1(
            fitted1,
            train_df,
            steps=len(test_df),
            future_exog_df=test_df,
        )

        # Fit & forecast Model 2
        fitted2_dict, sig_lags, _ = fit_model2(train_df)
        acct_forecast = forecast_model2(fitted2_dict, sig_lags, forecast_df, train_df)

        # Align actuals with forecast
        # Use trading-day-adjusted log volume (model target) for metric computation
        actual_lv = test_df["log_volume_adj"].values
        forecast_lv = forecast_df["forecast_log_volume"].values
        lower = forecast_df["lower_ci"].values
        upper = forecast_df["upper_ci"].values
        # Recover total weekly volume: exp(log_volume_adj) * trading_days
        actual_vol = np.exp(actual_lv) * test_df["trading_days"].values
        forecast_vol = forecast_df["forecast_volume"].values

        # new_accounts actuals
        if "new_accounts" in test_df.columns and not test_df["new_accounts"].isna().all():
            actual_accts = test_df["new_accounts"].fillna(0).values
        else:
            actual_accts = np.zeros(len(test_df))
        forecast_accts = acct_forecast["forecast_new_accounts"].values

        # Metrics
        n = min(len(actual_lv), len(forecast_lv))
        metrics = compute_cycle_metrics(
            pd.Series(actual_lv[:n]),
            pd.Series(forecast_lv[:n]),
            pd.Series(lower[:n]),
            pd.Series(upper[:n]),
        )
        acct_mae = compute_accounts_mae(pd.Series(actual_accts[:n]), pd.Series(forecast_accts[:n]))
        metrics["acct_mae"] = acct_mae

        # IPO-week vs non-IPO-week MAE for new accounts
        ipo_flag = test_df.get("ipo_announcement_week", pd.Series(0, index=test_df.index)).values[:n]
        ipo_mask     = ipo_flag == 1
        non_ipo_mask = ipo_flag == 0
        if ipo_mask.sum() > 0:
            metrics["acct_mae_ipo"] = float(np.mean(np.abs(
                actual_accts[:n][ipo_mask] - forecast_accts[:n][ipo_mask]
            )))
        else:
            metrics["acct_mae_ipo"] = None
        if non_ipo_mask.sum() > 0:
            metrics["acct_mae_non_ipo"] = float(np.mean(np.abs(
                actual_accts[:n][non_ipo_mask] - forecast_accts[:n][non_ipo_mask]
            )))
        else:
            metrics["acct_mae_non_ipo"] = None

        cycle_metrics.append(metrics)

        # Build result rows
        cycle_df = pd.DataFrame(
            {
                "cycle":                cycle_num,
                "week_end_date":        test_df["week_end_date"].values[:n],
                "trading_days":         test_df["trading_days"].values[:n],
                "actual_log_volume":    actual_lv[:n],
                "forecast_log_volume":  forecast_lv[:n],
                "lower_ci":             lower[:n],
                "upper_ci":             upper[:n],
                "actual_volume":        actual_vol[:n],
                "forecast_volume":      forecast_vol[:n],
                "actual_new_accounts":  actual_accts[:n],
                "forecast_new_accounts": forecast_accts[:n],
                "ipo_announcement_week": ipo_flag,
                "error_log_vol":        actual_lv[:n] - forecast_lv[:n],
                "error_vol_pct":        (actual_vol[:n] - forecast_vol[:n]) / actual_vol[:n] * 100,
                "in_ci":                ((actual_lv[:n] >= lower[:n]) & (actual_lv[:n] <= upper[:n])).astype(int),
            }
        )
        all_results.append(cycle_df)

    if not all_results:
        raise RuntimeError("Backtest produced no results — check your data coverage.")

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(BACKTEST_RESULTS_PATH, index=False)
    print(f"\n  Backtest results saved → {BACKTEST_RESULTS_PATH}")

    _save_backtest_summary(cycle_metrics)
    _plot_backtest_results(results_df, weekly_df, cycle_metrics)

    return results_df


# ── Summary report ────────────────────────────────────────────────────────────

def _save_backtest_summary(cycle_metrics: list):
    headers = ["Metric", "Cycle 1", "Cycle 2", "Cycle 3"]
    rows = []

    def _fmt(metrics, key, fmt):
        return fmt.format(metrics[key]) if metrics else "N/A"

    def _fmt_nullable(metrics, key, fmt):
        if metrics is None:
            return "N/A"
        val = metrics.get(key)
        if val is None:
            return "n/a"
        return fmt.format(val)

    for i, row_def in enumerate([
        ("MAE (log vol)",           "mae",              "{:.3f}"),
        ("RMSE (log vol)",          "rmse",             "{:.3f}"),
        ("MAPE (volume %)",         "mape",             "{:.1f}%"),
        ("Direction accuracy",      "direction_acc",    "{:.0f}%"),
        ("95% CI coverage",         "coverage",         "{:.0f}%"),
        ("MAE (new accounts)",      "acct_mae",         "{:.0f}"),
        ("MAE — IPO weeks",         "acct_mae_ipo",     "{:.0f}"),
        ("MAE — non-IPO weeks",     "acct_mae_non_ipo", "{:.0f}"),
    ]):
        label, key, fmt = row_def
        row = [label]
        for cm in cycle_metrics:
            if key in ("acct_mae_ipo", "acct_mae_non_ipo"):
                row.append(_fmt_nullable(cm, key, fmt))
            else:
                row.append(_fmt(cm, key, fmt))
        # Pad to 3 cycles
        while len(row) < 4:
            row.append("N/A")
        rows.append(row)

    table = tabulate(rows, headers=headers, tablefmt="fancy_grid")
    title_line = "IHSG FORECAST MODEL — BACKTEST SUMMARY"
    report = f"\n{title_line}\n\n{table}\n"

    print(report)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"  Summary report saved → {REPORT_PATH}")


# ── CI clip helper (mirrors scenario_output.py) ──────────────────────────────
_CI_CLIP_DELTA = 1.5

def _clip_ci_bt(forecast_lv, lower, upper):
    lo = np.maximum(lower, forecast_lv - _CI_CLIP_DELTA)
    hi = np.minimum(upper, forecast_lv + _CI_CLIP_DELTA)
    return lo, hi


# ── Visualisations ────────────────────────────────────────────────────────────

def _plot_combined_backtest(results_df: pd.DataFrame, weekly_df: pd.DataFrame,
                            cycle_metrics: list):
    """
    Combined 2-panel backtest chart for both Model 1 (volume) and Model 2 (new accounts).
    Saves to outputs/charts/backtest_combined.png.
    """
    os.makedirs(CHARTS_DIR, exist_ok=True)
    cycle_colors = {1: COLORS["cycle1"], 2: COLORS["cycle2"], 3: COLORS["cycle3"]}

    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

        # ── Panel A: Volume (IDR Bn) ────────────────────────────────────────
        # Full history anchor
        ax1.plot(
            weekly_df["week_end_date"],
            np.exp(weekly_df["log_volume"]) / 1e9,
            color=COLORS["actual"], linewidth=1.8, label="Actual (full history)", zorder=5,
        )

        for cyc, grp in results_df.groupby("cycle"):
            color = cycle_colors.get(cyc, "#555555")
            # Forecast line
            ax1.plot(grp["week_end_date"], grp["forecast_volume"] / 1e9,
                     color=color, linestyle="--", linewidth=1.8,
                     label=f"Forecast — Cycle {cyc}", zorder=4)
            # Clipped CI band
            lo_c, hi_c = _clip_ci_bt(
                grp["forecast_log_volume"].values,
                grp["lower_ci"].values,
                grp["upper_ci"].values,
            )
            ax1.fill_between(
                grp["week_end_date"],
                np.exp(lo_c) * grp["trading_days"].values / 1e9,
                np.exp(hi_c) * grp["trading_days"].values / 1e9,
                color=color, alpha=0.15, zorder=3,
            )
            # Cycle start marker
            ax1.axvline(grp["week_end_date"].iloc[0], color=color, linestyle=":", alpha=0.6)

            # Annotate MAPE on chart
            cm = cycle_metrics[cyc - 1] if cyc - 1 < len(cycle_metrics) else None
            if cm and cm.get("mape") is not None:
                mid_date = grp["week_end_date"].iloc[len(grp) // 2]
                y_pos = (np.exp(grp["forecast_log_volume"]) * grp["trading_days"] / 1e9).mean()
                ax1.annotate(f"MAPE {cm['mape']:.1f}%",
                             xy=(mid_date, y_pos),
                             fontsize=8, color=color,
                             xytext=(4, 8), textcoords="offset points")

        ax1.set_title("Model 1 — Weekly Volume Backtest: Forecast vs Actual (3 cycles × 2 months)",
                      fontsize=11, fontweight="bold")
        ax1.set_ylabel("Volume (IDR Billion)", fontsize=10)
        ax1.legend(fontsize=8, loc="upper left")
        ax1.tick_params(axis="x", rotation=30)

        # ── Panel B: New Accounts ───────────────────────────────────────────
        for cyc, grp in results_df.groupby("cycle"):
            color = cycle_colors.get(cyc, "#555555")
            # Check for sane actual values (guard against cycle 2 overflow)
            actuals = pd.to_numeric(grp["actual_new_accounts"], errors="coerce")
            forecasts = pd.to_numeric(grp["forecast_new_accounts"], errors="coerce")
            max_sane = 2e6  # cap at 2 million to exclude overflow rows
            mask = actuals.abs() < max_sane
            if mask.sum() == 0:
                continue
            ax2.bar(grp.loc[mask, "week_end_date"], actuals[mask],
                    color=color, alpha=0.35, width=5, label=f"Actual — Cycle {cyc}", zorder=2)
            ax2.plot(grp.loc[mask, "week_end_date"], forecasts[mask],
                     color=color, linestyle="--", linewidth=1.8,
                     label=f"Forecast — Cycle {cyc}", zorder=4)
            ax2.axvline(grp["week_end_date"].iloc[0], color=color, linestyle=":", alpha=0.6)

        ax2.set_title("Model 2 — New Account Registrations Backtest: Forecast vs Actual",
                      fontsize=11, fontweight="bold")
        ax2.set_xlabel("Week End Date", fontsize=10)
        ax2.set_ylabel("New Accounts", fontsize=10)
        ax2.legend(fontsize=8, loc="upper left")
        ax2.tick_params(axis="x", rotation=30)

        fig.tight_layout(pad=2.0)
        path = os.path.join(CHARTS_DIR, "backtest_combined.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved → {path}")

    except Exception as e:
        print(f"  WARNING: Combined backtest chart failed: {e}")


def _plot_backtest_results(results_df: pd.DataFrame, weekly_df: pd.DataFrame,
                           cycle_metrics: list = None):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    cycle_colors = {1: COLORS["cycle1"], 2: COLORS["cycle2"], 3: COLORS["cycle3"]}

    # Combined chart first (the primary deliverable)
    _plot_combined_backtest(results_df, weekly_df, cycle_metrics or [])

    # ── Chart 1 — Log volume forecast vs actual ─────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            weekly_df["week_end_date"],
            weekly_df["log_volume_adj"],
            color=COLORS["actual"],
            linewidth=1.5,
            label="Actual (full history)",
        )
        for cyc, grp in results_df.groupby("cycle"):
            color = cycle_colors[cyc]
            ax.plot(grp["week_end_date"], grp["forecast_log_volume"],
                    color=color, linestyle="--", linewidth=1.5, label=f"Forecast Cycle {cyc}")
            ax.fill_between(grp["week_end_date"], grp["lower_ci"], grp["upper_ci"],
                            color=color, alpha=0.15)
            ax.axvline(grp["week_end_date"].iloc[0], color=color, linestyle=":", alpha=0.5)
        ax.set_title("IHSG Weekly Volume — Backtest Forecast vs Actual (3 cycles × 2 months)")
        ax.set_xlabel("Week End Date")
        ax.set_ylabel("Log Volume")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(CHARTS_DIR, "backtest_volume_forecast.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart saved → {path}")
    except Exception as e:
        print(f"  WARNING: Chart 1 failed: {e}")

    # ── Chart 2 — Volume levels (IDR) ──────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            weekly_df["week_end_date"],
            np.exp(weekly_df["log_volume"]) / 1e9,
            color=COLORS["actual"],
            linewidth=1.5,
            label="Actual",
        )
        for cyc, grp in results_df.groupby("cycle"):
            color = cycle_colors[cyc]
            ax.plot(grp["week_end_date"], grp["forecast_volume"] / 1e9,
                    color=color, linestyle="--", linewidth=1.5, label=f"Forecast Cycle {cyc}")
            # CI bands: exp(ci on log_volume_adj) * trading_days → total weekly volume
            ax.fill_between(
                grp["week_end_date"],
                np.exp(grp["lower_ci"]) * grp["trading_days"] / 1e9,
                np.exp(grp["upper_ci"]) * grp["trading_days"] / 1e9,
                color=color, alpha=0.15,
            )
        ax.set_title("IHSG Weekly Volume Levels — Backtest Forecast vs Actual")
        ax.set_xlabel("Week End Date")
        ax.set_ylabel("Volume (IDR Billion)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(CHARTS_DIR, "backtest_volume_levels.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart saved → {path}")
    except Exception as e:
        print(f"  WARNING: Chart 2 failed: {e}")

    # ── Chart 3 — New accounts ─────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(14, 6))
        for cyc, grp in results_df.groupby("cycle"):
            color = cycle_colors[cyc]
            ax.bar(grp["week_end_date"], grp["actual_new_accounts"],
                   color=color, alpha=0.4, width=5, label=f"Actual Cycle {cyc}")
            ax.plot(grp["week_end_date"], grp["forecast_new_accounts"],
                    color=color, linestyle="--", linewidth=1.5, label=f"Forecast Cycle {cyc}")
        ax.set_title("IHSG New Brokerage Accounts — Backtest Forecast vs Actual")
        ax.set_xlabel("Week End Date")
        ax.set_ylabel("New Accounts")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = os.path.join(CHARTS_DIR, "backtest_new_accounts.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart saved → {path}")
    except Exception as e:
        print(f"  WARNING: Chart 3 failed: {e}")

    # ── Chart 4 — Error distribution ───────────────────────────────────────
    try:
        cycles = sorted(results_df["cycle"].unique())
        fig, axes = plt.subplots(1, len(cycles), figsize=(5 * len(cycles), 5), sharey=False)
        if len(cycles) == 1:
            axes = [axes]
        for ax, cyc in zip(axes, cycles):
            grp = results_df[results_df["cycle"] == cyc]
            errors = grp["error_vol_pct"].dropna()
            color = cycle_colors.get(cyc, "#555555")
            sns.histplot(errors, ax=ax, color=color, bins=10, kde=True)
            ax.axvline(errors.mean(), color="red", linestyle="--", linewidth=1.5,
                       label=f"Mean: {errors.mean():.1f}%")
            ax.set_title(f"Cycle {cyc} — Forecast Error Distribution")
            ax.set_xlabel("% Forecast Error")
            ax.legend(fontsize=8)
        fig.suptitle("MAPE Distribution per Backtest Cycle", fontsize=12)
        fig.tight_layout()
        path = os.path.join(CHARTS_DIR, "backtest_error_distribution.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Chart saved → {path}")
    except Exception as e:
        print(f"  WARNING: Chart 4 failed: {e}")

    # ── Chart 5 — IPO effect analysis ──────────────────────────────────────
    try:
        from config import IPO_IMPACT_PATH
        if os.path.exists(IPO_IMPACT_PATH):
            impact_df = pd.read_csv(IPO_IMPACT_PATH)
            if len(impact_df) > 0:
                fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(14, 10))

                # Panel A: actual vs fitted with IPO events marked
                has_actuals = "new_accounts" in weekly_df.columns and not weekly_df["new_accounts"].isna().all()
                if has_actuals:
                    ax_a.bar(weekly_df["week_end_date"], weekly_df["new_accounts"].fillna(0),
                             color="#AAAAAA", alpha=0.6, width=4, label="Actual new accounts")

                # Mark IPO announcement weeks with vertical lines
                for _, row in impact_df.iterrows():
                    wdt = pd.Timestamp(row["week_end_date"])
                    ax_a.axvline(wdt, color="#FF8C00", linewidth=1.5, alpha=0.7)
                    ticker = str(row.get("ipo_ticker", ""))
                    if ticker and ticker != "nan":
                        ax_a.text(wdt, ax_a.get_ylim()[1] * 0.9 if ax_a.get_ylim()[1] > 0 else 1,
                                  ticker, rotation=45, fontsize=7, color="#FF8C00")
                    if int(row.get("ipo_large_flag", 0)) == 1:
                        ax_a.plot(wdt, 0, marker="*", color="red", markersize=10, zorder=5)

                ax_a.set_title("New account registrations — actual with IPO events")
                ax_a.set_ylabel("New Accounts")
                ax_a.legend(fontsize=8)

                # Panel B: horizontal bar of uplift %
                colors_b = ["#c06a1b" if int(r.get("ipo_large_flag", 0)) == 0 else "#a01515"
                            for _, r in impact_df.iterrows()]
                labels_b = [str(r.get("week_end_date", i))[:10] for i, r in impact_df.iterrows()]
                ax_b.barh(labels_b, impact_df["ipo_uplift_pct"].fillna(0), color=colors_b)
                ax_b.set_xlabel("% uplift vs baseline")
                ax_b.set_title("Estimated new-user uplift per IPO announcement")
                ax_b.axvline(0, color="black", linewidth=0.8)

                fig.tight_layout()
                path = os.path.join(CHARTS_DIR, "ipo_effect_analysis.png")
                fig.savefig(path, dpi=150)
                plt.close(fig)
                print(f"  Chart saved → {path}")
    except Exception as e:
        print(f"  WARNING: IPO effect chart failed: {e}")
