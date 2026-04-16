import os
import pandas as pd
import numpy as np

from config import MACRO_INPUT_PATH, PROCESSED_DATA_PATH, FORMULA_NOTES_PATH, MACRO_SHOCK_MAX

# ── Formula registry — serialised to formula_notes.csv ────────────────────────
FORMULA_REGISTRY = {
    "index_level": {
        "formula": "P_t = Friday closing price",
        "inputs_required": "close (weekly)",
        "source": "yahoo_finance",
        "notes": "Friday close of IHSG (IDX Composite)",
    },
    "weekly_return": {
        "formula": "r_t = ln(P_t / P_{t-1})",
        "inputs_required": "index_level",
        "source": "computed",
        "notes": "Log return ensures stationarity; first row is NaN",
    },
    "log_volume": {
        "formula": "lv_t = ln(V_t)",
        "inputs_required": "volume (weekly sum)",
        "source": "yahoo_finance",
        "notes": "Target variable for Model 1; log-transform stabilises variance",
    },
    "realized_volatility": {
        "formula": "sigma_t = std(r_daily in week t) * sqrt(n_days)",
        "inputs_required": "daily close prices",
        "source": "computed",
        "notes": "Realised weekly volatility proxy scaled by sqrt of trading days",
    },
    "volume_momentum": {
        "formula": "delta_V_t = lv_t - lv_{t-1}",
        "inputs_required": "log_volume",
        "source": "computed",
        "notes": "First-difference of log volume; captures acceleration in activity",
    },
    "lag_lv_1": {
        "formula": "lv_{t-1}",
        "inputs_required": "log_volume",
        "source": "computed",
        "notes": "1-week lagged log volume — AR input",
    },
    "lag_lv_2": {
        "formula": "lv_{t-2}",
        "inputs_required": "log_volume",
        "source": "computed",
        "notes": "2-week lagged log volume — AR input",
    },
    "lag_lv_3": {
        "formula": "lv_{t-3}",
        "inputs_required": "log_volume",
        "source": "computed",
        "notes": "3-week lagged log volume — AR input",
    },
    "lag_lv_4": {
        "formula": "lv_{t-4}",
        "inputs_required": "log_volume",
        "source": "computed",
        "notes": "4-week lagged log volume — AR input",
    },
    "cumulative_4w_return": {
        "formula": "CR_t = sum(r_{t-k}, k=0..3)",
        "inputs_required": "weekly_return",
        "source": "computed",
        "notes": "Rolling 4-week sum of returns; bull/bear momentum signal",
    },
    "interest_rate_direction": {
        "formula": "IR_t = rate_t - rate_{t-4}",
        "inputs_required": "policy_rate (macro_shocks.csv)",
        "source": "macro_shocks.csv",
        "notes": "4-week change in BI policy rate; positive=tightening, negative=easing",
    },
    "macro_shock_score": {
        "formula": "S_t in {-2, -1.5, -1, -0.5, 0, +0.5, +1, +1.5, +2}",
        "inputs_required": "shock_score (macro_shocks.csv)",
        "source": "macro_shocks.csv",
        "notes": "Analyst-assigned weekly macro sentiment score",
    },
    "d_geo": {
        "formula": "D_geo in {0,1}",
        "inputs_required": "event_type (macro_shocks.csv)",
        "source": "macro_shocks.csv",
        "notes": "Binary dummy: geopolitical event week",
    },
    "d_mp": {
        "formula": "D_mp in {0,1}",
        "inputs_required": "event_type (macro_shocks.csv)",
        "source": "macro_shocks.csv",
        "notes": "Binary dummy: monetary policy event week",
    },
    "d_trade": {
        "formula": "D_trade in {0,1}",
        "inputs_required": "event_type (macro_shocks.csv)",
        "source": "macro_shocks.csv",
        "notes": "Binary dummy: trade event week",
    },
    "d_corporate": {
        "formula": "D_corporate in {0,1}",
        "inputs_required": "event_type (macro_shocks.csv)",
        "source": "macro_shocks.csv",
        "notes": "Binary dummy: corporate event week",
    },
}


# ── Macro file loader ─────────────────────────────────────────────────────────

def load_macro(path: str = MACRO_INPUT_PATH) -> pd.DataFrame:
    """Load user macro input CSV. Returns empty DataFrame if file not found."""
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found. All macro variables set to 0.")
        return pd.DataFrame()
    df = pd.read_csv(path, comment="#", parse_dates=["week_end_date"])
    print(f"  Loaded macro input: {len(df)} rows from {path}")
    return df


# ── Weekly aggregation ────────────────────────────────────────────────────────

def _aggregate_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly (week ending Friday)."""
    df = daily_df.copy()
    df = df.set_index("date")
    df.index = pd.to_datetime(df.index)

    weekly = df.resample("W-FRI").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    weekly["trading_days"] = df["close"].resample("W-FRI").count()
    weekly.index.name = "week_end_date"
    weekly = weekly.reset_index()

    # Drop weeks with zero volume (market closed weeks) before any log transforms
    n_before = len(weekly)
    weekly = weekly[weekly["volume"] > 0].copy()
    n_dropped = n_before - len(weekly)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} zero-volume week(s) before modelling.")

    return weekly


# ── Variable computations ─────────────────────────────────────────────────────

def _add_index_level(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly["index_level"] = weekly["close"]
    return weekly


def _add_weekly_return(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly["weekly_return"] = np.log(weekly["index_level"] / weekly["index_level"].shift(1))
    return weekly


def _add_log_volume(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly["log_volume"] = np.log(weekly["volume"])
    # Trading-day-adjusted: log of avg daily volume — removes calendar swing
    weekly["log_volume_adj"] = np.log(weekly["volume"] / weekly["trading_days"])
    # log(trading_days) is the correct linear regressor under volume proportionality
    weekly["log_trading_days"] = np.log(weekly["trading_days"])
    return weekly


def _add_realized_volatility(weekly: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    """Compute realised weekly volatility from daily returns."""
    daily = daily_df.copy()
    daily = daily.set_index("date")
    daily.index = pd.to_datetime(daily.index)
    daily["daily_return"] = np.log(daily["close"] / daily["close"].shift(1))
    daily["week_period"] = daily.index.to_period("W-FRI")

    vol_df = daily.groupby("week_period").agg(
        daily_ret_std=("daily_return", "std"),
        n_days=("daily_return", "count"),
    ).reset_index()
    vol_df["realized_volatility"] = vol_df["daily_ret_std"] * np.sqrt(vol_df["n_days"])

    # Map week_period back to the week_end_date in weekly DataFrame
    weekly["week_period"] = weekly["week_end_date"].dt.to_period("W-FRI")
    weekly = weekly.merge(
        vol_df[["week_period", "realized_volatility"]],
        on="week_period",
        how="left",
    )
    weekly = weekly.drop(columns=["week_period"])
    return weekly


def _add_volume_momentum(weekly: pd.DataFrame) -> pd.DataFrame:
    # Use adjusted volume so momentum reflects true market activity, not calendar
    weekly["volume_momentum"] = weekly["log_volume_adj"].diff(1)
    return weekly


def _add_lagged_log_volume(weekly: pd.DataFrame) -> pd.DataFrame:
    # Lags derived from adjusted volume — prevents holiday weeks from polluting AR structure
    for k in range(1, 5):
        weekly[f"lag_lv_{k}"] = weekly["log_volume_adj"].shift(k)
    return weekly


def _add_cumulative_4w_return(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly["cumulative_4w_return"] = weekly["weekly_return"].rolling(4).sum()
    return weekly


def _add_macro_variables(weekly: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Merge macro shock variables onto weekly DataFrame."""

    if macro_df.empty:
        # No macro file — set all to zero
        weekly["macro_shock_score"] = 0.0
        weekly["macro_shock_abs"] = 0.0
        weekly["macro_neg_shock"] = 0.0
        weekly["macro_neg_lag1"] = 0.0
        weekly["macro_neg_lag2"] = 0.0
        weekly["macro_pos_shock"] = 0.0
        weekly["macro_pos_lag1"] = 0.0
        weekly["macro_pos_lag2"] = 0.0
        weekly["interest_rate_direction"] = 0.0
        for col in ["d_geo", "d_mp", "d_trade", "d_corporate"]:
            weekly[col] = 0
        return weekly

    # ── shock_score validation ──────────────────────────────────────────────
    if "shock_score" in macro_df.columns:
        macro_df["shock_score"] = pd.to_numeric(macro_df["shock_score"], errors="coerce").fillna(0)
        bad_mask = (macro_df["shock_score"] < -MACRO_SHOCK_MAX) | (macro_df["shock_score"] > MACRO_SHOCK_MAX)
        if bad_mask.any():
            for _, row in macro_df[bad_mask].iterrows():
                print(
                    f"  WARNING: shock_score {row['shock_score']} out of range "
                    f"at {row['week_end_date']}. Clipping to [{-MACRO_SHOCK_MAX}, {MACRO_SHOCK_MAX}]."
                )
            macro_df["shock_score"] = macro_df["shock_score"].clip(-MACRO_SHOCK_MAX, MACRO_SHOCK_MAX)
    else:
        macro_df["shock_score"] = 0.0

    # ── Merge macro onto weekly ─────────────────────────────────────────────
    weekly = weekly.merge(
        macro_df[["week_end_date", "shock_score"]].rename(
            columns={"shock_score": "macro_shock_score"}
        ),
        on="week_end_date",
        how="left",
    )
    weekly["macro_shock_score"] = weekly["macro_shock_score"].fillna(0.0)

    # Decompose shock into three distinct effects:
    #   abs   — announcement-week volume spike (both pos & neg events drive volume up)
    #   neg lags — negative events suppress volume in subsequent 1-2 weeks (fear/uncertainty)
    #   pos lags — positive events sustain elevated volume in subsequent 1-2 weeks (fund rebalancing)
    weekly["macro_shock_abs"] = weekly["macro_shock_score"].abs()
    weekly["macro_neg_shock"]  = (-weekly["macro_shock_score"]).clip(lower=0)
    weekly["macro_neg_lag1"]   = weekly["macro_neg_shock"].shift(1).fillna(0.0)
    weekly["macro_neg_lag2"]   = weekly["macro_neg_shock"].shift(2).fillna(0.0)
    weekly["macro_pos_shock"]  = weekly["macro_shock_score"].clip(lower=0)
    weekly["macro_pos_lag1"]   = weekly["macro_pos_shock"].shift(1).fillna(0.0)
    weekly["macro_pos_lag2"]   = weekly["macro_pos_shock"].shift(2).fillna(0.0)

    # ── interest_rate_direction ─────────────────────────────────────────────
    if "policy_rate" in macro_df.columns:
        macro_df["policy_rate"] = pd.to_numeric(macro_df["policy_rate"], errors="coerce")
        weekly = weekly.merge(
            macro_df[["week_end_date", "policy_rate"]],
            on="week_end_date",
            how="left",
        )
        weekly["policy_rate"] = weekly["policy_rate"].ffill()
        weekly["interest_rate_direction"] = weekly["policy_rate"] - weekly["policy_rate"].shift(4)
        weekly = weekly.drop(columns=["policy_rate"])
    else:
        print(
            "  WARNING: policy_rate column not found in macro input. "
            "interest_rate_direction set to 0. "
            "Please add policy_rate column to macro_shocks.csv."
        )
        weekly["interest_rate_direction"] = 0.0

    # ── new_accounts (optional for Model 2) ────────────────────────────────
    if "new_accounts" in macro_df.columns:
        weekly = weekly.merge(
            macro_df[["week_end_date", "new_accounts"]],
            on="week_end_date",
            how="left",
        )

    # ── event_type dummies ──────────────────────────────────────────────────
    if "event_type" in macro_df.columns:
        macro_df["event_type"] = macro_df["event_type"].fillna("")
        event_map = {
            "geopolitical": "d_geo",
            "monetary_policy": "d_mp",
            "trade": "d_trade",
            "corporate": "d_corporate",
        }
        for val, col in event_map.items():
            macro_df[col] = (macro_df["event_type"] == val).astype(int)

        weekly = weekly.merge(
            macro_df[["week_end_date"] + list(event_map.values())],
            on="week_end_date",
            how="left",
        )
    else:
        for col in ["d_geo", "d_mp", "d_trade", "d_corporate"]:
            weekly[col] = 0

    for col in ["d_geo", "d_mp", "d_trade", "d_corporate"]:
        if col in weekly.columns:
            weekly[col] = weekly[col].fillna(0).astype(int)
        else:
            weekly[col] = 0

    return weekly


# ── Save formula notes ────────────────────────────────────────────────────────

def _save_formula_notes():
    rows = []
    for var, meta in FORMULA_REGISTRY.items():
        rows.append({"variable_name": var, **meta})
    pd.DataFrame(rows).to_csv(FORMULA_NOTES_PATH, index=False)
    print(f"  Formula notes saved → {FORMULA_NOTES_PATH}")


# ── Main pipeline function ────────────────────────────────────────────────────

def compute_all_variables(daily_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw daily OHLCV and macro shock input.
    Returns weekly DataFrame with all computed variables.
    Saves weekly_variables.csv and formula_notes.csv.
    """
    weekly = _aggregate_weekly(daily_df)

    weekly = _add_index_level(weekly)
    weekly = _add_weekly_return(weekly)
    weekly = _add_log_volume(weekly)
    weekly = _add_realized_volatility(weekly, daily_df)
    weekly = _add_volume_momentum(weekly)
    weekly = _add_lagged_log_volume(weekly)
    weekly = _add_cumulative_4w_return(weekly)
    weekly = _add_macro_variables(weekly, macro_df)

    # Final column ordering
    base_cols = [
        "week_end_date", "index_level", "weekly_return",
        "log_volume", "log_volume_adj", "log_trading_days",
        "realized_volatility", "volume_momentum",
        "lag_lv_1", "lag_lv_2", "lag_lv_3", "lag_lv_4",
        "cumulative_4w_return", "interest_rate_direction", "macro_shock_score",
        "macro_shock_abs",
        "macro_neg_shock", "macro_neg_lag1", "macro_neg_lag2",
        "macro_pos_shock", "macro_pos_lag1", "macro_pos_lag2",
        "d_geo", "d_mp", "d_trade", "d_corporate", "trading_days",
    ]
    # Include new_accounts if present
    if "new_accounts" in weekly.columns:
        base_cols.append("new_accounts")

    weekly = weekly[[c for c in base_cols if c in weekly.columns]]
    weekly.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"  Weekly variables saved → {PROCESSED_DATA_PATH} ({len(weekly)} rows)")

    _save_formula_notes()

    return weekly


# ── IPO dummy variables ───────────────────────────────────────────────────────

def compute_ipo_dummies(weekly_df: pd.DataFrame, ipo_df) -> pd.DataFrame:
    """
    Maps each IPO announcement_date to its Friday week and adds four binary columns:
        ipo_announcement_week  — 1 on the Friday of the announcement week
        ipo_effect_week_1      — 1 one Friday later  (subscription period)
        ipo_effect_week_2      — 1 two Fridays later (allotment / early trading)
        ipo_large_flag         — 1 if market_cap_idr_trillion >= IPO_MIN_MARKET_CAP_IDR_T

    Multiple IPOs in the same week are OR-combined (max, not sum).
    If ipo_df is None or empty, all four columns are set to 0.
    """
    from config import IPO_MIN_MARKET_CAP_IDR_T

    result = weekly_df.copy()
    for col in ["ipo_announcement_week", "ipo_effect_week_1", "ipo_effect_week_2", "ipo_large_flag"]:
        result[col] = 0

    if ipo_df is None or len(ipo_df) == 0:
        print("  WARNING: No IPO calendar provided — all IPO dummies set to 0.")
        return result

    # Build a mapping from Friday dates → (ann_flag, large_flag)
    friday_ann   = {}   # friday → 1
    friday_large = {}   # friday → 1

    for _, row in ipo_df.iterrows():
        try:
            dt = pd.Timestamp(row["announcement_date"])
        except Exception:
            continue
        # Map to containing or next Friday
        days_to_friday = (4 - dt.weekday()) % 7
        friday = dt + pd.Timedelta(days=days_to_friday)

        friday_ann[friday] = 1
        cap = row.get("market_cap_idr_trillion", None)
        try:
            cap_val = float(cap)
            if cap_val >= IPO_MIN_MARKET_CAP_IDR_T:
                friday_large[friday] = 1
        except (TypeError, ValueError):
            pass

    # Apply flags to the weekly DataFrame
    date_series = pd.to_datetime(result["week_end_date"])
    for friday, _ in friday_ann.items():
        mask_ann  = date_series == friday
        mask_eff1 = date_series == (friday + pd.Timedelta(weeks=1))
        mask_eff2 = date_series == (friday + pd.Timedelta(weeks=2))
        result.loc[mask_ann,  "ipo_announcement_week"] = 1
        result.loc[mask_eff1, "ipo_effect_week_1"]     = 1
        result.loc[mask_eff2, "ipo_effect_week_2"]      = 1
        if friday in friday_large:
            result.loc[mask_ann,  "ipo_large_flag"] = 1
            result.loc[mask_eff1, "ipo_large_flag"] = 1
            result.loc[mask_eff2, "ipo_large_flag"] = 1

    n_ann   = int(result["ipo_announcement_week"].sum())
    n_large = int((result["ipo_announcement_week"] & result["ipo_large_flag"]).sum())
    n_ipo   = len(friday_ann)
    print(f"  [IPO] {n_ipo} IPOs loaded → {n_ann} announcement weeks flagged ({n_large} large).")

    return result
