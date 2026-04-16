import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

from config import MAX_LAGS

# Exogenous columns for Model 2 (lags added dynamically based on CCF)
BASE_EXOG_COLS = [
    "log_volume_adj",      # per-trading-day log volume (calendar-adjusted)
    "log_trading_days",    # calendar swing regressor
    "volume_momentum",
    "cumulative_4w_return",
    "macro_shock_score",
    "weekly_return",
]
LAG_COLS = [f"lag_lv_{k}" for k in range(1, MAX_LAGS + 1)]

IPO_COLS = [
    "ipo_announcement_week",
    "ipo_effect_week_1",
    "ipo_effect_week_2",
    "ipo_large_flag",
]


def _generate_synthetic_new_accounts(weekly_df: pd.DataFrame) -> pd.Series:
    """Generate synthetic new_accounts for demo purposes."""
    print(
        "  WARNING: 'new_accounts' column not found in weekly data.\n"
        "  Using synthetic new_accounts series for demonstration.\n"
        "  Add a 'new_accounts' column to macro_shocks.csv to enable real Model 2 fitting."
    )
    lv = weekly_df["log_volume_adj"].fillna(weekly_df["log_volume_adj"].mean())
    cr = weekly_df["cumulative_4w_return"].fillna(0)
    base = 5000
    np.random.seed(42)
    synthetic = (
        base
        * np.exp(0.4 * (lv - lv.mean()) / lv.std())
        * np.exp(0.3 * cr)
        * np.random.lognormal(0, 0.15, len(weekly_df))
    ).astype(int)
    return synthetic


def build_model2_exog(weekly_df: pd.DataFrame, selected_lags: list) -> pd.DataFrame:
    """
    Assemble the exogenous feature matrix for Model 2 with a fixed column order.
    This is the SINGLE place that builds exog — called by both fit and forecast.

    Fixed column order:
        lag_lv_{k} for k in selected_lags,
        volume_momentum, cumulative_4w_return, log_trading_days,
        macro_shock_score, weekly_return,
        ipo_announcement_week, ipo_effect_week_1, ipo_effect_week_2, ipo_large_flag

    IPO columns absent from weekly_df are filled with 0 (graceful degradation).
    A constant column is added via sm.add_constant.
    """
    df = weekly_df.copy()

    # Fill missing IPO columns with 0
    for col in IPO_COLS:
        if col not in df.columns:
            df[col] = 0

    # log_volume_adj is the current-period volume (set from Model 1 forecast during prediction)
    feature_cols = (
        selected_lags
        + ["log_volume_adj", "volume_momentum", "cumulative_4w_return", "log_trading_days",
           "macro_shock_score", "weekly_return"]
        + IPO_COLS
    )

    model_df = df[["new_accounts"] + feature_cols].dropna()
    n_dropped = len(df) - len(model_df)
    if n_dropped:
        print(f"  build_model2_exog: retained {len(model_df)} rows, dropped {n_dropped} (NaN).")

    # has_constant='add' forces adding const even when all-zero IPO columns are present
    # (statsmodels 'skip' mode would otherwise treat zero columns as an existing constant)
    X = sm.add_constant(model_df[feature_cols], has_constant="add")
    y = model_df["new_accounts"].astype(int)
    return X, y, model_df


def fit_model2(weekly_df: pd.DataFrame):
    """
    Fits Negative Binomial model on weekly_df.
    Returns (fitted_model_dict, significant_lag_cols, used_synthetic: bool).

    fitted_model_dict keys:
        'result'         : statsmodels NegativeBinomial / Poisson result
        'selected_lags'  : list of lag column names used in fit
        'used_synthetic' : bool
    """
    df = weekly_df.copy()

    # ── Resolve target variable ─────────────────────────────────────────────
    used_synthetic = False
    if "new_accounts" not in df.columns or df["new_accounts"].isna().all():
        df["new_accounts"] = _generate_synthetic_new_accounts(df)
        used_synthetic = True
    else:
        df["new_accounts"] = pd.to_numeric(df["new_accounts"], errors="coerce")
        n_missing = df["new_accounts"].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {n_missing} NaN in new_accounts. Dropping those rows.")

    # ── CCF analysis: find significant lags ────────────────────────────────
    valid = df[["new_accounts"] + LAG_COLS].dropna()
    correlations = {}
    for col in LAG_COLS:
        c = np.corrcoef(valid["new_accounts"], valid[col])[0, 1]
        correlations[col] = abs(c)

    dominant_lag = max(correlations, key=correlations.get)
    print(f"  Dominant lag: {dominant_lag} (|corr| = {correlations[dominant_lag]:.3f})")
    print("  Lag correlations: " + ", ".join(f"{k}={v:.3f}" for k, v in correlations.items()))

    significant_lags = [col for col, corr in correlations.items() if corr > 0.2]
    if not significant_lags:
        significant_lags = [dominant_lag]
        print(f"  No lags above 0.2 threshold; using dominant lag {dominant_lag}.")

    # ── Fit Negative Binomial ───────────────────────────────────────────────
    X, y, _ = build_model2_exog(df, significant_lags)

    nb_model = NegativeBinomial(y, X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = nb_model.fit(maxiter=200, disp=False)
        except Exception as e:
            print(f"  WARNING: NegativeBinomial failed ({e}). Falling back to Poisson GLM.")
            poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
            result = poisson_model.fit(maxiter=200, disp=False)

    pseudo_r2 = 1 - (result.llf / result.llnull) if hasattr(result, "llnull") and result.llnull else None
    if pseudo_r2 is not None:
        print(f"  Model 2 pseudo R²: {pseudo_r2:.4f}")

    # ── Print IPO coefficient summary ───────────────────────────────────────
    print("  [Model 2] IPO coefficient summary")
    any_negative = False
    for col in IPO_COLS:
        coef_name = col  # sm.add_constant keeps column names
        if coef_name in result.params.index:
            b = result.params[coef_name]
            p = result.pvalues[coef_name]
            print(f"    {col:<26}: b = {b:+.4f}  (p = {p:.3f})")
            if b < 0:
                any_negative = True
        else:
            print(f"    {col:<26}: not in model (column was all-zero)")
    if any_negative:
        print(
            "  WARNING: negative IPO coefficient detected. "
            "Verify that announcement_date values are book-open dates, not listing dates."
        )

    fitted_dict = {
        "result":         result,
        "selected_lags":  significant_lags,
        "used_synthetic": used_synthetic,
    }
    return fitted_dict, significant_lags, used_synthetic


def forecast_model2(
    fitted_model2_dict,
    significant_lags: list,
    volume_forecast_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    future_exog_df: pd.DataFrame = None,
    future_ipo_df=None,
) -> pd.DataFrame:
    """
    Takes Model 1 volume forecast, returns weekly new_accounts forecast.

    Parameters
    ----------
    fitted_model2_dict : dict returned by fit_model2()
    significant_lags   : list of lag column names used in fit
    volume_forecast_df : output of forecast_model1 (contains forecast_log_volume)
    weekly_df          : historical weekly DataFrame (for lag values)
    future_exog_df     : optional scenario future exog (supplies macro_shock_score per week)
    future_ipo_df      : optional rows from ipo_calendar where announcement_date > today

    Returns
    -------
    DataFrame with columns: week_end_date, forecast_new_accounts, baseline_new_accounts,
                             ipo_contribution_accounts
    """
    from compute_variables import compute_ipo_dummies

    result = fitted_model2_dict["result"] if isinstance(fitted_model2_dict, dict) else fitted_model2_dict
    steps = len(volume_forecast_df)

    last_row = weekly_df[BASE_EXOG_COLS + LAG_COLS].dropna().iloc[-1].copy()

    # Build forecast scaffold DataFrame (one row per forecast step)
    rows = []
    for i, (_, frow) in enumerate(volume_forecast_df.iterrows()):
        row = last_row.copy()
        row["log_volume_adj"]      = frow["forecast_log_volume"]
        row["log_trading_days"]    = np.log(5)
        row["volume_momentum"]     = 0.0
        row["weekly_return"]       = last_row["weekly_return"]
        row["cumulative_4w_return"] = last_row.get("cumulative_4w_return", 0.0)
        # Scenario-aware macro_shock_score
        if future_exog_df is not None and "macro_shock_score" in future_exog_df.columns:
            row["macro_shock_score"] = future_exog_df.iloc[i]["macro_shock_score"]
        else:
            row["macro_shock_score"] = 0.0
        rows.append(row)

    future_df = pd.DataFrame(rows)
    future_df["week_end_date"] = volume_forecast_df["week_end_date"].values
    # Assign placeholder new_accounts so build_model2_exog doesn't fail on dropna
    future_df["new_accounts"] = 0

    # Apply IPO dummies to the forecast scaffold
    future_df = compute_ipo_dummies(future_df, future_ipo_df)

    # Build exog with IPO dummies at computed values
    X_future, _, _ = build_model2_exog(future_df, significant_lags)

    # Build exog with IPO dummies zeroed out (baseline)
    future_df_baseline = future_df.copy()
    for col in IPO_COLS:
        future_df_baseline[col] = 0
    X_baseline, _, _ = build_model2_exog(future_df_baseline, significant_lags)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions  = result.predict(X_future)
        predictions_base = result.predict(X_baseline)

    forecast_accounts   = np.maximum(0, predictions.values.astype(int))
    baseline_accounts   = np.maximum(0, predictions_base.values.astype(int))
    ipo_contribution    = forecast_accounts - baseline_accounts

    result_df = volume_forecast_df[["week_end_date"]].copy().reset_index(drop=True)
    result_df["forecast_new_accounts"]    = forecast_accounts
    result_df["baseline_new_accounts"]    = baseline_accounts
    result_df["ipo_contribution_accounts"] = ipo_contribution

    # Print IPO-week annotations
    ipo_ann_col = future_df["ipo_announcement_week"].values
    for i, row in result_df.iterrows():
        if i < len(ipo_ann_col) and ipo_ann_col[i] == 1:
            ticker = ""
            if future_ipo_df is not None and len(future_ipo_df) > 0:
                # Try to match ticker by week
                week_dt = pd.Timestamp(row["week_end_date"])
                for _, irow in future_ipo_df.iterrows():
                    try:
                        ipo_dt = pd.Timestamp(irow["announcement_date"])
                        days_to_fri = (4 - ipo_dt.weekday()) % 7
                        ipo_friday = ipo_dt + pd.Timedelta(days=days_to_fri)
                        if ipo_friday == week_dt:
                            ticker = str(irow.get("ticker", ""))
                            break
                    except Exception:
                        pass
            print(
                f"  Week {row['week_end_date'].date() if hasattr(row['week_end_date'], 'date') else row['week_end_date']}: "
                f"{row['forecast_new_accounts']:,} accounts forecast, "
                f"of which {row['ipo_contribution_accounts']:,} from IPO"
                + (f": {ticker}" if ticker else "")
            )

    return result_df


def compute_ipo_impact_analysis(fitted_model2_dict, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each week in the fitted sample where ipo_announcement_week == 1,
    compute baseline vs IPO-inclusive predictions and the uplift %.

    Saves to outputs/csv/ipo_impact_analysis.csv.

    Returns
    -------
    DataFrame with columns:
        week_end_date, ipo_ticker, company_name, market_cap_idr_trillion,
        baseline_accounts, ipo_accounts, actual_accounts, ipo_uplift_pct, ipo_large_flag
    """
    from config import IPO_IMPACT_PATH

    result_model = fitted_model2_dict["result"]
    significant_lags = fitted_model2_dict["selected_lags"]

    df = weekly_df.copy()
    if "new_accounts" not in df.columns or df["new_accounts"].isna().all():
        from models.model2_users import _generate_synthetic_new_accounts
        df["new_accounts"] = _generate_synthetic_new_accounts(df)

    # Only analyse IPO announcement weeks
    ipo_mask = df.get("ipo_announcement_week", pd.Series(0, index=df.index)) == 1
    ipo_weeks = df[ipo_mask].copy()

    if len(ipo_weeks) == 0:
        print("  [IPO impact analysis] No IPO announcement weeks in training data — skipping.")
        return pd.DataFrame()

    records = []
    for _, row in ipo_weeks.iterrows():
        week_df = pd.DataFrame([row])
        week_df["new_accounts"] = row.get("new_accounts", 0)

        # With IPO dummies
        X_ipo, _, _ = build_model2_exog(week_df, significant_lags)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_ipo = result_model.predict(X_ipo).values[0]

        # Without IPO dummies
        week_df_base = week_df.copy()
        for col in IPO_COLS:
            week_df_base[col] = 0
        X_base, _, _ = build_model2_exog(week_df_base, significant_lags)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_base = result_model.predict(X_base).values[0]

        uplift_pct = (pred_ipo - pred_base) / pred_base * 100 if pred_base > 0 else float("nan")

        records.append({
            "week_end_date":          row["week_end_date"],
            "ipo_ticker":             "",  # filled from ipo_calendar if available
            "company_name":           "",
            "market_cap_idr_trillion": float("nan"),
            "baseline_accounts":      int(max(0, pred_base)),
            "ipo_accounts":           int(max(0, pred_ipo)),
            "actual_accounts":        row.get("new_accounts", float("nan")),
            "ipo_uplift_pct":         uplift_pct,
            "ipo_large_flag":         int(row.get("ipo_large_flag", 0)),
        })

    impact_df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(IPO_IMPACT_PATH), exist_ok=True)
    impact_df.to_csv(IPO_IMPACT_PATH, index=False)
    print(f"  IPO impact analysis saved → {IPO_IMPACT_PATH} ({len(impact_df)} IPO weeks)")
    return impact_df


import os  # needed by compute_ipo_impact_analysis
