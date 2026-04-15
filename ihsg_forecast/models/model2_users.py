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


def fit_model2(weekly_df: pd.DataFrame):
    """
    Fits Negative Binomial model on weekly_df.
    Returns (fitted_model, significant_lag_cols, used_synthetic: bool).
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
    exog_cols = BASE_EXOG_COLS + significant_lags
    model_df = df[["new_accounts"] + exog_cols].dropna()
    X = sm.add_constant(model_df[exog_cols])
    y = model_df["new_accounts"].astype(int)

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

    return result, significant_lags, used_synthetic


def forecast_model2(
    fitted_model2,
    significant_lags: list,
    volume_forecast_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Takes Model 1 volume forecast, returns weekly new_accounts forecast.

    Parameters
    ----------
    fitted_model2    : fitted NegativeBinomial (or Poisson) result
    significant_lags : list of lag column names used in fit
    volume_forecast_df : output of forecast_model1 (contains forecast_log_volume)
    weekly_df        : historical weekly DataFrame (for lag values)

    Returns
    -------
    DataFrame with columns: week_end_date, forecast_new_accounts
    """
    steps = len(volume_forecast_df)
    last_row = weekly_df[BASE_EXOG_COLS + LAG_COLS].dropna().iloc[-1].copy()

    rows = []
    for i, (_, frow) in enumerate(volume_forecast_df.iterrows()):
        row = last_row.copy()
        row["log_volume_adj"] = frow["forecast_log_volume"]  # model target is log_volume_adj
        row["log_trading_days"] = np.log(5)  # assume standard 5-day week
        row["volume_momentum"] = 0.0  # neutral forward assumption
        row["macro_shock_score"] = 0.0
        row["weekly_return"] = last_row["weekly_return"]
        row["cumulative_4w_return"] = last_row.get("cumulative_4w_return", 0.0)
        rows.append(row)

    future_df = pd.DataFrame(rows)[BASE_EXOG_COLS + significant_lags]
    X_future = sm.add_constant(future_df, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = fitted_model2.predict(X_future)

    result_df = volume_forecast_df[["week_end_date"]].copy()
    result_df["forecast_new_accounts"] = np.maximum(0, predictions.values.astype(int))
    return result_df
