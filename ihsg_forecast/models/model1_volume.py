import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER, FORECAST_WEEKS

# Exogenous columns used by Model 1 — defined once, reused by backtest
EXOG_COLS = [
    "weekly_return",
    "realized_volatility",
    "macro_shock_score",
    "interest_rate_direction",
    "d_geo",
    "d_mp",
    "d_trade",
]


def fit_model1(weekly_df: pd.DataFrame):
    """
    Fits SARIMAX on full weekly_df.
    Returns fitted SARIMAXResults object.
    """
    df = weekly_df[["log_volume"] + EXOG_COLS].dropna().copy()

    model = SARIMAX(
        endog=df["log_volume"],
        exog=df[EXOG_COLS],
        order=SARIMAX_ORDER,
        seasonal_order=SARIMAX_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = model.fit(disp=False)
        except Exception as e:
            print(f"  WARNING: SARIMAX default optimizer failed ({e}). Retrying with method='nm'.")
            result = model.fit(method="nm", disp=False)

    print(f"  Model 1 AIC: {result.aic:.2f}  BIC: {result.bic:.2f}")
    return result


def forecast_model1(
    fitted_model,
    weekly_df: pd.DataFrame,
    steps: int = FORECAST_WEEKS,
    future_exog_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Generates steps-ahead volume forecast with 95% CI.

    Parameters
    ----------
    fitted_model : SARIMAXResults
    weekly_df    : DataFrame used for training (to derive naive forward exog)
    steps        : forecast horizon in weeks
    future_exog_df : If provided (backtest mode), use actual future exog values.
                     If None (live forecast), carry forward last observed values.

    Returns
    -------
    DataFrame with columns: week_end_date, forecast_log_volume, lower_ci,
                             upper_ci, forecast_volume
    """
    if future_exog_df is None:
        # Naive forward fill: carry last observed exog, zero out shock score
        last_row = weekly_df[EXOG_COLS].dropna().iloc[-1].copy()
        last_row["macro_shock_score"] = 0.0
        future_exog = pd.DataFrame([last_row.values] * steps, columns=EXOG_COLS)
    else:
        future_exog = future_exog_df[EXOG_COLS].copy()
        if len(future_exog) != steps:
            steps = len(future_exog)
        # Fill any NaN in test-window exog with last known training values
        last_train = weekly_df[EXOG_COLS].dropna().iloc[-1]
        future_exog = future_exog.fillna(last_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecast_obj = fitted_model.get_forecast(steps=steps, exog=future_exog)

    predicted = forecast_obj.predicted_mean.values
    ci = forecast_obj.conf_int(alpha=0.05)
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values

    # Build future dates (week-ending Fridays)
    last_date = weekly_df["week_end_date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq="W-FRI")

    # If future_exog_df was passed (backtest), use the test window dates instead
    if future_exog_df is not None and "week_end_date" in future_exog_df.columns:
        future_dates = future_exog_df["week_end_date"].values

    forecast_df = pd.DataFrame(
        {
            "week_end_date": future_dates,
            "forecast_log_volume": predicted,
            "lower_ci": lower,
            "upper_ci": upper,
        }
    )
    forecast_df["forecast_volume"] = np.exp(forecast_df["forecast_log_volume"])

    # Validate: no NaN allowed in output
    nan_cols = forecast_df[["forecast_log_volume", "lower_ci", "upper_ci"]].isna().any()
    if nan_cols.any():
        bad = nan_cols[nan_cols].index.tolist()
        raise ValueError(f"NaN detected in Model 1 forecast output columns: {bad}")

    return forecast_df
