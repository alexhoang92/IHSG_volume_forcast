import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import SARIMAX_ORDER, SARIMAX_SEASONAL_ORDER, FORECAST_WEEKS

# Exogenous columns used by Model 1 — defined once, reused by backtest
EXOG_COLS = [
    "weekly_return",
    "realized_volatility",
    "macro_shock_abs",         # |shock_score| — announcement-week spike (both pos & neg events)
    "macro_neg_lag1",          # max(0, -shock_score) lag 1 wk — residual panic selling
    "macro_neg_lag2",          # max(0, -shock_score) lag 2 wks — fear/uncertainty suppression
    "macro_neg_lag3",          # max(0, -shock_score) lag 3 wks — structural break tail (MSCI downgrade)
    "macro_pos_lag1",          # max(0, +shock_score) lag 1 wk — fund rebalancing wave 1
    "macro_pos_lag2",          # max(0, +shock_score) lag 2 wks — sustained buying
    "macro_pos_lag3",          # max(0, +shock_score) lag 3 wks — MSCI effective-date second wave
    "interest_rate_direction",
    "d_geo",
    "d_mp",
    "d_trade",
    "log_trading_days",   # calendar swing adjustment: log(trading_days) per week
]


def fit_model1(weekly_df: pd.DataFrame):
    """
    Fits SARIMAX on full weekly_df.
    Returns fitted SARIMAXResults object.
    """
    df = weekly_df[["log_volume_adj"] + EXOG_COLS].dropna().copy()

    model = SARIMAX(
        endog=df["log_volume_adj"],
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
        last_row["macro_shock_abs"] = 0.0
        last_row["macro_neg_lag1"]  = 0.0
        last_row["macro_neg_lag2"]  = 0.0
        last_row["macro_neg_lag3"]  = 0.0
        last_row["macro_pos_lag1"]  = 0.0
        last_row["macro_pos_lag2"]  = 0.0
        last_row["macro_pos_lag3"]  = 0.0
        # Assume standard 5-day trading week; override log_trading_days if known holiday week
        last_row["log_trading_days"] = np.log(5)
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

    # Determine expected trading days per forecast step for volume back-conversion
    if future_exog_df is not None and "week_end_date" in future_exog_df.columns:
        future_dates = future_exog_df["week_end_date"].values
        # Use actual trading days from test window (backtest mode)
        if "trading_days" in future_exog_df.columns:
            future_trading_days = future_exog_df["trading_days"].fillna(5).values
        else:
            future_trading_days = np.full(steps, 5.0)
    else:
        future_trading_days = np.full(steps, 5.0)  # standard week assumption

    forecast_df = pd.DataFrame(
        {
            "week_end_date": future_dates,
            "forecast_log_volume": predicted,   # log(avg daily volume)
            "lower_ci": lower,
            "upper_ci": upper,
        }
    )
    # Recover total weekly volume: exp(log_volume_adj) * trading_days
    forecast_df["forecast_volume"] = np.exp(forecast_df["forecast_log_volume"]) * future_trading_days

    # Validate: no NaN allowed in output
    nan_cols = forecast_df[["forecast_log_volume", "lower_ci", "upper_ci"]].isna().any()
    if nan_cols.any():
        bad = nan_cols[nan_cols].index.tolist()
        raise ValueError(f"NaN detected in Model 1 forecast output columns: {bad}")

    return forecast_df


def compute_contribution_analysis(
    fitted_model,
    future_exog_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decompose Model 1 forecast into per-variable contributions.

    For each forecast week t:
        forecast_log_volume_t = intercept + Σ(β_i × x_i_t) + ARMA_component_t

    contribution of variable x_i at step t = β_i × x_i_t  (log-volume units)
    ARMA component = forecast_log_volume - intercept - Σ(exog contributions)

    Parameters
    ----------
    fitted_model   : SARIMAXResults
    future_exog_df : DataFrame with EXOG_COLS (output of scenario_engine or forecast_model1)
    forecast_df    : DataFrame output of forecast_model1() (must have week_end_date, forecast_log_volume)

    Returns
    -------
    DataFrame with columns:
        week_end_date, contrib_{col} for each col in EXOG_COLS,
        contrib_intercept, contrib_arma, exog_total, forecast_log_volume
    """
    params = fitted_model.params

    # Extract intercept / const (statsmodels uses either name depending on trend setting)
    const = 0.0
    for pname in ["const", "intercept"]:
        if pname in params.index:
            const = float(params[pname])
            break

    # Mirror the fillna applied inside forecast_model1: fill NaN exog with last training values
    last_train = fitted_model.model.data.exog[-1]  # last row used during fit (numpy array)
    last_train_series = pd.Series(last_train, index=EXOG_COLS)

    exog = future_exog_df[EXOG_COLS].reset_index(drop=True).fillna(last_train_series)
    steps = min(len(exog), len(forecast_df))

    rows = []
    for i in range(steps):
        contrib_row = {"week_end_date": forecast_df["week_end_date"].iloc[i]}
        exog_total = 0.0
        for col in EXOG_COLS:
            coef = float(params[col]) if col in params.index else 0.0
            val = float(exog[col].iloc[i]) if col in exog.columns else 0.0
            contrib = coef * val
            contrib_row[f"contrib_{col}"] = round(contrib, 6)
            exog_total += contrib

        forecast_lv = float(forecast_df["forecast_log_volume"].iloc[i])
        contrib_row["contrib_intercept"] = round(const, 6)
        contrib_row["contrib_arma"] = round(forecast_lv - const - exog_total, 6)
        contrib_row["exog_total"] = round(exog_total, 6)
        contrib_row["forecast_log_volume"] = round(forecast_lv, 6)
        rows.append(contrib_row)

    return pd.DataFrame(rows)


def compute_sensitivity_analysis(
    fitted_model,
    weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute sensitivity of Model 1 forecast to each exogenous variable.

    For each variable:
        1-std log impact = coefficient × historical_std
        1-std vol pct   = (exp(1-std log impact) - 1) × 100

    Parameters
    ----------
    fitted_model : SARIMAXResults
    weekly_df    : DataFrame used for training (to compute historical std)

    Returns
    -------
    DataFrame with columns:
        variable, coefficient, historical_std, 1std_log_impact, 1std_vol_pct
    Sorted by abs(1std_log_impact) descending.
    """
    params = fitted_model.params
    hist = weekly_df[EXOG_COLS].dropna()

    rows = []
    for col in EXOG_COLS:
        coef = float(params[col]) if col in params.index else 0.0
        std = float(hist[col].std()) if col in hist.columns else 0.0
        one_std_log = coef * std
        one_std_pct = (np.exp(one_std_log) - 1) * 100
        rows.append({
            "variable": col,
            "coefficient": round(coef, 6),
            "historical_std": round(std, 6),
            "1std_log_impact": round(one_std_log, 6),
            "1std_vol_pct": round(one_std_pct, 2),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("1std_log_impact", key=abs, ascending=False).reset_index(drop=True)
    return df
