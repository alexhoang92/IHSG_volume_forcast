"""
scenario_engine.py — Parse scenario CSV and build future_exog_df per scenario.

Each scenario_df row defines macro conditions for one forecast week.
The output dicts map scenario names → future_exog_df DataFrames that are
directly accepted by forecast_model1(future_exog_df=...) and
forecast_model2(future_exog_df=...).
"""

import os
import warnings
import numpy as np
import pandas as pd

from config import SCENARIOS_INPUT_PATH, FORECAST_WEEKS, MACRO_SHOCK_MAX
from models.model1_volume import EXOG_COLS


def _auto_fridays(start_date: pd.Timestamp, steps: int) -> list:
    """Generate `steps` consecutive Fridays starting from the first Friday after start_date."""
    days_to_friday = (4 - start_date.weekday()) % 7
    if days_to_friday == 0:
        days_to_friday = 7
    first_friday = start_date + pd.Timedelta(days=days_to_friday)
    return [first_friday + pd.Timedelta(weeks=i) for i in range(steps)]


def _event_type_to_dummies(event_type_series: pd.Series) -> pd.DataFrame:
    """Convert event_type string series to d_geo, d_mp, d_trade columns (0/1)."""
    et = event_type_series.fillna("").str.strip().str.lower()
    return pd.DataFrame({
        "d_geo":   (et == "geopolitical").astype(float),
        "d_mp":    (et == "monetary_policy").astype(float),
        "d_trade": (et == "trade").astype(float),
    })


def build_future_exog(
    scenario_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    steps: int,
) -> pd.DataFrame:
    """
    Convert one scenario's raw rows into a future_exog_df compatible with
    forecast_model1() and forecast_model2().

    Steps:
      1. Start every row from last_obs = last row of weekly_df[EXOG_COLS]
      2. Override macro_shock_score and event dummies from scenario rows
      3. Compute interest_rate_direction using last 4 known policy_rates + scenario
      4. Fix log_trading_days = log(5); add week_end_date + trading_days columns
      5. Clip shock_score to [-2, 2]
      6. Pad short scenarios (< steps) by repeating last row; truncate long ones
    """
    last_obs = weekly_df[EXOG_COLS].dropna().iloc[-1].copy()
    last_date = weekly_df["week_end_date"].max()

    # Auto-generate future Friday dates
    future_dates = _auto_fridays(last_date, steps)

    # Assign auto-dates to scenario rows with blank week_end_date
    sdf = scenario_df.copy().reset_index(drop=True)
    if "week_end_date" not in sdf.columns or sdf["week_end_date"].isna().all():
        sdf["week_end_date"] = future_dates[:len(sdf)]
    else:
        blank = sdf["week_end_date"].isna()
        auto_idx = 0
        for i in sdf[blank].index:
            if auto_idx < len(future_dates):
                sdf.at[i, "week_end_date"] = future_dates[auto_idx]
                auto_idx += 1

    # Pad or truncate to exactly `steps` rows
    if len(sdf) < steps:
        last_row = sdf.iloc[[-1]]
        pad_rows = pd.concat([last_row] * (steps - len(sdf)), ignore_index=True)
        sdf = pd.concat([sdf, pad_rows], ignore_index=True)
        # Re-assign dates for padded rows
        for i in range(steps):
            sdf.at[i, "week_end_date"] = future_dates[i]
    elif len(sdf) > steps:
        scenario_name = sdf.get("scenario", pd.Series(["?"])).iloc[0]
        print(f"  WARNING: Scenario '{scenario_name}' has {len(sdf)} rows > FORECAST_WEEKS={steps}. Truncating.")
        sdf = sdf.iloc[:steps].copy()

    # Build exog rows starting from last observed values
    # Seed lag history from last 2 weeks of training data
    if "macro_neg_shock" in weekly_df.columns:
        hist_neg = weekly_df["macro_neg_shock"].dropna().tolist()
    else:
        hist_neg = [0.0, 0.0]
    neg_shock_window = hist_neg[-3:] if len(hist_neg) >= 3 else ([0.0] * (3 - len(hist_neg)) + hist_neg)

    if "macro_pos_shock" in weekly_df.columns:
        hist_pos = weekly_df["macro_pos_shock"].dropna().tolist()
    else:
        hist_pos = [0.0, 0.0, 0.0]
    pos_shock_window = hist_pos[-3:] if len(hist_pos) >= 3 else ([0.0] * (3 - len(hist_pos)) + hist_pos)

    rows = []
    for i in range(steps):
        row = last_obs.copy()
        row["log_trading_days"] = np.log(5)  # assume standard 5-day week

        # Compute scenario shock for this week (allow ±MACRO_SHOCK_MAX for structural breaks)
        raw_shock = 0.0
        if "shock_score" in sdf.columns:
            val = sdf.at[i, "shock_score"]
            try:
                fval = float(val)
                raw_shock = 0.0 if (fval != fval) else fval  # NaN check: NaN != NaN
            except (TypeError, ValueError):
                raw_shock = 0.0
        raw_shock = float(np.clip(raw_shock, -MACRO_SHOCK_MAX, MACRO_SHOCK_MAX))

        # Decomposed shock variables
        row["macro_shock_abs"] = abs(raw_shock)
        row["macro_neg_lag1"]  = neg_shock_window[-1]
        row["macro_neg_lag2"]  = neg_shock_window[-2] if len(neg_shock_window) >= 2 else 0.0
        row["macro_neg_lag3"]  = neg_shock_window[-3] if len(neg_shock_window) >= 3 else 0.0
        row["macro_pos_lag1"]  = pos_shock_window[-1]
        row["macro_pos_lag2"]  = pos_shock_window[-2] if len(pos_shock_window) >= 2 else 0.0
        row["macro_pos_lag3"]  = pos_shock_window[-3] if len(pos_shock_window) >= 3 else 0.0

        # Slide shock windows forward for next iteration
        neg_shock_window.append(max(0.0, -raw_shock))
        pos_shock_window.append(max(0.0,  raw_shock))

        # Override event type dummies
        if "event_type" in sdf.columns:
            et = str(sdf.at[i, "event_type"]).strip().lower()
            row["d_geo"]   = 1.0 if et == "geopolitical"    else 0.0
            row["d_mp"]    = 1.0 if et == "monetary_policy" else 0.0
            row["d_trade"] = 1.0 if et == "trade"           else 0.0

        rows.append(row)

    exog_df = pd.DataFrame(rows, columns=EXOG_COLS).reset_index(drop=True)

    # Compute interest_rate_direction from scenario policy_rate
    if "policy_rate" in sdf.columns and macro_df is not None and len(macro_df) > 0:
        # Get last 4 known policy rates from macro_df
        known_rates = macro_df["policy_rate"].dropna().tail(4).tolist() if "policy_rate" in macro_df.columns else []
        for i in range(steps):
            raw = sdf.at[i, "policy_rate"]
            try:
                future_rate = float(raw)
            except (TypeError, ValueError):
                future_rate = known_rates[-1] if known_rates else 0.0
            window = known_rates[-3:] + [future_rate] if len(known_rates) >= 3 else known_rates + [future_rate]
            if len(window) >= 4:
                ir_dir = window[-1] - window[-4]
            elif len(window) > 1:
                ir_dir = window[-1] - window[0]
            else:
                ir_dir = exog_df.at[i, "interest_rate_direction"]  # carry forward
            exog_df.at[i, "interest_rate_direction"] = ir_dir
            # Slide window forward for next iteration
            known_rates = known_rates + [future_rate]

    # Add date and trading_days columns (consumed by forecast_model1 lines 100-108)
    exog_df.insert(0, "week_end_date", [d for d in future_dates[:steps]])
    exog_df["trading_days"] = 5.0

    return exog_df


def load_scenarios(path: str) -> dict:
    """
    Parse scenarios CSV.
    Returns {'BASE': df_base, 'BULL': df_bull, ...}
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_csv(path, comment="#")

    if "scenario" not in raw.columns:
        raise ValueError(f"Scenarios CSV at '{path}' must have a 'scenario' column.")

    # Parse week_end_date column if present
    if "week_end_date" in raw.columns:
        raw["week_end_date"] = pd.to_datetime(raw["week_end_date"], errors="coerce")

    scenarios = {}
    for name, grp in raw.groupby("scenario"):
        scenarios[str(name)] = grp.reset_index(drop=True)

    print(f"  [Scenarios] Loaded {len(scenarios)} scenario(s): {', '.join(scenarios.keys())}")
    return scenarios


def get_all_scenarios(
    scenario_path,
    macro_df,
    weekly_df: pd.DataFrame,
    steps: int = FORECAST_WEEKS,
) -> dict:
    """
    Top-level entry: load scenarios CSV and build future_exog_df for each scenario.

    Returns dict of {scenario_name: future_exog_df}.
    Falls back to {'BASE': None} if no scenario file is found,
    which triggers the existing neutral forward-fill in forecast_model1.
    """
    # Determine file to load
    if scenario_path == "":
        # Explicitly disabled via --scenarios ""
        print("  [Scenarios] --scenarios '' supplied — using neutral BASE forecast.")
        return {"BASE": None}

    candidates = []
    if scenario_path:
        candidates.append(scenario_path)
    candidates.append(SCENARIOS_INPUT_PATH)

    path_to_use = None
    for c in candidates:
        if os.path.exists(c):
            path_to_use = c
            break

    if path_to_use is None:
        print("  [Scenarios] No scenarios file found — using neutral BASE forecast.")
        return {"BASE": None}

    print(f"  [Scenarios] Reading scenarios from {path_to_use}")
    raw_scenarios = load_scenarios(path_to_use)

    result = {}
    for name, sdf in raw_scenarios.items():
        exog = build_future_exog(sdf, weekly_df, macro_df, steps)
        result[name] = exog

    return result
