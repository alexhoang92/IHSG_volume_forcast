import io
from pathlib import Path

import pandas as pd

from pipeline_runner import PIPELINE_DIR

VOLUME_PATH = PIPELINE_DIR / "data/raw/ihsg_volume.csv"


def _read_csv(content: bytes) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
        return df, None
    except Exception as exc:
        return None, f"Cannot parse CSV: {exc}"


def validate_volume(content: bytes) -> list[str]:
    df, err = _read_csv(content)
    if df is None:
        return [err]  # type: ignore[list-item]

    errors: list[str] = []
    required = {"Date", "volume"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        errors.append("Date column contains unparseable values")

    if not pd.api.types.is_numeric_dtype(df["volume"]):
        errors.append("volume column must be numeric")

    if df["Date"].notna().any():
        min_date = df["Date"].dropna().min()
        cutoff = pd.Timestamp("2023-01-01")
        if min_date > cutoff:
            errors.append(
                f"Volume data starts on {min_date.date()}, but must start on or before 2023-01-01"
            )

    return errors


def validate_macro(content: bytes) -> list[str]:
    df, err = _read_csv(content)
    if df is None:
        return [err]  # type: ignore[list-item]

    errors: list[str] = []
    required = {"week_end_date", "shock_score", "policy_rate"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors

    df["week_end_date"] = pd.to_datetime(df["week_end_date"], errors="coerce")
    if df["week_end_date"].isna().any():
        errors.append("week_end_date column contains unparseable dates")

    return errors


def validate_ipo(content: bytes) -> list[str]:
    df, err = _read_csv(content)
    if df is None:
        return [err]  # type: ignore[list-item]

    errors: list[str] = []
    required = {"announcement_date", "ticker", "company_name", "market_cap_idr_trillion"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors

    df["announcement_date"] = pd.to_datetime(df["announcement_date"], errors="coerce")
    if df["announcement_date"].isna().any():
        errors.append("announcement_date column contains unparseable dates")

    return errors


def validate_scenarios(content: bytes) -> list[str]:
    df, err = _read_csv(content)
    if df is None:
        return [err]  # type: ignore[list-item]

    errors: list[str] = []
    required = {"scenario", "week_end_date", "shock_score", "policy_rate"}
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing required columns: {sorted(missing)}")
        return errors

    df["week_end_date"] = pd.to_datetime(df["week_end_date"], errors="coerce")
    if df["week_end_date"].isna().any():
        errors.append("week_end_date column contains unparseable dates")

    valid_scenarios = {"BASE", "BULL", "BEAR"}
    if "scenario" in df.columns:
        unknown = set(df["scenario"].dropna().unique()) - valid_scenarios
        if unknown:
            errors.append(f"Unknown scenario names: {unknown}. Must be BASE, BULL, or BEAR.")

    # Cross-file check: scenarios must start after the historical volume data ends
    if VOLUME_PATH.exists() and df["week_end_date"].notna().any():
        try:
            vol_df = pd.read_csv(VOLUME_PATH, encoding="utf-8-sig", parse_dates=["Date"])
            last_hist_date = vol_df["Date"].max()
            scenario_start = df["week_end_date"].dropna().min()
            if scenario_start <= last_hist_date:
                errors.append(
                    f"Scenarios start on {scenario_start.date()} but historical volume data "
                    f"extends through {last_hist_date.date()}. "
                    "Scenarios must start after the last date in the volume file."
                )
        except Exception:
            pass  # volume file unreadable — skip cross-file check

    return errors


VALIDATORS = {
    "volume": validate_volume,
    "macro": validate_macro,
    "ipo": validate_ipo,
    "scenarios": validate_scenarios,
}


def validate_file(file_type: str, content: bytes) -> list[str]:
    validator = VALIDATORS.get(file_type)
    if validator is None:
        return [f"Unknown file type: {file_type}"]
    return validator(content)
