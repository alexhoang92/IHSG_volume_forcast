import os
import pandas as pd
import numpy as np
import yfinance as yf

from config import TICKER, FETCH_START, FETCH_END, RAW_DATA_PATH


def _clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns returned by yfinance into the standard 6-column schema."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.dropna(subset=["close"])
    raw = raw.reset_index()
    raw = raw.rename(columns={"index": "date", "Date": "date", "Datetime": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    return raw[["date", "open", "high", "low", "close", "volume"]]


def fetch_ihsg_daily() -> pd.DataFrame:
    """
    Fetches IHSG daily OHLCV from Yahoo Finance.
    Returns cleaned daily DataFrame and saves to RAW_DATA_PATH.
    """
    end = FETCH_END or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"  Downloading {TICKER} from {FETCH_START} to {end} ...")

    raw = yf.download(TICKER, start=FETCH_START, end=end, auto_adjust=True, progress=False)
    raw = _clean_raw(raw)

    if len(raw) < 100:
        raise ValueError(
            f"Insufficient data: only {len(raw)} rows returned for {TICKER}. "
            "Check the ticker symbol and date range."
        )

    # Detect missing business days (expected gaps = Indonesian public holidays)
    business_days = pd.bdate_range(start=raw["date"].min(), end=raw["date"].max())
    actual_dates = set(raw["date"].dt.normalize())
    missing = [d for d in business_days if d not in actual_dates]
    print(f"  Fetched {len(raw):,} rows | {raw['date'].min().date()} → {raw['date'].max().date()}")
    print(f"  Missing business days (holidays/closures): {len(missing)}")

    raw.to_csv(RAW_DATA_PATH, index=False)
    print(f"  Raw data saved → {RAW_DATA_PATH}")

    return raw


def fetch_ihsg_incremental() -> pd.DataFrame:
    """
    Append-only market data fetch.

    Loads the existing raw CSV, determines the last date already stored, then
    downloads only the days from (last_date + 1) to today from Yahoo Finance.
    New rows are appended, the combined set is deduplicated and re-saved.

    Falls back to a full fetch via fetch_ihsg_daily() if no existing file is found.
    """
    if not os.path.exists(RAW_DATA_PATH):
        print("  No existing raw data file found — running full fetch.")
        return fetch_ihsg_daily()

    existing = pd.read_csv(RAW_DATA_PATH, parse_dates=["date"])
    last_date = existing["date"].max()
    today = pd.Timestamp.today().normalize()
    fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = today.strftime("%Y-%m-%d")

    if fetch_start >= fetch_end:
        print(f"  Market data already up-to-date (last: {last_date.date()}). No new data to fetch.")
        return existing

    print(f"  Incremental fetch: {fetch_start} → {fetch_end} ...")
    raw_new = yf.download(TICKER, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)

    if raw_new.empty:
        print("  No new trading days returned by yfinance.")
        return existing

    new_rows = _clean_raw(raw_new)

    combined = (
        pd.concat([existing, new_rows], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(combined) < 100:
        raise ValueError(
            f"Insufficient data after merge: only {len(combined)} rows. "
            "Check the ticker symbol and date range."
        )

    business_days = pd.bdate_range(start=combined["date"].min(), end=combined["date"].max())
    actual_dates  = set(combined["date"].dt.normalize())
    missing = [d for d in business_days if d not in actual_dates]
    print(f"  Total {len(combined):,} rows | {combined['date'].min().date()} → {combined['date'].max().date()}")
    print(f"  New rows appended: {len(new_rows)} | Missing business days (holidays): {len(missing)}")

    combined.to_csv(RAW_DATA_PATH, index=False)
    print(f"  Raw data saved → {RAW_DATA_PATH}")

    return combined


if __name__ == "__main__":
    df = fetch_ihsg_daily()
    print(df.tail())
