import os
import pandas as pd
import numpy as np
import yfinance as yf

from config import TICKER, FETCH_START, FETCH_END, RAW_DATA_PATH, VOLUME_INPUT_PATH


def _clean_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise columns returned by yfinance — returns price-only (no volume)."""
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.dropna(subset=["close"])
    raw = raw.reset_index()
    raw = raw.rename(columns={"index": "date", "Date": "date", "Datetime": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    return raw[["date", "open", "high", "low", "close"]]


def _load_external_volume(path: str = VOLUME_INPUT_PATH) -> pd.DataFrame:
    """
    Load the external IDR volume file (data/raw/ihsg_volume.csv).
    Returns a DataFrame with columns ["date", "volume"] where volume is in IDR.
    """
    vol = pd.read_csv(path, parse_dates=["Date"])
    vol = vol.rename(columns={"Date": "date"})
    vol["date"] = pd.to_datetime(vol["date"]).dt.tz_localize(None)
    vol["volume"] = pd.to_numeric(vol["volume"], errors="coerce")
    vol = vol.dropna(subset=["volume"])[["date", "volume"]].copy()
    return vol


def fetch_ihsg_daily() -> pd.DataFrame:
    """
    Fetches IHSG daily OHLCV from Yahoo Finance.
    Returns cleaned daily DataFrame and saves to RAW_DATA_PATH.
    """
    end = FETCH_END or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"  Downloading {TICKER} from {FETCH_START} to {end} ...")

    price_df = _clean_raw(
        yf.download(TICKER, start=FETCH_START, end=end, auto_adjust=True, progress=False)
    )

    if len(price_df) < 100:
        raise ValueError(
            f"Insufficient data: only {len(price_df)} rows returned for {TICKER}. "
            "Check the ticker symbol and date range."
        )

    # Merge with external IDR volume source
    vol_df = _load_external_volume()
    raw = price_df.merge(vol_df, on="date", how="inner")
    print(f"  Merged with external volume: {len(price_df)} price rows × {len(vol_df)} volume rows → {len(raw)} combined")

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
    # Keep only price columns — volume will be replaced from the external source
    price_cols = [c for c in ["date", "open", "high", "low", "close"] if c in existing.columns]
    existing_price = existing[price_cols].copy()

    last_date = existing_price["date"].max()
    today = pd.Timestamp.today().normalize()
    fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    fetch_end   = today.strftime("%Y-%m-%d")

    if fetch_start >= fetch_end:
        print(f"  Market data already up-to-date (last: {last_date.date()}). No new price data to fetch.")
        # Still re-merge with current external volume (catches any volume file updates)
        vol_df = _load_external_volume()
        combined = existing_price.merge(vol_df, on="date", how="inner")
        combined.to_csv(RAW_DATA_PATH, index=False)
        return combined

    print(f"  Incremental fetch: {fetch_start} → {fetch_end} ...")
    raw_new = yf.download(TICKER, start=fetch_start, end=fetch_end, auto_adjust=True, progress=False)

    if raw_new.empty:
        print("  No new trading days returned by yfinance.")
        vol_df = _load_external_volume()
        combined = existing_price.merge(vol_df, on="date", how="inner")
        combined.to_csv(RAW_DATA_PATH, index=False)
        return combined

    new_price_rows = _clean_raw(raw_new)

    all_price = (
        pd.concat([existing_price, new_price_rows], ignore_index=True)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Merge with external IDR volume source
    vol_df = _load_external_volume()
    combined = all_price.merge(vol_df, on="date", how="inner")

    if len(combined) < 100:
        raise ValueError(
            f"Insufficient data after merge: only {len(combined)} rows. "
            "Check the ticker symbol and date range."
        )

    business_days = pd.bdate_range(start=combined["date"].min(), end=combined["date"].max())
    actual_dates  = set(combined["date"].dt.normalize())
    missing = [d for d in business_days if d not in actual_dates]
    print(f"  Total {len(combined):,} rows | {combined['date'].min().date()} → {combined['date'].max().date()}")
    print(f"  New price rows appended: {len(new_price_rows)} | Missing business days (holidays): {len(missing)}")

    combined.to_csv(RAW_DATA_PATH, index=False)
    print(f"  Raw data saved → {RAW_DATA_PATH}")

    return combined


if __name__ == "__main__":
    df = fetch_ihsg_daily()
    print(df.tail())
