import pandas as pd
import numpy as np
import yfinance as yf

from config import TICKER, FETCH_START, FETCH_END, RAW_DATA_PATH


def fetch_ihsg_daily() -> pd.DataFrame:
    """
    Fetches IHSG daily OHLCV from Yahoo Finance.
    Returns cleaned daily DataFrame and saves to RAW_DATA_PATH.
    """
    end = FETCH_END or pd.Timestamp.today().strftime("%Y-%m-%d")
    print(f"  Downloading {TICKER} from {FETCH_START} to {end} ...")

    raw = yf.download(TICKER, start=FETCH_START, end=end, auto_adjust=True, progress=False)

    # Flatten MultiIndex columns produced by yfinance (e.g. ('Close', '^JKSE'))
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Normalise column names to lowercase
    raw.columns = [c.lower() for c in raw.columns]

    # Drop rows where close is NaN
    raw = raw.dropna(subset=["close"])

    # Reset index so Date becomes a column
    raw = raw.reset_index()
    raw = raw.rename(columns={"index": "date", "Date": "date"})
    raw["date"] = pd.to_datetime(raw["date"])

    # Keep only the expected columns
    raw = raw[["date", "open", "high", "low", "close", "volume"]]

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


if __name__ == "__main__":
    df = fetch_ihsg_daily()
    print(df.tail())
