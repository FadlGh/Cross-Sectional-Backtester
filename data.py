"""Download and cache daily adjusted prices.

Usage:
    python data.py                      # 2014-01-01 to today, saves to data/
    python data.py --start 2012-01-01   # different start date

Outputs:
    data/close.parquet   adjusted close, index = date, columns = tickers
    data/volume.parquet  daily volume, same shape
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", module="yfinance")

# 50 US large caps that traded for the whole 2014+ period.
# Known limitation: these are TODAY's large caps, so the universe has
# survivorship bias (see README).
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "PG",
    "UNH", "HD", "MA", "XOM", "CVX", "PFE", "KO", "PEP", "MRK", "ABBV",
    "BAC", "WMT", "DIS", "CSCO", "ORCL", "INTC", "CMCSA", "VZ", "T", "ADBE",
    "CRM", "NFLX", "COST", "MCD", "NKE", "TXN", "QCOM", "AMGN", "HON", "IBM",
    "CAT", "GS", "MS", "LLY", "MDT", "BA", "MMM", "UPS", "LOW", "SBUX",
]

MAX_MISSING = 0.05  # drop a ticker if more than 5% of its days are missing


def download(tickers, start, end=None):
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # prices adjusted for splits and dividends
        progress=False,
        group_by="column",
    )
    close = raw["Close"].sort_index()
    volume = raw["Volume"].sort_index()
    return close, volume


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    close, volume = download(TICKERS, args.start, args.end)

    missing = close.isna().mean()
    bad = missing[missing > MAX_MISSING].index.tolist()
    if bad:
        print(f"Dropping tickers with >{MAX_MISSING:.0%} missing days: {bad}")
        close = close.drop(columns=bad)
        volume = volume.drop(columns=bad)

    # No forward-filling here on purpose: decide how to handle gaps in step 2.
    close.to_parquet(out / "close.parquet")
    volume.to_parquet(out / "volume.parquet")

    print(f"Saved {close.shape[1]} tickers, {close.shape[0]} days "
          f"({close.index[0].date()} to {close.index[-1].date()}) to {out}/")


if __name__ == "__main__":
    main()
