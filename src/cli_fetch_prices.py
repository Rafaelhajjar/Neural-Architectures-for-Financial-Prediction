"""CLI tool to fetch and cache prices from Yahoo Finance."""
from __future__ import annotations

import argparse
from pathlib import Path

from .data import get_prices


def main() -> None:
    """Fetch prices for a single ticker."""
    parser = argparse.ArgumentParser(
        description="Fetch and cache adjusted prices via yfinance with Parquet cache."
    )
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g., AAPL, MSFT, TSLA")
    parser.add_argument("--start", default="2015-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="1d", help="Interval, e.g., 1d, 1h, 1wk")
    parser.add_argument("--raw", action="store_true", help="Use unadjusted prices (default is adjusted)")
    parser.add_argument("--cache-dir", default="data/curated/prices", help="Cache directory root")
    parser.add_argument("--force", action="store_true", help="Force refresh from yfinance and overwrite cache")
    args = parser.parse_args()

    print(f"Fetching {args.ticker} from {args.start} to {args.end or 'today'}...")
    
    df = get_prices(
        args.ticker,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=not args.raw,
        cache_dir=Path(args.cache_dir),
        force_refresh=bool(args.force),
    )
    
    if df.empty:
        print(f"No data returned for {args.ticker}")
    else:
        print(f"\nLast 10 rows:")
        print(df.tail(10).to_string())
        print(f"\nRows: {len(df):,} | Columns: {list(df.columns)} | Adjusted: {not args.raw}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")


if __name__ == "__main__":
    main()

