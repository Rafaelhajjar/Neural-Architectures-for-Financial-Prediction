"""CLI tool to bulk fetch prices from a universe CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import get_prices_bulk


def main() -> None:
    """Bulk fetch prices for tickers in a universe CSV."""
    parser = argparse.ArgumentParser(
        description="Bulk fetch prices for all tickers in a universe CSV."
    )
    parser.add_argument(
        "--universe",
        required=True,
        help="Path to universe CSV file",
    )
    parser.add_argument(
        "--ticker-column",
        default="ticker",
        help="Name of ticker column in CSV (default: ticker)",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--interval",
        default="1d",
        help="Interval (default: 1d)",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/curated/prices",
        help="Cache directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force refresh all tickers",
    )
    args = parser.parse_args()

    # Load universe
    universe_path = Path(args.universe)
    if not universe_path.exists():
        print(f"Error: Universe file not found: {universe_path}")
        return

    df_universe = pd.read_csv(universe_path)
    
    if args.ticker_column not in df_universe.columns:
        print(f"Error: Column '{args.ticker_column}' not found in CSV")
        print(f"Available columns: {list(df_universe.columns)}")
        return

    tickers = df_universe[args.ticker_column].dropna().unique().tolist()
    
    print(f"Loaded {len(tickers)} tickers from {universe_path}")
    print(f"Fetching prices from {args.start} to {args.end or 'today'}...")
    print()

    # Fetch prices
    results = get_prices_bulk(
        tickers,
        start=args.start,
        end=args.end,
        interval=args.interval,
        auto_adjust=True,
        cache_dir=Path(args.cache_dir),
        force_refresh=args.force,
        verbose=True,
    )

    # Summary
    success = sum(1 for df in results.values() if not df.empty)
    failed = len(tickers) - success
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total tickers: {len(tickers)}")
    print(f"Successfully fetched: {success}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        failed_tickers = [t for t, df in results.items() if df.empty]
        print(f"\nFailed tickers: {', '.join(failed_tickers[:20])}")
        if len(failed_tickers) > 20:
            print(f"... and {len(failed_tickers) - 20} more")


if __name__ == "__main__":
    main()

