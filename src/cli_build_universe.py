"""CLI tool to build stock universe and fetch prices."""
from __future__ import annotations

import argparse
from pathlib import Path

from .universe.us_universe import UniverseBuildConfig, build_us_universe
from .data import get_prices_bulk


def main() -> None:
    """Build US stock universe and optionally prefetch prices."""
    parser = argparse.ArgumentParser(
        description="Build a US small-to-mid cap stock universe and optionally prefetch prices."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["russell_2000", "sp600", "sp400"],
        help="Sources to include (russell_2000, sp600, sp400)",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=300e6,
        help="Minimum market cap in dollars (default: 300M)",
    )
    parser.add_argument(
        "--max-market-cap",
        type=float,
        default=10e9,
        help="Maximum market cap in dollars (default: 10B)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=5.0,
        help="Minimum stock price (default: 5.0)",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1e6,
        help="Minimum average daily dollar volume (default: 1M)",
    )
    parser.add_argument(
        "--out",
        default="data/universe/us_universe.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Fetch prices for all filtered tickers",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date for price fetch",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="End date for price fetch",
    )
    args = parser.parse_args()

    # Build configuration
    cfg = UniverseBuildConfig(
        include_sources=args.sources,
        min_market_cap=args.min_market_cap,
        max_market_cap=args.max_market_cap,
        min_price=args.min_price,
        min_avg_volume=args.min_volume,
        out_csv=Path(args.out),
    )

    # Build universe
    print("=" * 80)
    print("Building US Stock Universe")
    print("=" * 80)
    df = build_us_universe(cfg, verbose=True)

    # Optionally prefetch prices
    if args.prefetch:
        print("\n" + "=" * 80)
        print("Prefetching Prices")
        print("=" * 80)
        
        # Get filtered tickers
        tickers = df[df["pass_all"]]["ticker"].tolist()
        print(f"\nFetching prices for {len(tickers)} tickers from {args.start} to {args.end or 'today'}...")
        
        results = get_prices_bulk(
            tickers,
            start=args.start,
            end=args.end,
            interval="1d",
            auto_adjust=True,
            verbose=True,
        )
        
        # Summary
        success = sum(1 for df in results.values() if not df.empty)
        print(f"\nSuccessfully fetched: {success}/{len(tickers)} tickers")
        print(f"Failed: {len(tickers) - success} tickers")


if __name__ == "__main__":
    main()

