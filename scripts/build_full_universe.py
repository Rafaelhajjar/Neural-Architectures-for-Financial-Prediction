"""Build a full universe of small-to-mid cap US stocks.

This script will:
1. Scrape Russell 2000, S&P 600, and S&P 400 from Wikipedia
2. Filter by market cap, price, volume, and exchange
3. Optionally fetch prices for all filtered stocks

WARNING: This will take 30-60 minutes as it fetches metadata for ~2000+ stocks.
"""
from pathlib import Path
from src.universe.us_universe import UniverseBuildConfig, build_us_universe
from src.data import get_prices_bulk
import argparse

def main():
    parser = argparse.ArgumentParser(description="Build full US stock universe")
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Fetch prices after building universe (adds 15-30 min)",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date for price fetch",
    )
    args = parser.parse_args()
    
    # Configuration for full universe
    config = UniverseBuildConfig(
        include_sources=["russell_2000", "sp600", "sp400"],
        min_market_cap=300e6,  # $300M
        max_market_cap=10e9,   # $10B
        min_price=5.0,
        min_avg_volume=1e6,    # $1M average daily volume
        out_csv=Path("data/universe/us_universe_full.csv"),
    )
    
    print("\n" + "=" * 80)
    print("Building Full US Stock Universe")
    print("=" * 80)
    print("\nThis will scrape Russell 2000, S&P 600, and S&P 400 constituents")
    print("and filter for small-to-mid cap stocks.")
    print("\nEstimated time: 30-60 minutes")
    print("\nFilters:")
    print(f"  Market cap: ${config.min_market_cap/1e6:.0f}M - ${config.max_market_cap/1e9:.1f}B")
    print(f"  Min price: ${config.min_price}")
    print(f"  Min avg dollar volume: ${config.min_avg_volume/1e6:.1f}M")
    print(f"  Exchanges: NYSE, NASDAQ, AMEX")
    print("\n" + "=" * 80)
    
    # Ask for confirmation
    response = input("\nContinue? (yes/no): ")
    if response.lower() not in ["yes", "y"]:
        print("Aborted.")
        return
    
    # Build universe
    df = build_us_universe(config, verbose=True)
    
    # Optionally prefetch prices
    if args.prefetch:
        print("\n" + "=" * 80)
        print("Prefetching Prices")
        print("=" * 80)
        
        tickers = df[df["pass_all"]]["ticker"].tolist()
        print(f"\nFetching prices for {len(tickers)} tickers from {args.start}...")
        print("Estimated time: 15-30 minutes")
        print()
        
        response = input("Continue with price fetch? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Skipping price fetch.")
            return
        
        results = get_prices_bulk(
            tickers,
            start=args.start,
            interval="1d",
            auto_adjust=True,
            verbose=True,
        )
        
        success = sum(1 for df in results.values() if not df.empty)
        print(f"\nSuccessfully fetched: {success}/{len(tickers)} tickers")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)
    print(f"\nResults saved to:")
    print(f"  Full: {config.out_csv}")
    print(f"  Filtered: {config.out_csv.parent / f'{config.out_csv.stem}_filtered.csv'}")
    print("\nTo fetch prices later, run:")
    print(f"  python -m src.cli_bulk_fetch --universe {config.out_csv.parent / f'{config.out_csv.stem}_filtered.csv'} --start 2015-01-01")

if __name__ == "__main__":
    main()

