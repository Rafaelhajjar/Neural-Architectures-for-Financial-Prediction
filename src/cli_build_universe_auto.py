"""
Automated universe builder using iShares ETF holdings.

This script automatically downloads the full list of stocks from iShares ETFs
(no manual selection or scraping required).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from .universe.universe_sources import build_universe_from_ishares
from .universe.us_universe import UniverseBuildConfig, get_ticker_info
from .data import get_prices_bulk


def filter_universe(
    holdings_df: pd.DataFrame,
    config: UniverseBuildConfig,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter universe by market cap, price, volume, and exchange.
    
    Args:
        holdings_df: DataFrame from iShares with Ticker column
        config: Filter configuration
        verbose: Print progress
    
    Returns:
        Filtered DataFrame with metadata
    """
    tickers = holdings_df['Ticker'].unique().tolist()
    
    if verbose:
        print(f"\nFiltering {len(tickers)} tickers...")
        print(f"  Market cap: ${config.min_market_cap/1e6:.0f}M - ${config.max_market_cap/1e9:.1f}B")
        print(f"  Min price: ${config.min_price}")
        print(f"  Min avg dollar volume: ${config.min_avg_volume/1e6:.1f}M")
        print()
    
    results = []
    
    if verbose:
        try:
            from tqdm import tqdm
            ticker_iter = tqdm(tickers, desc="Filtering")
        except ImportError:
            ticker_iter = tickers
            print(f"Processing {len(tickers)} tickers...")
    else:
        ticker_iter = tickers
    
    for ticker in ticker_iter:
        info = get_ticker_info(ticker)
        
        # Apply filters
        market_cap = info.get("market_cap", 0)
        price = info.get("price", 0)
        avg_vol = max(info.get("avg_volume", 0), info.get("avg_volume_10d", 0))
        exchange = info.get("exchange", "")
        quote_type = info.get("quote_type", "")
        
        dollar_volume = price * avg_vol if price and avg_vol else 0
        
        pass_market_cap = config.min_market_cap <= market_cap <= config.max_market_cap if market_cap > 0 else False
        pass_price = price >= config.min_price if price > 0 else False
        pass_volume = dollar_volume >= config.min_avg_volume if dollar_volume > 0 else False
        pass_exchange = exchange in config.exchanges if exchange else False
        pass_quote_type = quote_type == "EQUITY"
        
        info["dollar_volume"] = dollar_volume
        info["pass_market_cap"] = pass_market_cap
        info["pass_price"] = pass_price
        info["pass_volume"] = pass_volume
        info["pass_exchange"] = pass_exchange
        info["pass_quote_type"] = pass_quote_type
        info["pass_all"] = all([pass_market_cap, pass_price, pass_volume, pass_exchange, pass_quote_type])
        
        results.append(info)
        time.sleep(0.1)  # Rate limiting
    
    df = pd.DataFrame(results)
    
    if verbose:
        print(f"\nFilter results:")
        print(f"  Total processed: {len(df)}")
        print(f"  Passed all filters: {df['pass_all'].sum()}")
        print(f"  Pass rate: {df['pass_all'].sum()/len(df)*100:.1f}%")
        print(f"\nFilter breakdown:")
        print(f"  Market cap: {df['pass_market_cap'].sum()}")
        print(f"  Price: {df['pass_price'].sum()}")
        print(f"  Volume: {df['pass_volume'].sum()}")
        print(f"  Exchange: {df['pass_exchange'].sum()}")
        print(f"  Quote type: {df['pass_quote_type'].sum()}")
    
    return df


def main() -> None:
    """Build universe from iShares ETF holdings."""
    parser = argparse.ArgumentParser(
        description="Build US stock universe from iShares ETF holdings (automatic, no scraping)"
    )
    parser.add_argument(
        "--etfs",
        nargs="+",
        default=["IWM", "IJR", "IJH"],
        help="iShares ETFs to use (IWM=Russell 2000, IJR=S&P 600, IJH=S&P 400, IWR=Russell MidCap)",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=300e6,
        help="Minimum market cap in dollars",
    )
    parser.add_argument(
        "--max-market-cap",
        type=float,
        default=10e9,
        help="Maximum market cap in dollars",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=5.0,
        help="Minimum stock price",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=1e6,
        help="Minimum average daily dollar volume",
    )
    parser.add_argument(
        "--out",
        default="data/universe/us_universe.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="Skip filtering, just save raw holdings",
    )
    parser.add_argument(
        "--prefetch",
        action="store_true",
        help="Fetch prices for filtered tickers",
    )
    parser.add_argument(
        "--start",
        default="2015-01-01",
        help="Start date for price fetch",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Building Universe from iShares ETF Holdings")
    print("=" * 80)
    print(f"\nETFs: {', '.join(args.etfs)}")
    print(f"Expected stocks: ~{sum({'IWM': 2000, 'IJR': 600, 'IJH': 400, 'IWR': 800}.get(e, 0) for e in args.etfs)}")
    
    # Step 1: Download holdings
    print("\n" + "-" * 80)
    print("Step 1: Downloading ETF Holdings")
    print("-" * 80)
    
    holdings = build_universe_from_ishares(
        args.etfs,
        output_path=Path(args.out).parent / "raw_holdings.csv",
        verbose=True
    )
    
    if holdings.empty:
        print("\nError: No holdings downloaded!")
        return
    
    # Step 2: Filter by criteria
    if not args.skip_filter:
        print("\n" + "-" * 80)
        print("Step 2: Filtering by Market Cap, Price, Volume")
        print("-" * 80)
        
        config = UniverseBuildConfig(
            include_sources=[],  # Not used
            min_market_cap=args.min_market_cap,
            max_market_cap=args.max_market_cap,
            min_price=args.min_price,
            min_avg_volume=args.min_volume,
            out_csv=Path(args.out),
        )
        
        filtered = filter_universe(holdings, config, verbose=True)
        
        # Save results
        config.out_csv.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(config.out_csv, index=False)
        
        # Save filtered-only version
        filtered_pass = filtered[filtered["pass_all"]].copy()
        filtered_path = config.out_csv.parent / f"{config.out_csv.stem}_filtered.csv"
        filtered_pass.to_csv(filtered_path, index=False)
        
        print(f"\nSaved:")
        print(f"  Full results: {config.out_csv}")
        print(f"  Filtered only: {filtered_path}")
        print(f"  Raw holdings: {Path(args.out).parent / 'raw_holdings.csv'}")
        
        # Step 3: Optional price fetch
        if args.prefetch:
            print("\n" + "-" * 80)
            print("Step 3: Fetching Prices")
            print("-" * 80)
            
            tickers = filtered_pass["ticker"].tolist()
            print(f"\nFetching prices for {len(tickers)} tickers from {args.start}...")
            print("This may take 15-30 minutes...\n")
            
            results = get_prices_bulk(
                tickers,
                start=args.start,
                interval="1d",
                auto_adjust=True,
                verbose=True,
            )
            
            success = sum(1 for df in results.values() if not df.empty)
            print(f"\nSuccessfully fetched: {success}/{len(tickers)} tickers")
    
    else:
        # Just save raw holdings
        holdings.to_csv(args.out, index=False)
        print(f"\nRaw holdings saved to: {args.out}")
    
    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

