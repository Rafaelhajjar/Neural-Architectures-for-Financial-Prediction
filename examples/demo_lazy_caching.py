"""
Demo: How lazy caching works (no pre-caching needed!)

The cache is built on-demand - only when you request data.
"""
import pandas as pd
from pathlib import Path
import time
from src.data import get_prices, get_prices_bulk

print("=" * 80)
print("Demo: Lazy Caching (On-Demand Only)")
print("=" * 80)

# Load universe
universe = pd.read_csv("data/universe/us_universe_ishares.csv")
print(f"\nUniverse size: {len(universe)} stocks")

# Pick 5 random stocks
tickers = ["NVDA", "AMD", "INTC", "QCOM", "AVGO"]
print(f"\nSelected tickers: {tickers}")

# Check which are already cached
cache_dir = Path("data/curated/prices/adj/1d")
cached = [t for t in tickers if (cache_dir / f"{t}.parquet").exists()]
not_cached = [t for t in tickers if t not in cached]

print(f"\nBefore fetching:")
print(f"  Already cached: {cached if cached else 'None'}")
print(f"  Not cached yet: {not_cached if not_cached else 'None'}")

# Now fetch data (cache is built automatically as needed)
print("\n" + "-" * 80)
print("Fetching data (cache builds on-demand)...")
print("-" * 80)

for ticker in tickers:
    start_time = time.time()
    df = get_prices(ticker, start="2023-01-01", end="2024-01-01")
    elapsed = time.time() - start_time
    
    cached_status = "from cache" if ticker in cached else "downloaded & cached"
    print(f"  {ticker}: {len(df)} rows, {elapsed:.3f}s ({cached_status})")

# Now fetch again (should be instant from cache)
print("\n" + "-" * 80)
print("Fetching same data again (should be instant from cache)...")
print("-" * 80)

for ticker in tickers:
    start_time = time.time()
    df = get_prices(ticker, start="2023-01-01", end="2024-01-01")
    elapsed = time.time() - start_time
    print(f"  {ticker}: {len(df)} rows, {elapsed:.3f}s ⚡ (from cache)")

# Show cache statistics
print("\n" + "-" * 80)
print("Cache Statistics")
print("-" * 80)

all_cached = list(cache_dir.glob("*.parquet"))
total_size = sum(f.stat().st_size for f in all_cached) / 1024 / 1024  # MB

print(f"  Total cached stocks: {len(all_cached)}")
print(f"  Cache size: {total_size:.2f} MB")
print(f"  Cache location: {cache_dir}")

print("\n" + "=" * 80)
print("Key Takeaway: Cache builds naturally as you request data!")
print("=" * 80)
print("\n✓ No pre-caching needed")
print("✓ First request downloads & caches")
print("✓ Subsequent requests are instant")
print("✓ Cache grows organically with usage")

