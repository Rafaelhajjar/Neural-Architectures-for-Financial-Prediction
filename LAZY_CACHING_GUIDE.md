# Lazy Caching Guide - No Pre-Caching Needed! ⚡

## TL;DR

**You don't need to pre-cache anything!** The cache builds automatically when you request data.

## How It Works

### ✅ First Request: Downloads & Caches
```python
from src.data import get_prices

# First time accessing AAPL
df = get_prices("AAPL", start="2020-01-01")
# → Downloads from Yahoo Finance (1-2 seconds)
# → Saves to data/curated/prices/adj/1d/AAPL.parquet
# → Returns data
```

### ✅ Second Request: Instant from Cache
```python
# Second time accessing AAPL
df = get_prices("AAPL", start="2020-01-01")
# → Reads from cache (0.01 seconds) ⚡
# → No network request
# → Returns data
```

### ✅ Incremental Updates
```python
# Later, request more recent data
df = get_prices("AAPL", start="2020-01-01", end="2024-12-31")
# → Reads cached data up to last date
# → Downloads only new dates from Yahoo
# → Updates cache
# → Returns combined data
```

## Recommended Workflow

### For Your ML Project

**DON'T do this (pre-caching):**
```bash
# ❌ This takes 30-60 minutes and you may not need all stocks
python -m src.cli_bulk_fetch \
  --universe data/universe/us_universe_ishares.csv \
  --start 2015-01-01
```

**DO this instead (lazy loading):**
```python
import pandas as pd
from src.data.panel import build_adj_close_panel

# 1. Load universe
universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# 2. Start with a subset for development
tickers = universe["Ticker"].head(50).tolist()  # First 50 stocks

# 3. Build features (cache builds naturally)
prices = build_adj_close_panel(tickers, start="2020-01-01")
# → Only these 50 stocks are fetched and cached
# → Takes 1-2 minutes instead of 30-60 minutes

# 4. Once your pipeline works, scale up
tickers = universe["Ticker"].head(500).tolist()  # Expand to 500
prices = build_adj_close_panel(tickers, start="2020-01-01")
# → Only new stocks are fetched, previously cached ones are instant
```

## Cache Management

### Check Cache Size
```bash
du -sh data/curated/prices/adj/1d
```

### See What's Cached
```bash
ls data/curated/prices/adj/1d/ | wc -l  # Count cached stocks
```

### Clear Cache (if needed)
```bash
rm -rf data/curated/prices/adj/1d/*
```

### Force Refresh a Ticker
```python
from src.data import get_prices

# Force re-download (ignores cache)
df = get_prices("AAPL", start="2020-01-01", force_refresh=True)
```

## Real-World Example

### Scenario: Building ML Features for 100 Stocks

**Step 1: Define your universe**
```python
import pandas as pd

universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# Filter to tech sector
tech = universe[universe["Sector"] == "Information Technology"]
tickers = tech["Ticker"].head(100).tolist()

print(f"Selected {len(tickers)} tech stocks")
```

**Step 2: Load data (cache builds naturally)**
```python
from src.data.panel import build_adj_close_panel, compute_returns

# First time: Downloads & caches (takes ~2 minutes)
prices = build_adj_close_panel(tickers, start="2020-01-01")
returns = compute_returns(prices, periods=1)

# Save for later
prices.to_parquet("data/processed/tech_prices.parquet")
```

**Step 3: Next day, use cached data (instant)**
```python
# All 100 stocks are now cached
prices = build_adj_close_panel(tickers, start="2020-01-01")
# → Takes ~1 second instead of 2 minutes!
```

**Step 4: Add more stocks incrementally**
```python
# Add 50 more stocks
new_tickers = tech["Ticker"].iloc[100:150].tolist()
all_tickers = tickers + new_tickers

# Only the 50 new stocks are downloaded
prices = build_adj_close_panel(all_tickers, start="2020-01-01")
# → First 100 from cache (instant)
# → New 50 downloaded (~1 minute)
```

## Advantages of Lazy Caching

✅ **Fast startup** - Start working immediately  
✅ **Efficient** - Only download what you need  
✅ **Incremental** - Cache grows with usage  
✅ **Flexible** - Easy to experiment with different subsets  
✅ **Automatic** - No manual cache management

## When to Use Bulk Pre-Fetch (Optional)

You might want to pre-fetch if:
1. **Overnight batch job** - Let it run while you sleep
2. **Known universe** - You know you'll need all stocks
3. **Shared environment** - Multiple people using same cache
4. **Slow internet** - Cache once, use many times

But for development and experimentation, **lazy caching is better!**

## Performance Comparison

| Approach | Initial Setup | First Use | Subsequent Use | Flexibility |
|----------|---------------|-----------|----------------|-------------|
| **Pre-cache all 2,343 stocks** | 30-60 min | Instant | Instant | Low |
| **Lazy cache (50 stocks)** | 0 min | 1-2 min | Instant | High ✓ |
| **Lazy cache (500 stocks)** | 0 min | 10-15 min | Instant | High ✓ |

## Code Examples

### Example 1: Sector Analysis
```python
from src.data.panel import build_adj_close_panel, compute_momentum
import pandas as pd

universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# Analyze Healthcare sector
healthcare = universe[universe["Sector"] == "Health Care"]
tickers = healthcare["Ticker"].head(100).tolist()

# Cache builds on-demand
prices = build_adj_close_panel(tickers, start="2020-01-01")
momentum = compute_momentum(prices, lookback=126)

print(f"Top 10 momentum stocks:")
print(momentum.iloc[-1].sort_values(ascending=False).head(10))
```

### Example 2: Momentum Strategy Backtest
```python
from src.data.panel import (
    build_adj_close_panel,
    compute_momentum,
    rank_cross_sectional
)
import pandas as pd

universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# Start with liquid stocks (top by market value)
universe_sorted = universe.sort_values("Market Value", ascending=False)
tickers = universe_sorted["Ticker"].head(200).tolist()

# Load data (cache builds as needed)
prices = build_adj_close_panel(tickers, start="2020-01-01")

# Compute momentum ranks
momentum = compute_momentum(prices, lookback=126)
ranks = rank_cross_sectional(momentum)

# Long top 20, short bottom 20
long_positions = ranks.iloc[-1].nlargest(20)
short_positions = ranks.iloc[-1].nsmallest(20)

print(f"Long: {long_positions.index.tolist()}")
print(f"Short: {short_positions.index.tolist()}")
```

### Example 3: Add New Stocks Later
```python
# Week 1: Start with 50 stocks
tickers_week1 = ["AAPL", "MSFT", ...]  # 50 stocks
prices = build_adj_close_panel(tickers_week1, start="2020-01-01")

# Week 2: Add 50 more (only new ones are downloaded)
tickers_week2 = tickers_week1 + ["GOOGL", "AMZN", ...]  # 100 stocks
prices = build_adj_close_panel(tickers_week2, start="2020-01-01")
# → First 50 from cache (instant)
# → New 50 downloaded

# Week 3: Use all 100 (all cached now)
prices = build_adj_close_panel(tickers_week2, start="2020-01-01")
# → All from cache (instant) ⚡
```

## Summary

🎯 **Key Point:** The cache is **lazy by default** - you don't need to do anything special!

Just request the data you need, when you need it. The cache handles the rest.

**Start small, scale up as needed.** Your cache grows organically with your project.

