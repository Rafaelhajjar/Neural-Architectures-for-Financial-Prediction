# 🚀 START HERE - Quick Guide

## ✅ What's Ready

You have a **production-ready Yahoo Finance data infrastructure** with:
- ✅ **2,343 stocks** automatically downloaded (no manual picking!)
- ✅ **Lazy caching** - data fetches only when requested
- ✅ **Parquet storage** - fast, efficient
- ✅ **Feature engineering utilities** ready to use

## 🎯 Key Concept: Lazy Caching

**You DON'T need to pre-cache anything!** The cache builds automatically when you request data.

```python
from src.data import get_prices

# First time: Downloads & caches (~1 second)
df = get_prices("AAPL", start="2020-01-01")

# Second time: Instant from cache (~0.01 seconds)
df = get_prices("AAPL", start="2020-01-01")  # ⚡ Super fast!
```

## 🏃 Quick Start (3 Steps)

### Step 1: See Lazy Caching in Action

```bash
python demo_lazy_caching.py
```

**Shows:**
- How cache builds on first request
- How second request is instant
- Cache statistics

### Step 2: Run Complete Workflow Example

```bash
python example_workflow.py
```

**Does:**
1. Loads 30 tech stocks
2. Builds features (returns, momentum, volatility)
3. Creates ML feature matrix
4. Saves processed data

**First run:** ~30-60 seconds (downloads data)  
**Second run:** ~1 second (from cache) ⚡

### Step 3: Scale Up for Your Project

```python
import pandas as pd
from src.data.panel import build_adj_close_panel, compute_returns

# Load your 2,343 stocks
universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# Start with 100 stocks
tickers = universe["Ticker"].head(100).tolist()

# Build features (cache builds naturally)
prices = build_adj_close_panel(tickers, start="2020-01-01")
returns = compute_returns(prices, periods=1)

# Scale to 500 stocks (only 400 new ones downloaded)
tickers = universe["Ticker"].head(500).tolist()
prices = build_adj_close_panel(tickers, start="2020-01-01")
```

## 📁 Your Data

### Universe Files
- **`data/universe/us_universe_ishares.csv`** - 2,343 stocks (automatic!)
  - Includes: Ticker, Name, Sector, Market Value, Exchange
  - Source: iShares ETFs (IWM, IJR, IJH)

### Cached Prices
- **`data/curated/prices/adj/1d/*.parquet`** - Price cache
  - Builds as you request data
  - Check size: `ls data/curated/prices/adj/1d/ | wc -l`

## 📚 Documentation

Choose your reading style:

1. **Quick commands**: `QUICKSTART.md`
2. **Lazy caching explained**: `LAZY_CACHING_GUIDE.md`
3. **Automatic universe**: `AUTOMATIC_UNIVERSE.md`
4. **Complete docs**: `README.md`
5. **Setup summary**: `SETUP_COMPLETE.md`

## 🎓 For Your ML Project (CIS 5200)

### Recommended Approach

**Phase 1: Start Small (Week 1)**
```python
# Work with 50-100 stocks for development
tickers = universe["Ticker"].head(100).tolist()
prices = build_adj_close_panel(tickers, start="2020-01-01")

# Build your pipeline
# - Feature engineering
# - Model training
# - Evaluation
```

**Phase 2: Scale Up (Week 2-3)**
```python
# Once pipeline works, scale to 500-1000 stocks
tickers = universe["Ticker"].head(500).tolist()
prices = build_adj_close_panel(tickers, start="2020-01-01")
# Only new stocks download, cached ones instant!
```

**Phase 3: Full Universe (Week 4)**
```python
# Use all 2,343 stocks for final results
tickers = universe["Ticker"].tolist()
prices = build_adj_close_panel(tickers, start="2020-01-01")
```

### Why This Works Better

✅ **Fast iteration** - Test with small sets first  
✅ **Incremental caching** - Cache grows with usage  
✅ **Flexible** - Easy to experiment  
✅ **Efficient** - Only download what you need

## 🛠️ Common Tasks

### Fetch a Single Stock
```bash
python -m src.cli_fetch_prices TSLA --start 2020-01-01
```

### Check Cache Status
```bash
# Count cached stocks
ls data/curated/prices/adj/1d/ | wc -l

# Check cache size
du -sh data/curated/prices/adj/1d
```

### Filter Universe by Sector
```python
import pandas as pd

universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# Get tech stocks
tech = universe[universe["Sector"] == "Information Technology"]
print(f"Tech stocks: {len(tech)}")

# Get healthcare
healthcare = universe[universe["Sector"] == "Health Care"]
print(f"Healthcare stocks: {len(healthcare)}")
```

### Force Refresh Data
```python
from src.data import get_prices

# Ignore cache, re-download
df = get_prices("AAPL", start="2020-01-01", force_refresh=True)
```

## ⚠️ What NOT to Do

**❌ Don't pre-cache everything:**
```bash
# This takes 30-60 minutes and you may not need all stocks
python -m src.cli_bulk_fetch \
  --universe data/universe/us_universe_ishares.csv \
  --start 2015-01-01
```

**✅ Instead, let cache build naturally:**
```python
# Just request what you need, when you need it
prices = build_adj_close_panel(tickers, start="2020-01-01")
# Cache builds automatically!
```

## 🎯 Your Next Steps

1. ✅ **Run `demo_lazy_caching.py`** - See how it works
2. ✅ **Run `example_workflow.py`** - Complete example
3. ✅ **Read `LAZY_CACHING_GUIDE.md`** - Understand the system
4. 🚀 **Start building your ML pipeline!**

## 💡 Pro Tips

- **Start small**: Use 50-100 stocks for initial development
- **Scale gradually**: Add more stocks as your pipeline matures
- **Monitor cache**: Check `data/curated/prices/adj/1d/` to see what's cached
- **Sector focus**: Filter by sector to reduce noise
- **Run twice**: First run downloads, second run shows cache speed

## 🆘 Need Help?

**Cache not working?**
```bash
# Check if cache directory exists
ls -la data/curated/prices/adj/1d/

# Try fetching a single stock
python -m src.cli_fetch_prices AAPL --start 2023-01-01
```

**Want to start fresh?**
```bash
# Clear cache
rm -rf data/curated/prices/adj/1d/*

# Re-run your code - cache rebuilds automatically
```

## 🎉 You're Ready!

**Your infrastructure is complete.** The cache is lazy - it builds as you work.

**Just start requesting data and the system handles the rest!**

Happy coding! 🚀


