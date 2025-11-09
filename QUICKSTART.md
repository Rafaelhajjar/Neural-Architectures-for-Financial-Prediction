# Quick Start Guide

## What's Been Set Up

Your project now has a complete Yahoo Finance data infrastructure with:

✅ **Parquet-based caching** - Fast, incremental data storage  
✅ **Universe builder** - Small-to-mid cap US stock filtering  
✅ **CLI tools** - Easy data fetching  
✅ **Panel data utilities** - Feature engineering helpers  
✅ **Sample universe** - 20 stocks ready to use  

## Your Data Right Now

```
data/
  universe/
    us_universe_sample.csv           # 49 tickers with metadata
    us_universe_sample_filtered.csv  # 20 tickers passing all filters
  curated/prices/adj/1d/
    *.parquet                         # 21 stocks (including AAPL test)
```

**Date range:** 2023-01-01 to 2024-01-01 (250 trading days)

## Quick Commands

### 1. Fetch a Single Stock

```bash
python -m src.cli_fetch_prices TSLA --start 2020-01-01
```

### 2. See Your Sample Data

```bash
python demo_usage.py
```

This shows:
- Price panel construction (250 days × 20 stocks)
- Returns, momentum, volatility features
- Cross-sectional ranks and z-scores
- Summary statistics

### 3. Build a Larger Universe (Optional)

⚠️ **Takes 30-60 minutes** - fetches metadata for ~2000+ tickers

```bash
python build_full_universe.py
```

Or build without confirmation:

```bash
python -c "
from pathlib import Path
from src.universe.us_universe import UniverseBuildConfig, build_us_universe

config = UniverseBuildConfig(
    include_sources=['russell_2000', 'sp600', 'sp400'],
    min_market_cap=300e6,
    max_market_cap=10e9,
    out_csv=Path('data/universe/us_universe_full.csv')
)
build_us_universe(config)
"
```

Then fetch prices:

```bash
python -m src.cli_bulk_fetch \
  --universe data/universe/us_universe_full_filtered.csv \
  --start 2015-01-01
```

## Python Usage

### Load Data

```python
from src.data import get_prices
import pandas as pd

# Load a single stock
df = get_prices("AAPL", start="2020-01-01")
print(df.head())
```

### Build Features

```python
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    rank_cross_sectional,
)

# Load universe
universe = pd.read_csv("data/universe/us_universe_sample_filtered.csv")
tickers = universe["ticker"].tolist()

# Build price matrix
prices = build_adj_close_panel(tickers, start="2023-01-01", end="2024-01-01")

# Compute features
returns = compute_returns(prices, periods=1)
momentum_6m = compute_momentum(prices, lookback=126)
momentum_ranks = rank_cross_sectional(momentum_6m)

# Create feature matrix for ML
features = pd.DataFrame({
    "returns_1d": returns.stack(),
    "momentum_6m": momentum_6m.stack(),
    "momentum_rank": momentum_ranks.stack(),
})
features = features.dropna()
print(features.head())
```

## Next Steps for Your ML Project

### Phase 1: Expand Universe (Optional)
- Run `build_full_universe.py` to get 500-2000 stocks
- Or stick with 20 stocks for faster iteration

### Phase 2: Target Variable
Create binary labels (up/down):

```python
# Create target: 1 if next-day return > 0
target = (returns.shift(-1) > 0).astype(int)
```

### Phase 3: Add Sentiment Data
- Twitter/X scraper (`src/data/twitter_scraper.py`)
- FinBERT sentiment (`src/data/finbert.py`)
- Align by date (no look-ahead!)

### Phase 4: Train Models
- Logistic Regression baseline
- Random Forest
- XGBoost
- Neural Network (multimodal)

### Phase 5: Walk-Forward Validation
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(features):
    # Train on train_idx
    # Test on test_idx
    # Compute metrics
```

## Data Quality Notes

- **Caching:** Price data is automatically cached. Re-running fetches only updates with new dates.
- **Rate Limits:** Yahoo Finance has rate limits. The code includes 0.1s delays.
- **Missing Data:** Some tickers may fail or have gaps. Always check data quality.
- **Survivorship Bias:** Current constituents only. For better research, track historical additions/deletions.

## Files Reference

```
src/
  data/
    price_cache.py       # Core: fetch & cache from yfinance
    panel.py             # Build matrices, compute features
  universe/
    us_universe.py       # Universe builder with filters
  cli_fetch_prices.py    # CLI: fetch single ticker
  cli_build_universe.py  # CLI: build universe + optional prefetch
  cli_bulk_fetch.py      # CLI: bulk fetch from CSV

demo_usage.py            # Example: load data, compute features
build_full_universe.py   # Script: build large universe (30-60 min)
```

## Troubleshooting

**"No module named 'src'"**
```bash
cd /Users/rafaelhajjar/Documents/5200fp
python -m src.cli_fetch_prices AAPL
```

**"Missing lxml"**
```bash
pip install lxml html5lib beautifulsoup4
```

**"Rate limit exceeded"**
- Wait a few minutes and retry
- Use cached data if available
- Increase delays in code

## Questions?

Check `README.md` for detailed documentation or run:

```bash
python -m src.cli_fetch_prices --help
python -m src.cli_build_universe --help
python -m src.cli_bulk_fetch --help
```

