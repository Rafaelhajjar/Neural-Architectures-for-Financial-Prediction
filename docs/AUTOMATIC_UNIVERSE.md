# Automatic Universe Generation ✅

## Problem Solved!

Your universe is now **automatically generated** from real index data - **no manual stock picking**!

## What We Have Now

### ✅ Automatic Download from iShares

**File:** `data/universe/us_universe_ishares.csv`

**Contains:** **2,343 unique stocks** automatically downloaded from:
- **IWM**: Russell 2000 (~1,965 small cap stocks)
- **IJR**: S&P 600 SmallCap (~637 stocks)
- **IJH**: S&P 400 MidCap (~415 stocks)

These are the **actual current holdings** of these ETFs, updated daily by iShares (BlackRock).

### Data Included

Every stock has:
- **Ticker**: Stock symbol
- **Name**: Company name
- **Sector**: GICS sector classification
- **Market Value**: Current holding value
- **Weight**: ETF weight percentage
- **Exchange**: Trading venue
- **source_etf**: Which ETF it came from (IWM, IJR, or IJH)

## Quick Commands

### 1. Use Existing Universe (2,343 stocks)

The simplest approach - use the downloaded universe as-is:

```bash
# Just fetch prices for all 2,343 stocks (takes ~30 min)
python -m src.cli_bulk_fetch \
  --universe data/universe/us_universe_ishares.csv \
  --ticker-column Ticker \
  --start 2015-01-01
```

### 2. Build Filtered Universe (Small-to-Mid Cap Only)

Filter by market cap, price, and volume:

```bash
python -m src.cli_build_universe_auto \
  --etfs IWM IJR IJH \
  --min-market-cap 300000000 \
  --max-market-cap 10000000000 \
  --min-price 5.0 \
  --min-volume 1000000 \
  --out data/universe/us_universe_filtered_auto.csv
```

This will:
1. Download all holdings from IWM, IJR, IJH
2. Filter for market cap $300M - $10B
3. Require price ≥ $5
4. Require avg daily volume ≥ $1M
5. Save to CSV

**Expected result:** ~500-800 stocks that meet criteria

### 3. Build and Prefetch Prices

Do everything in one command:

```bash
python -m src.cli_build_universe_auto \
  --etfs IWM IJR IJH \
  --min-market-cap 300000000 \
  --max-market-cap 10000000000 \
  --prefetch \
  --start 2020-01-01
```

⚠️ This takes 30-60 minutes total (filtering + price fetch)

## Python Usage

### Load Existing Universe

```python
import pandas as pd

# Load the 2,343 stocks
universe = pd.read_csv("data/universe/us_universe_ishares.csv")

print(f"Total stocks: {len(universe)}")
print(f"Sectors: {universe['Sector'].value_counts()}")

# Get just the tickers
tickers = universe['Ticker'].tolist()
```

### Filter Programmatically

```python
import pandas as pd
from src.data import get_prices_bulk

# Load universe
universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# Apply your own filters
# Example: Only Technology sector
tech_stocks = universe[universe['Sector'] == 'Information Technology']
tickers = tech_stocks['Ticker'].tolist()

print(f"Tech stocks: {len(tickers)}")

# Fetch prices
prices = get_prices_bulk(tickers, start="2023-01-01")
```

### Build Custom Universe

```python
from pathlib import Path
from src.universe.universe_sources import build_universe_from_ishares

# Download from specific ETFs
etfs = ["IWM", "IJR"]  # Russell 2000 + S&P 600 only
universe = build_universe_from_ishares(
    etfs=etfs,
    output_path=Path("data/universe/my_custom_universe.csv"),
    verbose=True
)

print(f"Downloaded {len(universe)} stocks")
```

## Available ETFs

You can download from any of these iShares ETFs:

| ETF Code | Index | # Stocks | Market Cap | Description |
|----------|-------|----------|------------|-------------|
| **IWM** | Russell 2000 | ~2,000 | Small | Most comprehensive small cap |
| **IJR** | S&P 600 | ~600 | Small | Higher quality small cap |
| **IJH** | S&P 400 | ~400 | Mid | Mid cap stocks |
| **IWR** | Russell MidCap | ~800 | Mid | Russell mid cap |
| **IWB** | Russell 1000 | ~1,000 | Large | Large cap stocks |
| **IWV** | Russell 3000 | ~3,000 | All | Entire US market |

### Example: Get All US Stocks

```bash
python -m src.cli_build_universe_auto \
  --etfs IWV \
  --skip-filter \
  --out data/universe/us_all_stocks.csv
```

This downloads ~3,000 stocks covering the entire US market.

## Advantages Over Wikipedia Scraping

✅ **No 403 errors** - iShares allows programmatic access  
✅ **Daily updates** - Holdings updated every trading day  
✅ **Complete data** - Includes sector, market value, weights  
✅ **Reliable** - From BlackRock's official data feed  
✅ **No manual selection** - 100% automated  
✅ **Deduplication** - Automatically removes duplicates across ETFs  

## Sector Breakdown (Current Universe)

From our 2,343 stocks:

```
Financials:                 476 (20%)
Health Care:                431 (18%)
Industrials:                331 (14%)
Consumer Discretionary:     260 (11%)
Information Technology:     257 (11%)
Real Estate:                132 (6%)
Energy:                     121 (5%)
Materials:                  106 (5%)
Communication:               91 (4%)
Consumer Staples:            81 (3%)
```

## Next Steps

### For Your ML Project

1. **Use the existing 2,343 stocks**, or
2. **Filter to small-to-mid cap** (300M-10B market cap) → ~500-800 stocks
3. **Fetch prices** for your date range (2015-2024)
4. **Build features** using `src/data/panel.py`
5. **Train models**!

### Recommended Approach

For faster iteration during development:

```python
# Start with a subset for testing
import pandas as pd
universe = pd.read_csv("data/universe/us_universe_ishares.csv")

# Get 100 most liquid stocks
universe_sorted = universe.sort_values('Market Value', ascending=False)
test_tickers = universe_sorted['Ticker'].head(100).tolist()

# Build your ML pipeline with these 100 stocks
# Once it works, scale to all 2,343
```

## Files Created

```
data/universe/
  us_universe_ishares.csv           # 2,343 stocks from IWM+IJR+IJH
  raw_holdings.csv                  # Raw iShares data with weights
  us_universe_sample.csv            # Old test file (49 stocks)
  us_universe_sample_filtered.csv   # Old test file (20 stocks)
```

## Maintenance

The iShares data updates daily. To refresh:

```bash
# Re-download latest holdings
python -m src.cli_build_universe_auto \
  --etfs IWM IJR IJH \
  --skip-filter \
  --out data/universe/us_universe_ishares_latest.csv
```

Run this monthly or quarterly to keep your universe current.

---

**You now have a fully automated, production-ready universe generation system!** 🎉

