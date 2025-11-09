# Stock Movement Prediction with Multimodal Data

**CIS 5200 Final Project - Schuylkill River Trading**

This project predicts stock price movements using multimodal data: market prices, financial sentiment from Twitter/X, and FinBERT embeddings.

## Project Structure

```
5200fp/
  data/
    raw/                         # Raw data downloads
    curated/
      prices/
        adj/1d/                  # Adjusted daily prices (parquet)
        raw/1d/                  # Raw prices (parquet)
    universe/
      us_universe.csv            # Full universe with metadata
      us_universe_filtered.csv   # Filtered small-to-mid cap stocks
  
  src/
    data/
      price_cache.py            # Yahoo Finance caching
      panel.py                  # Panel data utilities
    universe/
      us_universe.py            # Universe builder
    cli_fetch_prices.py         # CLI: Fetch single ticker
    cli_build_universe.py       # CLI: Build universe
    cli_bulk_fetch.py           # CLI: Bulk fetch prices
  
  notebooks/                    # Jupyter notebooks
  reports/                      # Model results and reports
  
  requirements.txt
  README.md
  TODO.md
```

## Setup

### 1. Install Dependencies

```bash
cd /Users/rafaelhajjar/Documents/5200fp
pip install -r requirements.txt
```

### 2. Build Stock Universe

Build a universe of small-to-mid cap US stocks (market cap: $300M - $10B):

```bash
# Build universe only
python -m src.cli_build_universe

# Build universe and prefetch prices (recommended)
python -m src.cli_build_universe --prefetch --start 2015-01-01

# Custom filters
python -m src.cli_build_universe \
  --sources russell_2000 sp600 sp400 \
  --min-market-cap 500000000 \
  --max-market-cap 5000000000 \
  --min-price 10.0 \
  --min-volume 2000000 \
  --out data/universe/custom_universe.csv \
  --prefetch
```

**Note:** Building the universe takes time as it fetches metadata for each ticker from Yahoo Finance.

### 3. Fetch Prices

Fetch prices for individual tickers:

```bash
# Single ticker
python -m src.cli_fetch_prices AAPL --start 2015-01-01

# With custom options
python -m src.cli_fetch_prices MSFT --start 2020-01-01 --end 2023-12-31 --interval 1d
```

Bulk fetch for entire universe:

```bash
python -m src.cli_bulk_fetch \
  --universe data/universe/us_universe_filtered.csv \
  --start 2015-01-01
```

## Usage in Python

### Fetch Price Data

```python
from pathlib import Path
from src.data import get_prices, get_prices_bulk

# Single ticker
df = get_prices("AAPL", start="2020-01-01", end="2023-12-31")
print(df.head())

# Multiple tickers
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
prices = get_prices_bulk(tickers, start="2020-01-01")
```

### Build Panel Data

```python
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
)
import pandas as pd

# Load universe
universe = pd.read_csv("data/universe/us_universe_filtered.csv")
tickers = universe["ticker"].tolist()[:50]  # First 50 stocks

# Build price panel
prices = build_adj_close_panel(tickers, start="2020-01-01")

# Compute features
returns = compute_returns(prices, periods=1)
momentum_6m = compute_momentum(prices, lookback=126)
volatility = compute_volatility(returns, window=20)

# Cross-sectional ranks
momentum_ranks = rank_cross_sectional(momentum_6m)
```

### Build Universe Programmatically

```python
from pathlib import Path
from src.universe.us_universe import UniverseBuildConfig, build_us_universe

config = UniverseBuildConfig(
    include_sources=["russell_2000", "sp600"],
    min_market_cap=300e6,
    max_market_cap=10e9,
    min_price=5.0,
    min_avg_volume=1e6,
    out_csv=Path("data/universe/my_universe.csv"),
)

df = build_us_universe(config, verbose=True)
print(f"Universe size: {df['pass_all'].sum()} stocks")
```

## Data Format

### Price Data (Parquet)

Cached in `data/curated/prices/adj/1d/{TICKER}.parquet`:

```
Columns: adj_close, close, high, low, open, volume
Index: DateTimeIndex (UTC)
```

### Universe CSV

Columns:
- `ticker`: Ticker symbol
- `name`: Company name
- `market_cap`: Market capitalization
- `price`: Current price
- `avg_volume`: Average volume
- `dollar_volume`: Average dollar volume
- `exchange`: Exchange (NYSE, NASDAQ, etc.)
- `sector`, `industry`: GICS classification
- `pass_*`: Filter results (True/False)
- `pass_all`: Passed all filters

## Universe Selection Criteria

**Small-to-Mid Cap Focus:**
- Market cap: $300M - $10B
- Price: ≥ $5 (avoid penny stocks)
- Avg daily dollar volume: ≥ $1M
- Exchanges: NYSE, NASDAQ, AMEX
- Type: Common equity only (no REITs, ADRs, preferreds)

**Sources:**
- Russell 2000 (small cap)
- S&P 600 SmallCap
- S&P 400 MidCap

## Next Steps for ML Pipeline

1. **Feature Engineering** (`notebooks/01_feature_engineering.ipynb`)
   - Technical indicators: momentum, volatility, RSI, MACD
   - Cross-sectional features: ranks, z-scores
   - Lagged features (avoid look-ahead bias)

2. **Sentiment Data** (`src/data/twitter_scraper.py`, `src/data/finbert.py`)
   - Twitter/X API for financial keywords
   - FinBERT sentiment scoring
   - Align sentiment with dates (point-in-time)

3. **Target Variable**
   - Binary: next-day up/down
   - Ranking: predict relative performance

4. **Model Training** (`src/models/`)
   - Logistic Regression baseline
   - Random Forest
   - XGBoost
   - Neural Network (multimodal)

5. **Evaluation** (`src/evaluation/`)
   - Walk-forward validation
   - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   - Ranking metrics: Spearman, Kendall-τ, NDCG
   - Backtesting: Sharpe ratio, max drawdown

## Notes

- **Caching**: Price data is cached in Parquet format for fast access. Incremental updates fetch only new data.
- **No Look-Ahead Bias**: All features must be point-in-time. Use lagged values.
- **Rate Limits**: Yahoo Finance has rate limits. The code includes delays and error handling.
- **Data Quality**: Some tickers may have missing data or errors. Always validate.

## Team

**Schuylkill River Trading**
- Rafael Hajjar (Data Pipeline, Neural Network, Evaluation)
- Monica (Baselines: Logistic Regression, Random Forest)
- Kylie (XGBoost, Ranking Model, Interpretability)

## License

Academic project for CIS 5200 at University of Pennsylvania.

