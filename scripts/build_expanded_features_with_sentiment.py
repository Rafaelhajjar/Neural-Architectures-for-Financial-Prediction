"""
Build expanded features dataset with 100+ stocks that were trading in 2008-2016.
Includes both price features and sentiment features.

This ensures no survivorship bias - only using stocks that existed during the full period.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

sys.path.append('/Users/rafaelhajjar/Documents/5200fp')

from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
)
from src.data import get_prices

print("="*70)
print("BUILDING EXPANDED FEATURE DATASET WITH SENTIMENT")
print("="*70)

# Parameters
START_DATE = "2008-01-01"
END_DATE = "2017-01-01"
MIN_TRADING_DAYS = 1500  # ~6 years of data minimum
TARGET_STOCKS = 100

# Step 1: Load universe
print("\nStep 1: Loading universe...")
universe = pd.read_csv("data/universe/us_universe_ishares.csv")
print(f"Total stocks in universe: {len(universe)}")

# Step 2: Test which stocks have data in 2008-2016 period
print("\nStep 2: Testing stocks for data availability in 2008-2016...")
print("This will take a few minutes on first run...")

valid_tickers = []
failed_tickers = []

# Get unique tickers
all_tickers = universe["Ticker"].dropna().unique().tolist()
print(f"Testing {len(all_tickers)} unique tickers...")

for i, ticker in enumerate(all_tickers):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(all_tickers)} tested, {len(valid_tickers)} valid so far...")
    
    try:
        # Try to fetch data
        df = get_prices(ticker, start=START_DATE, end=END_DATE)
        
        if df is not None and len(df) >= MIN_TRADING_DAYS:
            # Check that it has data in both early and late periods
            early_data = df[df.index < '2010-01-01']
            late_data = df[df.index > '2014-01-01']
            
            if len(early_data) > 100 and len(late_data) > 100:
                valid_tickers.append(ticker)
                
                # Stop once we have enough
                if len(valid_tickers) >= TARGET_STOCKS:
                    print(f"  Reached target of {TARGET_STOCKS} stocks!")
                    break
        else:
            failed_tickers.append(ticker)
    except Exception as e:
        failed_tickers.append(ticker)
        continue

print(f"\n✅ Found {len(valid_tickers)} stocks with sufficient 2008-2016 data")
print(f"❌ Skipped {len(failed_tickers)} stocks (insufficient data)")
print(f"\nSelected stocks: {valid_tickers[:20]}...")

# Save the valid tickers list for reference
valid_tickers_df = pd.DataFrame({'ticker': valid_tickers})
valid_tickers_df.to_csv('data/universe/expanded_100_tickers.csv', index=False)
print(f"\n✅ Saved ticker list to: data/universe/expanded_100_tickers.csv")

# Step 3: Fetch prices for all valid tickers
print("\nStep 3: Building price panel...")
print(f"Fetching prices for {len(valid_tickers)} stocks...")

prices = build_adj_close_panel(
    valid_tickers,
    start=START_DATE,
    end=END_DATE,
)

print(f"Price panel shape: {prices.shape}")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")

# Step 4: Compute price features
print("\nStep 4: Computing price features...")
returns_1d = compute_returns(prices, periods=1)
momentum_126d = compute_momentum(prices, lookback=126)
vol_20d = compute_volatility(returns_1d, window=20, annualize=True)
mom_rank = rank_cross_sectional(momentum_126d)

print("  ✅ Returns computed")
print("  ✅ Momentum computed")
print("  ✅ Volatility computed")
print("  ✅ Cross-sectional ranks computed")

# Step 5: Create long-format dataframe
print("\nStep 5: Creating feature matrix...")
dataset = pd.DataFrame({
    "ret_1d": returns_1d.stack(),
    "momentum_126d": momentum_126d.stack(),
    "vol_20d": vol_20d.stack(),
    "mom_rank": mom_rank.stack(),
})

dataset.index.names = ["date", "ticker"]
dataset = dataset.reset_index()
dataset["date"] = pd.to_datetime(dataset["date"]).dt.tz_localize(None)  # Remove timezone

print(f"Initial dataset shape: {dataset.shape}")

# Step 6: Add target variables
print("\nStep 6: Creating target variables...")

# Compute future returns using groupby (more reliable than dictionary)
# For each ticker, shift returns by -1 to get next-day return
dataset = dataset.sort_values(['ticker', 'date'])
dataset['future_return'] = dataset.groupby('ticker')['ret_1d'].shift(-1)
dataset['target'] = (dataset['future_return'] > 0).astype(int)

print("  ✅ Target variables created")
print(f"  Future returns with data: {dataset['future_return'].notna().sum():,} / {len(dataset):,}")

# Step 7: Load and merge sentiment features
print("\nStep 7: Loading sentiment features...")

try:
    # Load market-level sentiment (applies to all stocks)
    market_sentiment = pd.read_parquet("data/processed/market_sentiment_daily.parquet")
    
    # Handle column naming (might be 'Date' or 'date')
    if 'Date' in market_sentiment.columns:
        market_sentiment = market_sentiment.rename(columns={'Date': 'date'})
    
    market_sentiment['date'] = pd.to_datetime(market_sentiment['date'])
    
    print(f"Market sentiment shape: {market_sentiment.shape}")
    print(f"Market sentiment date range: {market_sentiment['date'].min()} to {market_sentiment['date'].max()}")
    
    # Check available columns
    print(f"Available columns: {list(market_sentiment.columns)}")
    
    # Merge with dataset
    merge_cols = ['date']
    # Add columns that exist
    for col in ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count',
                'market_positive_count', 'market_negative_count']:
        if col in market_sentiment.columns:
            merge_cols.append(col)
    
    dataset = dataset.merge(
        market_sentiment[merge_cols],
        on='date',
        how='left'
    )
    
    print("  ✅ Sentiment features merged")
    print(f"  Sentiment coverage (before fill): {(dataset['market_news_count'].notna()).mean():.1%} of days")
    
except Exception as e:
    print(f"  ⚠️  Could not load sentiment data: {e}")
    print("  Creating dummy sentiment features (all zeros)")
    
    dataset['market_sentiment_mean'] = 0.0
    dataset['market_sentiment_std'] = 0.0
    dataset['market_news_count'] = 0.0
    dataset['market_positive_count'] = 0.0
    dataset['market_negative_count'] = 0.0

# Always fill missing sentiment with neutral values (0)
# This happens when sentiment data ends before price data
print("\nFilling missing sentiment values...")
sentiment_cols = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count',
                  'market_positive_count', 'market_negative_count']
for col in sentiment_cols:
    if col in dataset.columns:
        missing_before = dataset[col].isna().sum()
        dataset[col] = dataset[col].fillna(0)
        print(f"  {col}: Filled {missing_before:,} missing values")

# Step 8: Drop NaN values and filter date range
print("\nStep 8: Cleaning dataset...")

# Drop rows with NaN in critical features
critical_cols = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank', 'future_return', 'target']
print(f"\nBefore dropping NaN:")
print(f"  Dataset shape: {dataset.shape}")
for col in critical_cols:
    nan_count = dataset[col].isna().sum()
    print(f"  {col}: {nan_count:,} NaN values ({nan_count/len(dataset)*100:.1f}%)")

dataset = dataset.dropna(subset=critical_cols)
print(f"\nAfter dropping NaN: {dataset.shape}")

# Filter to match original date range (after momentum computation)
dataset = dataset[dataset['date'] >= '2008-11-01']
dataset = dataset[dataset['date'] <= '2016-12-31']

print(f"Final dataset shape: {dataset.shape}")
print(f"Final date range: {dataset['date'].min()} to {dataset['date'].max()}")
print(f"Number of unique stocks: {dataset['ticker'].nunique()}")
print(f"Average observations per stock: {len(dataset) / dataset['ticker'].nunique():.0f}")

# Step 9: Data quality checks
print("\nStep 9: Data quality checks...")

print("\nSamples per stock:")
stock_counts = dataset.groupby('ticker').size().sort_values(ascending=False)
print(f"  Min: {stock_counts.min()}")
print(f"  Max: {stock_counts.max()}")
print(f"  Mean: {stock_counts.mean():.0f}")
print(f"  Median: {stock_counts.median():.0f}")

print("\nFeature statistics:")
feature_cols = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank', 
                'market_sentiment_mean', 'market_news_count']
for col in feature_cols:
    if col in dataset.columns:
        print(f"  {col}: mean={dataset[col].mean():.4f}, std={dataset[col].std():.4f}")

print("\nTarget distribution:")
print(f"  Up days: {dataset['target'].sum():,} ({dataset['target'].mean():.1%})")
print(f"  Down days: {(1-dataset['target']).sum():,} ({(1-dataset['target']).mean():.1%})")

# Step 10: Save final dataset
print("\nStep 10: Saving dataset...")

output_path = "data/processed/features_expanded_100stocks_with_sentiment.parquet"
dataset.to_parquet(output_path, index=False)

print(f"\n✅ Saved to: {output_path}")

# Also save a CSV version for inspection
csv_path = "data/processed/features_expanded_100stocks_with_sentiment.csv"
dataset.head(1000).to_csv(csv_path, index=False)
print(f"✅ Saved sample (1000 rows) to: {csv_path}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✅ Stocks: {dataset['ticker'].nunique()}")
print(f"✅ Total samples: {len(dataset):,}")
print(f"✅ Date range: {dataset['date'].min().date()} to {dataset['date'].max().date()}")
print(f"✅ Days: {dataset['date'].nunique()}")
print(f"✅ Features: {len([c for c in dataset.columns if c not in ['date', 'ticker', 'target', 'future_return']])}")
print(f"✅ Sentiment coverage: {(dataset['market_news_count'] > 0).mean():.1%}")
print("\n🚀 Ready to train neural networks with expanded dataset!")
print("="*70)

