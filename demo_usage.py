"""Demo script showing how to use the data infrastructure."""
import pandas as pd
from src.data import get_prices, get_prices_bulk
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
    zscore_cross_sectional,
)

print("=" * 80)
print("Demo: Using Yahoo Finance Data Infrastructure")
print("=" * 80)

# Load universe
print("\n1. Loading universe...")
universe = pd.read_csv("data/universe/us_universe_sample_filtered.csv")
tickers = universe["ticker"].tolist()
print(f"   Universe size: {len(tickers)} stocks")
print(f"   Tickers: {', '.join(tickers[:10])}...")

# Build price panel
print("\n2. Building price panel...")
prices = build_adj_close_panel(tickers, start="2023-01-01", end="2024-01-01")
print(f"   Shape: {prices.shape} (dates x stocks)")
print(f"   Date range: {prices.index.min()} to {prices.index.max()}")

# Compute features
print("\n3. Computing features...")
returns_1d = compute_returns(prices, periods=1)
returns_5d = compute_returns(prices, periods=5)
momentum_126d = compute_momentum(prices, lookback=126)
volatility_20d = compute_volatility(returns_1d, window=20, annualize=True)

print(f"   Returns (1d): {returns_1d.shape}")
print(f"   Returns (5d): {returns_5d.shape}")
print(f"   Momentum (6m): {momentum_126d.shape}")
print(f"   Volatility (20d): {volatility_20d.shape}")

# Cross-sectional features
print("\n4. Computing cross-sectional features...")
momentum_ranks = rank_cross_sectional(momentum_126d, ascending=False)
momentum_zscores = zscore_cross_sectional(momentum_126d)

print(f"   Momentum ranks: {momentum_ranks.shape}")
print(f"   Momentum z-scores: {momentum_zscores.shape}")

# Show sample data
print("\n5. Sample data (last date):")
last_date = prices.index[-1]
sample_df = pd.DataFrame({
    "Price": prices.loc[last_date],
    "Return_1d": returns_1d.loc[last_date],
    "Momentum_6m": momentum_126d.loc[last_date],
    "Mom_Rank": momentum_ranks.loc[last_date],
    "Mom_Zscore": momentum_zscores.loc[last_date],
    "Vol_20d": volatility_20d.loc[last_date],
})
sample_df = sample_df.dropna().sort_values("Mom_Rank", ascending=False)
print(sample_df.head(10).to_string())

# Show summary statistics
print("\n6. Summary statistics:")
print("\nAverage returns (1d):")
print(f"   Mean: {returns_1d.mean().mean()*100:.3f}%")
print(f"   Std: {returns_1d.std().mean()*100:.3f}%")

print("\nAverage volatility (20d annualized):")
print(f"   Mean: {volatility_20d.mean().mean()*100:.1f}%")
print(f"   Median: {volatility_20d.median().mean()*100:.1f}%")

print("\n" + "=" * 80)
print("Demo complete!")
print("=" * 80)
print("\nNext steps:")
print("1. Add sentiment data (Twitter, FinBERT)")
print("2. Create target variable (next-day up/down)")
print("3. Merge features and align by date")
print("4. Train ML models (Logistic Regression, Random Forest, XGBoost, Neural Network)")
print("5. Evaluate with walk-forward validation")

