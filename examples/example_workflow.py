"""
Example: Recommended workflow using lazy caching

This shows how to build your ML pipeline without pre-caching.
The cache builds naturally as you request data.
"""
import pandas as pd
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
)

print("=" * 80)
print("Example Workflow: Lazy Caching for ML Pipeline")
print("=" * 80)

# Step 1: Load universe
print("\nStep 1: Load Universe")
print("-" * 80)
universe = pd.read_csv("data/universe/us_universe_ishares.csv")
print(f"Total stocks available: {len(universe)}")

# Step 2: Start with a focused subset
print("\nStep 2: Select Initial Subset (e.g., Tech sector)")
print("-" * 80)
tech_stocks = universe[universe["Sector"] == "Information Technology"]
tickers = tech_stocks["Ticker"].head(30).tolist()
print(f"Selected {len(tickers)} tech stocks for initial development")
print(f"Sample: {tickers[:5]}...")

# Step 3: Build features (cache builds on-demand)
print("\nStep 3: Build Features (cache builds automatically)")
print("-" * 80)
print("Fetching prices for 30 stocks (takes ~30-60 seconds first time)...")

prices = build_adj_close_panel(tickers, start="2023-01-01", end="2024-01-01")
print(f"✓ Price panel: {prices.shape}")

returns = compute_returns(prices, periods=1)
print(f"✓ Returns: {returns.shape}")

momentum = compute_momentum(prices, lookback=126)
print(f"✓ Momentum: {momentum.shape}")

volatility = compute_volatility(returns, window=20)
print(f"✓ Volatility: {volatility.shape}")

momentum_ranks = rank_cross_sectional(momentum)
print(f"✓ Momentum ranks: {momentum_ranks.shape}")

# Step 4: Create feature matrix for ML
print("\nStep 4: Create Feature Matrix for ML")
print("-" * 80)

# Stack to long format (date, ticker, features)
feature_df = pd.DataFrame({
    "returns_1d": returns.stack(),
    "momentum_126d": momentum.stack(),
    "momentum_rank": momentum_ranks.stack(),
    "volatility_20d": volatility.stack(),
})

# Create target: next-day return
feature_df["target_return"] = feature_df.groupby(level=1)["returns_1d"].shift(-1)
feature_df["target_binary"] = (feature_df["target_return"] > 0).astype(int)

# Drop NaN
feature_df = feature_df.dropna()

print(f"Feature matrix shape: {feature_df.shape}")
print(f"\nSample features:")
print(feature_df.head(10))

print(f"\nTarget distribution:")
print(feature_df["target_binary"].value_counts(normalize=True))

# Step 5: Save processed data
print("\nStep 5: Save Processed Data")
print("-" * 80)
output_path = "data/processed/tech_features_sample.parquet"
feature_df.to_parquet(output_path)
print(f"✓ Saved to: {output_path}")

# Step 6: Next time - instant from cache!
print("\nStep 6: Next Run - Data Loads from Cache (Instant)")
print("-" * 80)
print("Run this script again and see how fast it is!")
print("All 30 stocks are now cached.")

print("\n" + "=" * 80)
print("Workflow Complete!")
print("=" * 80)
print("\n✓ Started small (30 stocks)")
print("✓ Cache built naturally")
print("✓ Features created")
print("✓ Ready for model training")
print("\nTo scale up:")
print("  1. Increase head(30) to head(100) or more")
print("  2. Only new stocks will be downloaded")
print("  3. Cached stocks load instantly")


