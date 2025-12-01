"""
This script loads the universe, fetches price data,
builds features, and trains two baseline models:
- Logistic Regression
- Random Forest

It uses the same data pipeline and feature set as xg_pipeline.py
so the baselines are directly comparable to the XGBoost models.
"""

from pathlib import Path

import pandas as pd

# project imports
from src.data.panel import (
    build_adj_close_panel,
    compute_returns,
    compute_momentum,
    compute_volatility,
    rank_cross_sectional,
)
from src.models.logistic_regression import train_logistic_regression
from src.models.random_forest import train_random_forest

# --------------------------------------------------------------------
# 1. Load the universe of stocks
# --------------------------------------------------------------------

universe_path = Path("data/universe/us_universe_sample.csv")
universe = pd.read_csv(universe_path)
tickers = universe["ticker"].dropna().unique().tolist()

print(f"Loaded {len(tickers)} tickers from {universe_path}")
print("First few tickers:", tickers[:10])

# --------------------------------------------------------------------
# 2. Load prices (cached after first run)
# --------------------------------------------------------------------

prices = build_adj_close_panel(
    tickers,
    start="2014-01-01",
    end="2020-12-31",
)

print("Price panel shape:", prices.shape)

if prices.empty:
    raise SystemExit(
        "ERROR: prices DataFrame is empty.\n"
        "This means no price history was loaded for your tickers in the "
        "date range 2014-01-01 to 2020-12-31.\n"
        "Check your cached price files or adjust the date range."
    )

print("First few rows of prices:\n", prices.head())
print("First few columns (tickers):", list(prices.columns)[:10])

# --------------------------------------------------------------------
# 3. Build features (same as xg_pipeline)
# --------------------------------------------------------------------

returns_1d = compute_returns(prices, periods=1)
momentum_126d = compute_momentum(prices, lookback=126)
vol_20d = compute_volatility(returns_1d, window=20, annualize=True)
mom_rank = rank_cross_sectional(momentum_126d)

# combine into one long DataFrame (date × ticker)
dataset = pd.DataFrame(
    {
        "ret_1d": returns_1d.stack(),
        "momentum_126d": momentum_126d.stack(),
        "vol_20d": vol_20d.stack(),
        "mom_rank": mom_rank.stack(),
    }
)

dataset.index.names = ["date", "ticker"]
dataset = dataset.dropna()

# --------------------------------------------------------------------
# 4. Build targets (classification + future return)
# --------------------------------------------------------------------

y_class = (returns_1d.shift(-1).stack() > 0).astype(int)  # up/down
y_class.name = "target"

y_reg = returns_1d.shift(-1).stack()  # next-day return
y_reg.name = "future_return"

# join everything together
df = dataset.join([y_class, y_reg]).reset_index()
df["date"] = pd.to_datetime(df["date"])
df["date"] = df["date"].dt.tz_localize(None)

feature_cols = ["ret_1d", "momentum_126d", "vol_20d", "mom_rank"]

# drop any rows where features or targets are NaN
df = df.dropna(subset=feature_cols + ["target", "future_return"])

print("Final dataset shape after dropping NaNs:", df.shape)

# --------------------------------------------------------------------
# 5. Train baselines: Logistic Regression & Random Forest
# --------------------------------------------------------------------

SPLIT_DATE = "2018-01-01"  # same walk-forward split as xg_pipeline

print(f"\nUsing split_date = {SPLIT_DATE}")

print("\nTraining Logistic Regression baseline...")
lr_model, lr_metrics, lr_test = train_logistic_regression(
    df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date=SPLIT_DATE,
)
print("Logistic Regression metrics:")
for k, v in lr_metrics.items():
    print(f"{k}: {v:.4f}")

print("\nTraining Random Forest baseline...")
rf_model, rf_metrics, rf_test = train_random_forest(
    df,
    feature_cols=feature_cols,
    target_col="target",
    date_col="date",
    split_date=SPLIT_DATE,
)
print("Random Forest metrics:")
for k, v in rf_metrics.items():
    print(f"{k}: {v:.4f}")

# --------------------------------------------------------------------
# 6. Save predictions for further analysis
# --------------------------------------------------------------------

out = Path("results")
out.mkdir(exist_ok=True)

lr_test.to_csv(out / "lr_baseline_predictions.csv", index=False)
rf_test.to_csv(out / "rf_baseline_predictions.csv", index=False)

print("\nSaved baseline predictions to 'results/' folder.")
print("Baseline pipeline finished!")
