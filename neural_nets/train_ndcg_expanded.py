"""
Train NDCG loss models on expanded 100-stock dataset.

This completes the novel contribution testing by comparing:
- MSE vs NDCG on 17 stocks (already done)
- MSE vs NDCG on 100 stocks (this script)

Expected: NDCG to underperform MSE (as it did on 17 stocks)
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models import CombinedNet, LateFusionNet
from neural_nets.training.train_ranker import train_ranker
from neural_nets.training.data_loader import get_train_val_test_split
import pandas as pd

print("="*70)
print("TRAINING NDCG MODELS ON EXPANDED 100-STOCK DATASET")
print("="*70)
print("Testing novel ranking loss on diverse dataset")
print()

# Load data to verify
DATA_PATH = 'data/processed/features_expanded_100stocks_with_sentiment.parquet'
df = pd.read_parquet(DATA_PATH)
print(f"Dataset: {len(df):,} samples, {df['ticker'].nunique()} stocks")

train_df, val_df, test_df = get_train_val_test_split(df)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print("="*70)

# Training parameters
num_epochs = 100
batch_size = 256
learning_rate = 0.001

results = []

# Temporarily override data loader
import neural_nets.training.data_loader as data_loader_module
original_load = data_loader_module.load_and_prepare_data

def custom_load(*args, **kwargs):
    kwargs['data_path'] = DATA_PATH
    return original_load(*args, **kwargs)

data_loader_module.load_and_prepare_data = custom_load

# ========================================================================
# MODEL 1: Combined Ranker (NDCG)
# ========================================================================
print("\n" + "="*70)
print("MODEL 1/2: Combined Ranker (NDCG Loss)")
print("="*70)
print("Testing novel ranking loss on simple architecture...")

try:
    model1, hist1 = train_ranker(
        model_class=CombinedNet,
        model_name='combined_ranker_ndcg_expanded',
        loss_type='ndcg',  # Novel loss function!
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    results.append(('Combined Ranker (NDCG)', 'SUCCESS', hist1['val_loss'][-1]))
    print("\n✅ Combined Ranker (NDCG) trained successfully!")
except Exception as e:
    print(f"\n❌ Combined Ranker (NDCG) failed: {e}")
    results.append(('Combined Ranker (NDCG)', 'FAILED', None))
    import traceback
    traceback.print_exc()

# ========================================================================
# MODEL 2: Late Fusion Ranker (NDCG)
# ========================================================================
print("\n" + "="*70)
print("MODEL 2/2: Late Fusion Ranker (NDCG Loss)")
print("="*70)
print("Testing novel ranking loss on late fusion architecture...")

try:
    model2, hist2 = train_ranker(
        model_class=LateFusionNet,
        model_name='late_fusion_ranker_ndcg_expanded',
        loss_type='ndcg',  # Novel loss function!
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    results.append(('Late Fusion Ranker (NDCG)', 'SUCCESS', hist2['val_loss'][-1]))
    print("\n✅ Late Fusion Ranker (NDCG) trained successfully!")
except Exception as e:
    print(f"\n❌ Late Fusion Ranker (NDCG) failed: {e}")
    results.append(('Late Fusion Ranker (NDCG)', 'FAILED', None))
    import traceback
    traceback.print_exc()

# ========================================================================
# SUMMARY
# ========================================================================
print("\n\n" + "="*70)
print("NDCG TRAINING COMPLETE - SUMMARY")
print("="*70)

for name, status, val_loss in results:
    loss_str = f"Val Loss: {val_loss:.6f}" if val_loss else ""
    print(f"{name:30s} {status:10s} {loss_str}")

successful = sum(1 for _, status, _ in results if status == 'SUCCESS')
print(f"\nSuccessful: {successful}/{len(results)}")

if successful == len(results):
    print("\n🎉 NDCG models trained on expanded dataset!")
    print("\nNext steps:")
    print("  1. Evaluate NDCG models")
    print("  2. Compare with MSE models")
    print("  3. Analyze why NDCG underperforms (if it does)")
else:
    print("\n⚠️ Some models failed. Check errors above.")

print("="*70)

print("\n" + "="*70)
print("COMPARISON PREVIEW")
print("="*70)
print("\nOn 17 stocks:")
print("  Combined (MSE):  0.22 Sharpe")
print("  Combined (NDCG): -1.33 Sharpe ⚠️")
print("  Late Fusion (MSE):  1.58 Sharpe")
print("  Late Fusion (NDCG): 0.21 Sharpe ⚠️")
print("\nOn 100 stocks (MSE):")
print("  Combined (MSE):  -1.01 Sharpe")
print("  Late Fusion (MSE): -0.43 Sharpe")
print("\nOn 100 stocks (NDCG) - just trained:")
print("  Results pending evaluation...")
print("="*70)

