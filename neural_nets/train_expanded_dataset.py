"""
Train all neural network models on the expanded 100-stock dataset.

This uses the same model architectures but with 6x more data (100 stocks vs 17).
Expected result: Lower performance but more reliable/generalizable.
"""
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models import CombinedNet, LateFusionNet
from neural_nets.models.advanced_models import (
    DeepLateFusionNet,
    ResidualLateFusionNet,
    DeepCombinedNet
)
from neural_nets.training.train_ranker import train_ranker
from neural_nets.training.train_ensemble import train_ensemble
from neural_nets.training.data_loader import load_and_prepare_data

# Override the data path to use expanded dataset
DATA_PATH = 'data/processed/features_expanded_100stocks_with_sentiment.parquet'

print("="*70)
print("TRAINING ON EXPANDED 100-STOCK DATASET")
print("="*70)
print(f"Dataset: {DATA_PATH}")
print()

# Load data to verify
from neural_nets.training.data_loader import create_data_loaders, get_train_val_test_split
import pandas as pd

df = pd.read_parquet(DATA_PATH)
print(f"Total samples: {len(df):,}")
print(f"Stocks: {df['ticker'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

train_df, val_df, test_df = get_train_val_test_split(df)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nDevice: {device}")
print("="*70)

# Training parameters
num_epochs = 100
batch_size = 256
learning_rate = 0.001

results = []

# ========================================================================
# MODEL 1: Combined Ranker (MSE) - Baseline
# ========================================================================
print("\n" + "="*70)
print("MODEL 1/5: Combined Ranker (MSE Baseline)")
print("="*70)

try:
    # Temporarily override data loader
    import neural_nets.training.data_loader as data_loader_module
    original_load = data_loader_module.load_and_prepare_data
    
    def custom_load(*args, **kwargs):
        kwargs['data_path'] = DATA_PATH
        return original_load(*args, **kwargs)
    
    data_loader_module.load_and_prepare_data = custom_load
    
    model1, hist1 = train_ranker(
        model_class=CombinedNet,
        model_name='combined_ranker_mse_expanded',
        loss_type='mse',
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    results.append(('Combined Ranker (MSE)', 'SUCCESS', hist1['val_loss'][-1]))
    print("\n✅ Combined Ranker trained successfully!")
except Exception as e:
    print(f"\n❌ Combined Ranker failed: {e}")
    results.append(('Combined Ranker (MSE)', 'FAILED', None))

# ========================================================================
# MODEL 2: Late Fusion Ranker (MSE) - Main Model
# ========================================================================
print("\n" + "="*70)
print("MODEL 2/5: Late Fusion Ranker (MSE)")
print("="*70)

try:
    model2, hist2 = train_ranker(
        model_class=LateFusionNet,
        model_name='late_fusion_ranker_mse_expanded',
        loss_type='mse',
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    results.append(('Late Fusion Ranker (MSE)', 'SUCCESS', hist2['val_loss'][-1]))
    print("\n✅ Late Fusion Ranker trained successfully!")
except Exception as e:
    print(f"\n❌ Late Fusion Ranker failed: {e}")
    results.append(('Late Fusion Ranker (MSE)', 'FAILED', None))

# ========================================================================
# MODEL 3: Deep Late Fusion
# ========================================================================
print("\n" + "="*70)
print("MODEL 3/5: Deep Late Fusion")
print("="*70)

try:
    model3, hist3 = train_ranker(
        model_class=DeepLateFusionNet,
        model_name='deep_late_fusion_mse_expanded',
        loss_type='mse',
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    results.append(('Deep Late Fusion', 'SUCCESS', hist3['val_loss'][-1]))
    print("\n✅ Deep Late Fusion trained successfully!")
except Exception as e:
    print(f"\n❌ Deep Late Fusion failed: {e}")
    results.append(('Deep Late Fusion', 'FAILED', None))

# ========================================================================
# MODEL 4: Deep Combined
# ========================================================================
print("\n" + "="*70)
print("MODEL 4/5: Deep Combined")
print("="*70)

try:
    model4, hist4 = train_ranker(
        model_class=DeepCombinedNet,
        model_name='deep_combined_mse_expanded',
        loss_type='mse',
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    results.append(('Deep Combined', 'SUCCESS', hist4['val_loss'][-1]))
    print("\n✅ Deep Combined trained successfully!")
except Exception as e:
    print(f"\n❌ Deep Combined failed: {e}")
    results.append(('Deep Combined', 'FAILED', None))

# ========================================================================
# MODEL 5: Ensemble (3 members to save time)
# ========================================================================
print("\n" + "="*70)
print("MODEL 5/5: Late Fusion Ensemble (3 members)")
print("="*70)

try:
    models_ens, hists_ens = train_ensemble(
        model_class=LateFusionNet,
        base_name='late_fusion_ensemble_expanded',
        num_members=3,  # Reduced from 5 to save time
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    avg_val_loss = sum(h['val_loss'][-1] for h in hists_ens) / len(hists_ens)
    results.append(('Late Fusion Ensemble', 'SUCCESS', avg_val_loss))
    print("\n✅ Ensemble trained successfully!")
    
    # Save ensemble metadata
    import json
    ensemble_info = {
        'num_members': 3,
        'seeds': [42, 43, 44],
        'base_model': 'LateFusionNet',
        'dataset': 'expanded_100stocks',
        'val_losses': [h['val_loss'][-1] for h in hists_ens],
        'avg_val_loss': avg_val_loss
    }
    
    with open('neural_nets/trained_models/late_fusion_ensemble_expanded_info.json', 'w') as f:
        json.dump(ensemble_info, f, indent=2)
    
    print(f"\nEnsemble info saved")
    
except Exception as e:
    print(f"\n❌ Ensemble failed: {e}")
    results.append(('Late Fusion Ensemble', 'FAILED', None))

# ========================================================================
# SUMMARY
# ========================================================================
print("\n\n" + "="*70)
print("TRAINING COMPLETE - SUMMARY")
print("="*70)

for name, status, val_loss in results:
    loss_str = f"Val Loss: {val_loss:.6f}" if val_loss else ""
    print(f"{name:30s} {status:10s} {loss_str}")

successful = sum(1 for _, status, _ in results if status == 'SUCCESS')
print(f"\nSuccessful: {successful}/{len(results)}")

if successful == len(results):
    print("\n🎉 All models trained on expanded dataset!")
    print("\nExpected changes vs 17-stock dataset:")
    print("  - Lower Sharpe ratio (more realistic)")
    print("  - Lower absolute returns")
    print("  - But much more generalizable!")
    print("\nNext: Evaluate models and compare to original results")
else:
    print("\n⚠️ Some models failed. Check errors above.")

print("="*70)

