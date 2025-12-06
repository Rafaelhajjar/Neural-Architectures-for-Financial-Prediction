"""
Train NDCG loss models on expanded 100-stock dataset (FIXED VERSION).

Properly loads the 100-stock expanded dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models import CombinedNet, LateFusionNet
from neural_nets.models.ranking_losses import PerDateNDCGLoss
from neural_nets.training.data_loader import get_train_val_test_split, create_data_loaders
from neural_nets.training.trainer import Trainer
import pandas as pd

print("="*70)
print("TRAINING NDCG MODELS ON EXPANDED 100-STOCK DATASET (FIXED)")
print("="*70)
print("Testing novel ranking loss on 100-stock diverse dataset")
print()

# Load EXPANDED dataset
DATA_PATH = 'data/processed/features_expanded_100stocks_with_sentiment.parquet'
print(f"Loading: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"✅ Dataset loaded: {len(df):,} samples, {df['ticker'].nunique()} stocks")
print()

# Split data
train_df, val_df, test_df = get_train_val_test_split(df)

# Define features
price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']

# Create data loaders
print("Creating data loaders...")
loaders = create_data_loaders(
    train_df, val_df, test_df,
    price_features, sentiment_features,
    task='regression',
    batch_size=256,
    normalize=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print("="*70)

results = []

# ========================================================================
# MODEL 1: Combined Ranker (NDCG)
# ========================================================================
print("\n" + "="*70)
print("MODEL 1/2: Combined Ranker (NDCG Loss) on 100 stocks")
print("="*70)

try:
    # Initialize model
    model1 = CombinedNet(task='regression')
    model1 = model1.to(device)
    
    # NDCG loss
    criterion1 = PerDateNDCGLoss(k=5, temperature=0.1)
    
    # Optimizer
    optimizer1 = optim.Adam(model1.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Scheduler
    scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode='min', factor=0.5, patience=10
    )
    
    # Trainer
    trainer1 = Trainer(
        model=model1,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion1,
        optimizer=optimizer1,
        device=device,
        scheduler=scheduler1,
        early_stopping_patience=20
    )
    
    # Train
    print("Training Combined Ranker with NDCG loss...")
    history1 = trainer1.train(
        num_epochs=100, 
        model_name='combined_ranker_ndcg_expanded_fixed',
        verbose=True
    )
    
    results.append(('Combined Ranker (NDCG)', 'SUCCESS', history1['val_loss'][-1]))
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
print("MODEL 2/2: Late Fusion Ranker (NDCG Loss) on 100 stocks")
print("="*70)

try:
    # Initialize model
    model2 = LateFusionNet(task='regression')
    model2 = model2.to(device)
    
    # NDCG loss
    criterion2 = PerDateNDCGLoss(k=5, temperature=0.1)
    
    # Optimizer
    optimizer2 = optim.Adam(model2.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Scheduler
    scheduler2 = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='min', factor=0.5, patience=10
    )
    
    # Trainer
    trainer2 = Trainer(
        model=model2,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion2,
        optimizer=optimizer2,
        device=device,
        scheduler=scheduler2,
        early_stopping_patience=20
    )
    
    # Train
    print("Training Late Fusion Ranker with NDCG loss...")
    history2 = trainer2.train(
        num_epochs=100,
        model_name='late_fusion_ranker_ndcg_expanded_fixed',
        verbose=True
    )
    
    results.append(('Late Fusion Ranker (NDCG)', 'SUCCESS', history2['val_loss'][-1]))
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
    print("\n🎉 NDCG models trained on 100-stock expanded dataset!")
    print("\nNext: Evaluate and compare with MSE models")
else:
    print("\n⚠️ Some models failed. Check errors above.")

print("="*70)

