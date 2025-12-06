"""
Training script for controlled early vs late fusion comparison.

Trains two models with ~100K parameters each:
1. Early Fusion (concatenate at input)
2. Late Fusion (separate branches, fuse later)

Both trained with:
- Same loss function (MSE)
- Same hyperparameters
- Same dataset (100 stocks expanded)
- Same training procedure

This provides a fair, controlled comparison of fusion strategies.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models.controlled_fusion import EarlyFusion100K, LateFusion100K, count_parameters
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders
from neural_nets.training.trainer import Trainer


def train_controlled_comparison(device='cpu'):
    """
    Train both models for controlled comparison.
    """
    print("="*80)
    print("CONTROLLED FUSION COMPARISON TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print()
    
    # Load data
    print("Loading 100-stock expanded dataset...")
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_expanded_100stocks_with_sentiment.parquet'
    
    train_df, val_df, test_df = load_and_prepare_data(
        data_path=str(data_path),
        train_end='2013-12-31',
        val_end='2015-06-30'
    )
    
    print(f"Train samples: {len(train_df):,}")
    print(f"Val samples:   {len(val_df):,}")
    print(f"Test samples:  {len(test_df):,}")
    print()
    
    # Features
    price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
    sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']
    
    # Create data loaders
    loaders = create_data_loaders(
        train_df, val_df, test_df,
        price_features, sentiment_features,
        task='regression',
        batch_size=256
    )
    
    # Training configuration
    config = {
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'max_epochs': 100,
        'patience': 20,
        'device': device
    }
    
    # Loss function (same for both)
    criterion = nn.MSELoss()
    
    results = {}
    
    # ========================================
    # MODEL 1: EARLY FUSION
    # ========================================
    print("="*80)
    print("MODEL 1: EARLY FUSION (Concatenate at Input)")
    print("="*80)
    
    early_model = EarlyFusion100K(task='regression').to(device)
    early_params = count_parameters(early_model)
    
    print(f"Parameters: {early_params:,}")
    print(f"Architecture: Input(7) → 256 → 256 → 128 → 64 → 32 → Output")
    print()
    
    # Create optimizer and scheduler
    early_optimizer = torch.optim.Adam(
        early_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    early_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        early_optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train
    early_trainer = Trainer(
        model=early_model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=early_optimizer,
        device=config['device'],
        scheduler=early_scheduler,
        early_stopping_patience=config['patience']
    )
    
    print("Training Early Fusion model...")
    early_history = early_trainer.train(
        num_epochs=config['max_epochs'],
        model_name='early_fusion_100k',
        verbose=True
    )
    
    # Add best val loss to history
    early_history['best_val_loss'] = early_trainer.best_val_loss
    early_history['best_epoch'] = len(early_history['train_loss']) - early_trainer.patience_counter
    
    # Save model
    save_path = Path(__file__).parent / 'trained_models' / 'early_fusion_100k_best.pt'
    torch.save({
        'model_state_dict': early_trainer.best_model_state,
        'train_history': early_history,
        'config': config,
        'parameters': early_params
    }, save_path)
    
    print(f"\n✅ Early Fusion training complete!")
    print(f"Best val loss: {early_history['best_val_loss']:.6f}")
    print(f"Model saved to: {save_path}")
    print()
    
    results['early'] = {
        'model': early_model,
        'history': early_history,
        'params': early_params
    }
    
    # ========================================
    # MODEL 2: LATE FUSION
    # ========================================
    print("="*80)
    print("MODEL 2: LATE FUSION (Separate Branches)")
    print("="*80)
    
    late_model = LateFusion100K(task='regression').to(device)
    late_params = count_parameters(late_model)
    
    print(f"Parameters: {late_params:,}")
    print(f"Architecture:")
    print(f"  Price branch:     4 → 180 → 180 → 90")
    print(f"  Sentiment branch: 3 → 180 → 180 → 90")
    print(f"  Fusion:          180 → 80 → 32 → Output")
    print()
    
    # Create optimizer and scheduler
    late_optimizer = torch.optim.Adam(
        late_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    late_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        late_optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train
    late_trainer = Trainer(
        model=late_model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=late_optimizer,
        device=config['device'],
        scheduler=late_scheduler,
        early_stopping_patience=config['patience']
    )
    
    print("Training Late Fusion model...")
    late_history = late_trainer.train(
        num_epochs=config['max_epochs'],
        model_name='late_fusion_100k',
        verbose=True
    )
    
    # Add best val loss to history
    late_history['best_val_loss'] = late_trainer.best_val_loss
    late_history['best_epoch'] = len(late_history['train_loss']) - late_trainer.patience_counter
    
    # Save model
    save_path = Path(__file__).parent / 'trained_models' / 'late_fusion_100k_best.pt'
    torch.save({
        'model_state_dict': late_trainer.best_model_state,
        'train_history': late_history,
        'config': config,
        'parameters': late_params
    }, save_path)
    
    print(f"\n✅ Late Fusion training complete!")
    print(f"Best val loss: {late_history['best_val_loss']:.6f}")
    print(f"Model saved to: {save_path}")
    print()
    
    results['late'] = {
        'model': late_model,
        'history': late_history,
        'params': late_params
    }
    
    # ========================================
    # TRAINING SUMMARY
    # ========================================
    print("="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print(f"\nEarly Fusion ({results['early']['params']:,} params):")
    print(f"  Best val loss: {results['early']['history']['best_val_loss']:.6f}")
    print(f"  Best epoch:    {results['early']['history']['best_epoch']}")
    print(f"  Total epochs:  {len(results['early']['history']['train_loss'])}")
    
    print(f"\nLate Fusion ({results['late']['params']:,} params):")
    print(f"  Best val loss: {results['late']['history']['best_val_loss']:.6f}")
    print(f"  Best epoch:    {results['late']['history']['best_epoch']}")
    print(f"  Total epochs:  {len(results['late']['history']['train_loss'])}")
    
    # Compare
    early_loss = results['early']['history']['best_val_loss']
    late_loss = results['late']['history']['best_val_loss']
    
    if early_loss < late_loss:
        winner = "Early Fusion"
        diff_pct = ((late_loss - early_loss) / early_loss) * 100
        print(f"\n🏆 {winner} wins by {diff_pct:.2f}%")
    elif late_loss < early_loss:
        winner = "Late Fusion"
        diff_pct = ((early_loss - late_loss) / late_loss) * 100
        print(f"\n🏆 {winner} wins by {diff_pct:.2f}%")
    else:
        print(f"\n🤝 Tie!")
    
    print("\n" + "="*80)
    print("Next step: Run evaluation script to compute trading metrics")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train controlled fusion comparison')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    args = parser.parse_args()
    
    # Train both models
    results = train_controlled_comparison(device=args.device)
    
    print("\n✅ All training complete!")

