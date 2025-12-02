"""
Training script for ranking models (predict returns with MSE and NDCG loss).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_nets.models import CombinedNet, LateFusionNet, NDCGLoss
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders
from neural_nets.training.trainer import Trainer


def train_ranker(
    model_class,
    model_name: str,
    loss_type: str = 'mse',
    device: str = 'cpu',
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001
):
    """
    Train a ranking model.
    
    Args:
        model_class: Model class to instantiate
        model_name: Name for saving model
        loss_type: 'mse' or 'ndcg'
        device: Device to train on
        num_epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
    
    Returns:
        Trained model and history
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name} with {loss_type.upper()} loss")
    print(f"{'='*70}\n")
    
    # Load data
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Define features
    price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
    sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']
    
    # Create data loaders
    loaders = create_data_loaders(
        train_df, val_df, test_df,
        price_features, sentiment_features,
        task='regression',  # Ranking models predict continuous returns
        batch_size=batch_size,
        normalize=True
    )
    
    # Initialize model
    model = model_class(task='regression')
    
    # Loss function
    if loss_type == 'mse':
        criterion = nn.MSELoss()
    elif loss_type == 'ndcg':
        # Use per-date NDCG loss (proper ranking within each trading day)
        from neural_nets.models.ranking_losses import PerDateNDCGLoss
        criterion = PerDateNDCGLoss(k=5, temperature=0.1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=20
    )
    
    # Train
    history = trainer.train(num_epochs=num_epochs, model_name=model_name, verbose=True)
    
    return model, history


def main():
    """Train all ranking models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 4. Combined ranker (MSE baseline)
    print("\n" + "="*70)
    print("MODEL 4: Combined Ranker (MSE Baseline)")
    print("="*70)
    model4, hist4 = train_ranker(
        CombinedNet,
        'combined_ranker_mse',
        loss_type='mse',
        device=device
    )
    
    # 5. Combined ranker (NDCG)
    print("\n" + "="*70)
    print("MODEL 5: Combined Ranker (NDCG Loss)")
    print("="*70)
    model5, hist5 = train_ranker(
        CombinedNet,
        'combined_ranker_ndcg',
        loss_type='ndcg',
        device=device
    )
    
    # 6. Late fusion ranker (MSE baseline)
    print("\n" + "="*70)
    print("MODEL 6: Late Fusion Ranker (MSE Baseline)")
    print("="*70)
    model6, hist6 = train_ranker(
        LateFusionNet,
        'late_fusion_ranker_mse',
        loss_type='mse',
        device=device
    )
    
    # 7. Late fusion ranker (NDCG)
    print("\n" + "="*70)
    print("MODEL 7: Late Fusion Ranker (NDCG Loss)")
    print("="*70)
    model7, hist7 = train_ranker(
        LateFusionNet,
        'late_fusion_ranker_ndcg',
        loss_type='ndcg',
        device=device
    )
    
    print("\n" + "="*70)
    print("ALL RANKING MODELS TRAINED!")
    print("="*70)
    print("\nSaved models:")
    print("  4. neural_nets/trained_models/combined_ranker_mse_best.pt")
    print("  5. neural_nets/trained_models/combined_ranker_ndcg_best.pt")
    print("  6. neural_nets/trained_models/late_fusion_ranker_mse_best.pt")
    print("  7. neural_nets/trained_models/late_fusion_ranker_ndcg_best.pt")


if __name__ == "__main__":
    main()

