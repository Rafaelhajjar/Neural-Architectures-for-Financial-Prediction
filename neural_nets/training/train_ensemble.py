"""
Train ensemble of models with different random seeds.

Ensemble improves robustness and often improves performance by 10-15%.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders
from neural_nets.training.trainer import Trainer


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.backends.cudnn.deterministic = True  # Slower but deterministic


def train_ensemble_member(
    model_class,
    model_name,
    seed,
    device='cpu',
    num_epochs=100,
    batch_size=256,
    learning_rate=0.001
):
    """
    Train a single ensemble member with a specific seed.
    
    Args:
        model_class: Model class to instantiate
        model_name: Base name for saving
        seed: Random seed
        device: 'cpu' or 'cuda'
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        model: Trained model
        history: Training history
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name} (seed={seed})")
    print("="*70)
    
    # Set seed
    set_seed(seed)
    
    # Load data
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Features
    price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
    sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']
    
    # Create data loaders
    loaders = create_data_loaders(
        train_df, val_df, test_df,
        price_features, sentiment_features,
        task='regression',  # For ranking
        batch_size=batch_size
    )
    
    # Initialize model
    model = model_class(task='regression')
    model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
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
    history = trainer.train(num_epochs=num_epochs, model_name=f"{model_name}_seed{seed}", verbose=True)
    
    return model, history


def train_ensemble(
    model_class,
    base_name,
    num_members=5,
    device='cpu',
    num_epochs=100,
    batch_size=256,
    learning_rate=0.001
):
    """
    Train an ensemble of models.
    
    Args:
        model_class: Model class to instantiate
        base_name: Base name for ensemble
        num_members: Number of ensemble members
        device: 'cpu' or 'cuda'
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        models: List of trained models
        histories: List of training histories
    """
    print("\n" + "="*70)
    print(f"TRAINING ENSEMBLE: {base_name}")
    print(f"Members: {num_members}")
    print("="*70)
    
    models = []
    histories = []
    
    # Train each member with different seed
    for i in range(num_members):
        seed = 42 + i  # Seeds: 42, 43, 44, 45, 46
        model, history = train_ensemble_member(
            model_class=model_class,
            model_name=base_name,
            seed=seed,
            device=device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        models.append(model)
        histories.append(history)
        
        print(f"\n✅ Member {i+1}/{num_members} complete")
    
    print("\n" + "="*70)
    print(f"ENSEMBLE TRAINING COMPLETE: {base_name}")
    print("="*70)
    
    return models, histories


class EnsemblePredictor:
    """
    Ensemble predictor that averages predictions from multiple models.
    """
    
    def __init__(self, models):
        """
        Initialize ensemble predictor.
        
        Args:
            models: List of trained models
        """
        self.models = models
        for model in self.models:
            model.eval()
    
    def predict(self, x_price, x_sentiment):
        """
        Get ensemble prediction (average of all models).
        
        Args:
            x_price: Price features
            x_sentiment: Sentiment features
        
        Returns:
            Averaged predictions
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x_price, x_sentiment)
                predictions.append(pred)
        
        # Average predictions
        return torch.stack(predictions).mean(dim=0)
    
    def __call__(self, x_price, x_sentiment):
        """Allow calling like a regular model."""
        return self.predict(x_price, x_sentiment)


if __name__ == "__main__":
    print("Ensemble training module ready!")
    print("\nExample usage:")
    print("  from neural_nets.training.train_ensemble import train_ensemble")
    print("  from neural_nets.models.advanced_models import DeepLateFusionNet")
    print("")
    print("  models, histories = train_ensemble(")
    print("      model_class=DeepLateFusionNet,")
    print("      base_name='deep_late_fusion_ensemble',")
    print("      num_members=5")
    print("  )")

