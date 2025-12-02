"""
Training script for classification models (predict up/down).
"""
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_nets.models import PriceOnlyNet, CombinedNet, LateFusionNet
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders
from neural_nets.training.trainer import Trainer


def train_classifier(
    model_class,
    model_name: str,
    use_sentiment: bool = True,
    device: str = 'cpu',
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001
):
    """
    Train a classification model.
    
    Args:
        model_class: Model class to instantiate
        model_name: Name for saving model
        use_sentiment: Whether model uses sentiment features
        device: Device to train on
        num_epochs: Maximum number of epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
    
    Returns:
        Trained model and history
    """
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
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
        task='classification',
        batch_size=batch_size,
        normalize=True
    )
    
    # Initialize model
    model = model_class(task='classification')
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
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
    """Train all classification models."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # 1. Price-only classifier
    print("\n" + "="*70)
    print("MODEL 1: Price-Only Classifier")
    print("="*70)
    model1, hist1 = train_classifier(
        PriceOnlyNet,
        'price_only_classifier',
        use_sentiment=False,
        device=device
    )
    
    # 2. Combined classifier
    print("\n" + "="*70)
    print("MODEL 2: Combined Classifier")
    print("="*70)
    model2, hist2 = train_classifier(
        CombinedNet,
        'combined_classifier',
        use_sentiment=True,
        device=device
    )
    
    # 3. Late fusion classifier
    print("\n" + "="*70)
    print("MODEL 3: Late Fusion Classifier")
    print("="*70)
    model3, hist3 = train_classifier(
        LateFusionNet,
        'late_fusion_classifier',
        use_sentiment=True,
        device=device
    )
    
    print("\n" + "="*70)
    print("ALL CLASSIFICATION MODELS TRAINED!")
    print("="*70)
    print("\nSaved models:")
    print("  1. neural_nets/trained_models/price_only_classifier_best.pt")
    print("  2. neural_nets/trained_models/combined_classifier_best.pt")
    print("  3. neural_nets/trained_models/late_fusion_classifier_best.pt")


if __name__ == "__main__":
    main()

