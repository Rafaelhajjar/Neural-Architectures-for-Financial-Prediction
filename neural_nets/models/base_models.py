"""
Base neural network architectures for stock prediction.
"""
import torch
import torch.nn as nn


class PriceOnlyNet(nn.Module):
    """Neural network using only price features."""
    
    def __init__(self, task='classification'):
        """
        Initialize model.
        
        Args:
            task: 'classification' (2 outputs) or 'regression' (1 output)
        """
        super().__init__()
        self.task = task
        
        # Architecture: Input(4) → 64 → 32 → Output
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layer depends on task
        if task == 'classification':
            self.output = nn.Linear(32, 2)  # Binary classification
        else:  # regression
            self.output = nn.Linear(32, 1)  # Continuous output
    
    def forward(self, x_price, x_sentiment=None):
        """
        Forward pass.
        
        Args:
            x_price: Price features (batch_size, 4)
            x_sentiment: Ignored (for interface consistency)
            
        Returns:
            Predictions (batch_size, 2) or (batch_size, 1)
        """
        x = self.network(x_price)
        return self.output(x)


class CombinedNet(nn.Module):
    """Neural network with simple concatenation of all features."""
    
    def __init__(self, task='classification'):
        """
        Initialize model.
        
        Args:
            task: 'classification' or 'regression'
        """
        super().__init__()
        self.task = task
        
        # Architecture: Input(7) → 128 → 64 → 32 → Output
        self.network = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Output layer
        if task == 'classification':
            self.output = nn.Linear(32, 2)
        else:
            self.output = nn.Linear(32, 1)
    
    def forward(self, x_price, x_sentiment):
        """
        Forward pass.
        
        Args:
            x_price: Price features (batch_size, 4)
            x_sentiment: Sentiment features (batch_size, 3)
            
        Returns:
            Predictions
        """
        # Concatenate all features
        x = torch.cat([x_price, x_sentiment], dim=1)
        x = self.network(x)
        return self.output(x)


class LateFusionNet(nn.Module):
    """Neural network with separate branches for price and sentiment."""
    
    def __init__(self, task='classification'):
        """
        Initialize model.
        
        Args:
            task: 'classification' or 'regression'
        """
        super().__init__()
        self.task = task
        
        # Price branch: Input(4) → 64 → 32
        self.price_branch = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Sentiment branch: Input(3) → 64 → 32
        self.sentiment_branch = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Fusion layer: Concat(64) → 32 → Output
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output layer
        if task == 'classification':
            self.output = nn.Linear(32, 2)
        else:
            self.output = nn.Linear(32, 1)
    
    def forward(self, x_price, x_sentiment):
        """
        Forward pass.
        
        Args:
            x_price: Price features (batch_size, 4)
            x_sentiment: Sentiment features (batch_size, 3)
            
        Returns:
            Predictions
        """
        # Process each modality separately
        price_features = self.price_branch(x_price)
        sentiment_features = self.sentiment_branch(x_sentiment)
        
        # Concatenate and fuse
        combined = torch.cat([price_features, sentiment_features], dim=1)
        fused = self.fusion(combined)
        
        return self.output(fused)


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    batch_size = 32
    x_price = torch.randn(batch_size, 4)
    x_sentiment = torch.randn(batch_size, 3)
    
    print("Testing model architectures...\n")
    
    # Test PriceOnlyNet
    print("1. PriceOnlyNet (Classification)")
    model = PriceOnlyNet(task='classification')
    out = model(x_price, x_sentiment)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(model):,}\n")
    
    print("2. PriceOnlyNet (Regression)")
    model = PriceOnlyNet(task='regression')
    out = model(x_price, x_sentiment)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(model):,}\n")
    
    # Test CombinedNet
    print("3. CombinedNet (Classification)")
    model = CombinedNet(task='classification')
    out = model(x_price, x_sentiment)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(model):,}\n")
    
    print("4. CombinedNet (Regression)")
    model = CombinedNet(task='regression')
    out = model(x_price, x_sentiment)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(model):,}\n")
    
    # Test LateFusionNet
    print("5. LateFusionNet (Classification)")
    model = LateFusionNet(task='classification')
    out = model(x_price, x_sentiment)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(model):,}\n")
    
    print("6. LateFusionNet (Regression)")
    model = LateFusionNet(task='regression')
    out = model(x_price, x_sentiment)
    print(f"   Output shape: {out.shape}")
    print(f"   Parameters: {count_parameters(model):,}\n")
    
    print("All models initialized successfully!")

