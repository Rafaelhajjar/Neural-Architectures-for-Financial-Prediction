"""
Advanced neural network architectures for stock prediction.

Improvements over base models:
1. Deeper networks (4-6 layers)
2. Batch normalization
3. Better regularization
"""
import torch
import torch.nn as nn


class DeepLateFusionNet(nn.Module):
    """
    Deep Late Fusion with Batch Normalization.
    
    Improvements:
    - 6 layers per branch (vs 2-3 in base model)
    - Batch normalization after each layer
    - Dropout for regularization
    """
    
    def __init__(self, task='regression'):
        """
        Initialize deep late fusion network.
        
        Args:
            task: 'classification' or 'regression'
        """
        super().__init__()
        self.task = task
        
        # Deep price branch (6 layers)
        self.price_branch = nn.Sequential(
            nn.Linear(4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Deep sentiment branch (6 layers)
        self.sentiment_branch = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
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
            x_price: Price features (batch, 4)
            x_sentiment: Sentiment features (batch, 3)
        
        Returns:
            predictions: (batch, 2) for classification or (batch, 1) for regression
        """
        # Process each modality separately
        price_features = self.price_branch(x_price)
        sentiment_features = self.sentiment_branch(x_sentiment)
        
        # Late fusion
        combined = torch.cat([price_features, sentiment_features], dim=1)
        fused = self.fusion(combined)
        
        # Output
        return self.output(fused)


class ResidualBlock(nn.Module):
    """
    Residual block with batch normalization.
    
    Helps with gradient flow in deeper networks.
    """
    
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """Forward with residual connection."""
        return self.relu(x + self.block(x))


class ResidualLateFusionNet(nn.Module):
    """
    Late Fusion with Residual Connections.
    
    Improvements:
    - Residual blocks for better gradient flow
    - Batch normalization
    - Deeper architecture (5-6 layers)
    """
    
    def __init__(self, task='regression'):
        super().__init__()
        self.task = task
        
        # Price branch with residual blocks
        self.price_encoder = nn.Sequential(
            nn.Linear(4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.price_res1 = ResidualBlock(128, dropout=0.3)
        self.price_res2 = ResidualBlock(128, dropout=0.2)
        self.price_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Sentiment branch with residual blocks
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.sentiment_res1 = ResidualBlock(128, dropout=0.3)
        self.sentiment_res2 = ResidualBlock(128, dropout=0.2)
        self.sentiment_proj = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Output
        if task == 'classification':
            self.output = nn.Linear(32, 2)
        else:
            self.output = nn.Linear(32, 1)
    
    def forward(self, x_price, x_sentiment):
        # Price branch
        price = self.price_encoder(x_price)
        price = self.price_res1(price)
        price = self.price_res2(price)
        price = self.price_proj(price)
        
        # Sentiment branch
        sentiment = self.sentiment_encoder(x_sentiment)
        sentiment = self.sentiment_res1(sentiment)
        sentiment = self.sentiment_res2(sentiment)
        sentiment = self.sentiment_proj(sentiment)
        
        # Fuse and predict
        combined = torch.cat([price, sentiment], dim=1)
        fused = self.fusion(combined)
        return self.output(fused)


class DeepCombinedNet(nn.Module):
    """
    Deep Combined Network (early fusion) with batch normalization.
    
    Baseline to compare against deep late fusion.
    """
    
    def __init__(self, task='regression'):
        super().__init__()
        self.task = task
        
        # Deep network with early fusion
        self.network = nn.Sequential(
            nn.Linear(7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output
        if task == 'classification':
            self.output = nn.Linear(32, 2)
        else:
            self.output = nn.Linear(32, 1)
    
    def forward(self, x_price, x_sentiment):
        # Concatenate all features (early fusion)
        x = torch.cat([x_price, x_sentiment], dim=1)
        features = self.network(x)
        return self.output(features)


if __name__ == "__main__":
    print("Testing advanced model architectures...\n")
    
    # Test input
    batch_size = 32
    x_price = torch.randn(batch_size, 4)
    x_sentiment = torch.randn(batch_size, 3)
    
    # Test Deep Late Fusion
    print("1. Deep Late Fusion Net")
    model1 = DeepLateFusionNet(task='regression')
    out1 = model1(x_price, x_sentiment)
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"   Output shape: {out1.shape}")
    print(f"   Parameters: {params1:,}")
    print(f"   ✅ Working\n")
    
    # Test Residual Late Fusion
    print("2. Residual Late Fusion Net")
    model2 = ResidualLateFusionNet(task='regression')
    out2 = model2(x_price, x_sentiment)
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"   Output shape: {out2.shape}")
    print(f"   Parameters: {params2:,}")
    print(f"   ✅ Working\n")
    
    # Test Deep Combined
    print("3. Deep Combined Net")
    model3 = DeepCombinedNet(task='regression')
    out3 = model3(x_price, x_sentiment)
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"   Output shape: {out3.shape}")
    print(f"   Parameters: {params3:,}")
    print(f"   ✅ Working\n")
    
    print("="*60)
    print("All advanced models initialized successfully!")
    print("="*60)

