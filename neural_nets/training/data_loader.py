"""
Data loading and preprocessing for stock prediction neural networks.
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List
from pathlib import Path


class StockDataset(Dataset):
    """PyTorch dataset for stock prediction."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        price_features: List[str],
        sentiment_features: List[str],
        target_col: str = 'target',
        future_return_col: str = 'future_return',
        task: str = 'classification'
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with features and targets
            price_features: List of price feature column names
            sentiment_features: List of sentiment feature column names
            target_col: Binary target column (for classification)
            future_return_col: Continuous target column (for regression/ranking)
            task: 'classification' or 'regression'
        """
        self.df = df.copy()
        self.price_features = price_features
        self.sentiment_features = sentiment_features
        self.all_features = price_features + sentiment_features
        self.target_col = target_col
        self.future_return_col = future_return_col
        self.task = task
        
        # Extract features and targets
        self.X_price = df[price_features].values.astype(np.float32)
        self.X_sentiment = df[sentiment_features].values.astype(np.float32)
        self.X_combined = df[self.all_features].values.astype(np.float32)
        
        if task == 'classification':
            self.y = df[target_col].values.astype(np.int64)
        else:  # regression/ranking
            self.y = df[future_return_col].values.astype(np.float32)
        
        # Store metadata for ranking loss
        self.dates = df['date'].values
        self.tickers = df['ticker'].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        return {
            'X_price': torch.from_numpy(self.X_price[idx]),
            'X_sentiment': torch.from_numpy(self.X_sentiment[idx]),
            'X_combined': torch.from_numpy(self.X_combined[idx]),
            'y': torch.tensor(self.y[idx]),
            'date': str(self.dates[idx]),  # Convert to string for PyTorch
            'ticker': str(self.tickers[idx])  # Convert to string for PyTorch
        }


def get_train_val_test_split(
    df: pd.DataFrame,
    train_end: str = '2014-01-01',
    val_end: str = '2015-07-01'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/val/test based on dates.
    
    Args:
        df: Full dataset
        train_end: End date for training set (exclusive)
        val_end: End date for validation set (exclusive)
        
    Returns:
        train_df, val_df, test_df
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_df = df[df['date'] < train_end_dt].copy()
    val_df = df[(df['date'] >= train_end_dt) & (df['date'] < val_end_dt)].copy()
    test_df = df[df['date'] >= val_end_dt].copy()
    
    print(f"Train set: {len(train_df):,} samples ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Val set:   {len(val_df):,} samples ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"Test set:  {len(test_df):,} samples ({test_df['date'].min()} to {test_df['date'].max()})")
    
    return train_df, val_df, test_df


def normalize_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Normalize features using StandardScaler fit on training data.
    
    Args:
        train_df, val_df, test_df: DataFrames to normalize
        feature_cols: Columns to normalize
        
    Returns:
        Normalized train_df, val_df, test_df, and fitted scaler
    """
    scaler = StandardScaler()
    
    # Fit on training data
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    # Transform val and test
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, val_df, test_df, scaler


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    price_features: List[str],
    sentiment_features: List[str],
    task: str = 'classification',
    batch_size: int = 256,
    normalize: bool = True
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders for train/val/test sets.
    
    Args:
        train_df, val_df, test_df: DataFrames with features and targets
        price_features: List of price feature names
        sentiment_features: List of sentiment feature names
        task: 'classification' or 'regression'
        batch_size: Batch size for training
        normalize: Whether to normalize features
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Normalize if requested
    if normalize:
        all_features = price_features + sentiment_features
        train_df, val_df, test_df, scaler = normalize_features(
            train_df.copy(), val_df.copy(), test_df.copy(), all_features
        )
    
    # Create datasets
    train_dataset = StockDataset(
        train_df, price_features, sentiment_features, task=task
    )
    val_dataset = StockDataset(
        val_df, price_features, sentiment_features, task=task
    )
    test_dataset = StockDataset(
        test_df, price_features, sentiment_features, task=task
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def load_and_prepare_data(
    data_path: str = 'data/processed/features_with_sentiment.parquet',
    train_end: str = '2014-01-01',
    val_end: str = '2015-07-01'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load dataset and split into train/val/test.
    
    Args:
        data_path: Path to parquet file
        train_end: End date for training
        val_end: End date for validation
        
    Returns:
        train_df, val_df, test_df
    """
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of stocks: {df['ticker'].nunique()}")
    
    # Split data
    train_df, val_df, test_df = get_train_val_test_split(df, train_end, val_end)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test data loading
    train_df, val_df, test_df = load_and_prepare_data()
    
    # Define features
    price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
    sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']
    
    # Create data loaders
    loaders = create_data_loaders(
        train_df, val_df, test_df,
        price_features, sentiment_features,
        task='classification',
        batch_size=256
    )
    
    # Test batch
    batch = next(iter(loaders['train']))
    print(f"\nBatch shapes:")
    print(f"  X_price: {batch['X_price'].shape}")
    print(f"  X_sentiment: {batch['X_sentiment'].shape}")
    print(f"  X_combined: {batch['X_combined'].shape}")
    print(f"  y: {batch['y'].shape}")
    
    print("\nData loading test complete!")

