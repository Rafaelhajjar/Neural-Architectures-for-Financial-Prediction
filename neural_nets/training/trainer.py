"""
Training loop and utilities for neural network models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Callable
import time


class Trainer:
    """Trainer class for neural network models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 20,
        checkpoint_dir: str = 'neural_nets/trained_models'
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on ('cpu' or 'cuda')
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            # Get data
            x_price = batch['X_price'].to(self.device)
            x_sentiment = batch['X_sentiment'].to(self.device)
            y = batch['y'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x_price, x_sentiment)
            
            # Compute loss
            if self.model.task == 'classification':
                loss = self.criterion(outputs, y)
                
                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)
            else:  # regression
                # Check if criterion accepts dates (for per-date ranking losses)
                import inspect
                dates = batch.get('date', None)
                loss_sig = inspect.signature(self.criterion.forward)
                
                if 'dates' in loss_sig.parameters and dates is not None:
                    # Per-date ranking loss
                    loss = self.criterion(outputs.squeeze(), y, dates=dates)
                else:
                    # Standard loss (MSE, etc.)
                    loss = self.criterion(outputs.squeeze(), y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * y.size(0)
        
        avg_loss = total_loss / len(self.train_loader.dataset)
        
        metrics = {'loss': avg_loss}
        if self.model.task == 'classification':
            metrics['accuracy'] = 100.0 * correct / total
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # Get data
                x_price = batch['X_price'].to(self.device)
                x_sentiment = batch['X_sentiment'].to(self.device)
                y = batch['y'].to(self.device)
                
                # Forward pass
                outputs = self.model(x_price, x_sentiment)
                
                # Compute loss
                if self.model.task == 'classification':
                    loss = self.criterion(outputs, y)
                    
                    # Compute accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                else:  # regression
                    # Check if criterion accepts dates
                    import inspect
                    dates = batch.get('date', None)
                    loss_sig = inspect.signature(self.criterion.forward)
                    
                    if 'dates' in loss_sig.parameters and dates is not None:
                        loss = self.criterion(outputs.squeeze(), y, dates=dates)
                    else:
                        loss = self.criterion(outputs.squeeze(), y)
                
                total_loss += loss.item() * y.size(0)
        
        avg_loss = total_loss / len(self.val_loader.dataset)
        
        metrics = {'loss': avg_loss}
        if self.model.task == 'classification':
            metrics['accuracy'] = 100.0 * correct / total
        
        return metrics
    
    def train(
        self,
        num_epochs: int,
        model_name: str = 'model',
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            model_name: Name for saving checkpoints
            verbose: Whether to print progress
        
        Returns:
            Training history
        """
        if verbose:
            print(f"Training {model_name} for {num_epochs} epochs...")
            print(f"Device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print("="*70)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            
            if self.model.task == 'classification':
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_acc'].append(val_metrics['accuracy'])
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict().copy()
                self.save_checkpoint(model_name + '_best')
            else:
                self.patience_counter += 1
            
            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start
                print(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}", end='')
                if self.model.task == 'classification':
                    print(f" | Train Acc: {train_metrics['accuracy']:.2f}%", end='')
                print()
                print(f"  Val Loss:   {val_metrics['loss']:.4f}", end='')
                if self.model.task == 'classification':
                    print(f" | Val Acc:   {val_metrics['accuracy']:.2f}%", end='')
                print(f" | LR: {current_lr:.6f}")
                
                if self.patience_counter > 0:
                    print(f"  Early stopping: {self.patience_counter}/{self.early_stopping_patience}")
                print()
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if verbose:
                print(f"\nLoaded best model (val_loss: {self.best_val_loss:.4f})")
        
        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")
            print("="*70)
        
        return self.history
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
    
    def load_checkpoint(self, name: str):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']


if __name__ == "__main__":
    print("Trainer class defined successfully!")

