"""
Custom loss functions for stock prediction.
"""
import torch
import torch.nn as nn
import numpy as np


class NDCGLoss(nn.Module):
    """
    Normalized Discounted Cumulative Gain Loss for ranking.
    
    NDCG measures ranking quality by emphasizing correct ordering at top positions.
    Higher NDCG = better ranking.
    Loss = 1 - NDCG (to minimize)
    """
    
    def __init__(self, k=5, temperature=1.0):
        """
        Initialize NDCG loss.
        
        Args:
            k: Consider top-k positions (for long/short strategy)
            temperature: Softmax temperature for smooth approximation
        """
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, predictions, targets, groups=None):
        """
        Compute NDCG loss.
        
        Args:
            predictions: Predicted returns (batch_size, 1) or (batch_size,)
            targets: Actual returns (batch_size, 1) or (batch_size,)
            groups: Optional group indices (e.g., date IDs) for batch ranking
                   If None, treats entire batch as one ranking problem
        
        Returns:
            Loss value (1 - NDCG)
        """
        # Flatten if needed
        if predictions.dim() == 2:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        # If no groups, treat entire batch as one group
        if groups is None:
            return 1.0 - self._compute_ndcg(predictions, targets)
        
        # Otherwise, compute NDCG for each group and average
        unique_groups = torch.unique(groups)
        ndcg_scores = []
        
        for group in unique_groups:
            mask = (groups == group)
            group_preds = predictions[mask]
            group_targets = targets[mask]
            
            if len(group_preds) > 1:  # Need at least 2 items to rank
                ndcg = self._compute_ndcg(group_preds, group_targets)
                ndcg_scores.append(ndcg)
        
        if len(ndcg_scores) == 0:
            return torch.tensor(1.0, device=predictions.device)
        
        avg_ndcg = torch.stack(ndcg_scores).mean()
        return 1.0 - avg_ndcg
    
    def _compute_ndcg(self, predictions, targets):
        """
        Compute NDCG for a single group.
        
        Args:
            predictions: Predicted scores (n,)
            targets: Actual relevance scores (n,)
        
        Returns:
            NDCG@k score
        """
        k = min(self.k, len(predictions))
        
        # Get top-k indices by predictions
        _, pred_indices = torch.topk(predictions, k, largest=True)
        
        # Get actual relevance in predicted order
        pred_relevance = targets[pred_indices]
        
        # Compute DCG (Discounted Cumulative Gain)
        positions = torch.arange(1, k + 1, device=predictions.device).float()
        discounts = torch.log2(positions + 1)
        dcg = (pred_relevance / discounts).sum()
        
        # Compute ideal DCG (best possible ordering)
        ideal_relevance, _ = torch.sort(targets, descending=True)
        ideal_relevance = ideal_relevance[:k]
        idcg = (ideal_relevance / discounts).sum()
        
        # Avoid division by zero
        if idcg == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        return dcg / idcg


class ApproxNDCGLoss(nn.Module):
    """
    Differentiable approximation of NDCG using smooth ranking.
    
    Uses softmax to create differentiable "soft" rankings instead of hard rankings.
    This allows gradient flow through the ranking operation.
    """
    
    def __init__(self, k=5, temperature=1.0):
        """
        Initialize approximated NDCG loss.
        
        Args:
            k: Top-k positions to consider
            temperature: Softmax temperature (lower = sharper approximation)
        """
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, predictions, targets, groups=None):
        """
        Compute approximated NDCG loss.
        
        Args:
            predictions: Predicted returns
            targets: Actual returns
            groups: Optional grouping (e.g., by date)
        
        Returns:
            Loss value
        """
        # Flatten if needed
        if predictions.dim() == 2:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        # Compute soft rankings using softmax
        pred_probs = torch.softmax(predictions / self.temperature, dim=0)
        target_probs = torch.softmax(targets / self.temperature, dim=0)
        
        # Cross-entropy between probability distributions
        # This encourages predicted ranking to match actual ranking
        loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-10))
        
        return loss


class ListNetLoss(nn.Module):
    """
    ListNet loss - probability-based ranking loss.
    
    Treats ranking as a probability distribution over permutations.
    Uses top-k probabilities for efficient computation.
    """
    
    def __init__(self, temperature=1.0):
        """
        Initialize ListNet loss.
        
        Args:
            temperature: Temperature for softmax (controls sharpness)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, predictions, targets, groups=None):
        """
        Compute ListNet loss.
        
        Args:
            predictions: Predicted scores
            targets: Actual scores (returns)
            groups: Optional grouping
        
        Returns:
            Loss value
        """
        # Flatten
        if predictions.dim() == 2:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        # Convert to probability distributions
        pred_probs = torch.softmax(predictions / self.temperature, dim=0)
        target_probs = torch.softmax(targets / self.temperature, dim=0)
        
        # Cross-entropy loss
        loss = -torch.sum(target_probs * torch.log(pred_probs + 1e-10))
        
        return loss


if __name__ == "__main__":
    # Test NDCG loss
    print("Testing NDCG Loss\n")
    
    # Example: 5 stocks on one day
    predictions = torch.tensor([0.002, -0.001, 0.004, 0.001, -0.002])
    actual_returns = torch.tensor([0.003, 0.000, 0.005, 0.002, -0.001])
    
    print("Predictions:", predictions.numpy())
    print("Actual returns:", actual_returns.numpy())
    
    # Predicted ranking (by predictions)
    _, pred_rank = torch.sort(predictions, descending=True)
    print(f"Predicted ranking: {pred_rank.numpy()}")  # Which stocks to buy
    
    # Actual ranking (by actual returns)
    _, actual_rank = torch.sort(actual_returns, descending=True)
    print(f"Actual best ranking: {actual_rank.numpy()}")
    
    # Compute NDCG loss
    ndcg_loss = NDCGLoss(k=3)
    loss = ndcg_loss(predictions.unsqueeze(1), actual_returns.unsqueeze(1))
    
    print(f"\nNDCG@3 Loss: {loss.item():.4f}")
    print(f"NDCG@3 Score: {1 - loss.item():.4f}")
    
    # Test with perfect predictions
    perfect_loss = ndcg_loss(actual_returns.unsqueeze(1), actual_returns.unsqueeze(1))
    print(f"\nPerfect ranking loss: {perfect_loss.item():.4f}")
    print(f"Perfect NDCG: {1 - perfect_loss.item():.4f}")
    
    # Test ListNet
    print("\n" + "="*50)
    print("Testing ListNet Loss\n")
    listnet_loss = ListNetLoss()
    loss = listnet_loss(predictions, actual_returns)
    print(f"ListNet Loss: {loss.item():.4f}")
    
    print("\nLoss functions initialized successfully!")

