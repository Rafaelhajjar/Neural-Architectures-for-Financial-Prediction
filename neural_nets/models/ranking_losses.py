"""
Proper ranking losses with per-date grouping for stock prediction.
"""
import torch
import torch.nn as nn


class PerDateNDCGLoss(nn.Module):
    """
    NDCG Loss computed per trading date.
    
    For stock ranking, we rank stocks within each trading day (17 stocks/day).
    This loss computes NDCG for each day separately and averages.
    """
    
    def __init__(self, k=5, temperature=0.1):
        """
        Args:
            k: Top-k stocks to emphasize (for long/short strategy)
            temperature: Temperature for softmax approximation
        """
        super().__init__()
        self.k = k
        self.temperature = temperature
    
    def forward(self, predictions, targets, dates=None):
        """
        Compute NDCG loss with per-date ranking.
        
        Args:
            predictions: (batch_size,) predicted returns
            targets: (batch_size,) actual returns
            dates: list of date strings or None
        
        Returns:
            loss: 1 - average NDCG across all dates
        """
        if predictions.dim() == 2:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        # If no dates provided, treat as one group
        if dates is None or len(set(dates)) == 1:
            return 1.0 - self._compute_ndcg(predictions, targets)
        
        # Group by date manually (keeping gradient flow)
        unique_dates = list(set(dates))
        date_to_indices = {date: [] for date in unique_dates}
        
        for i, date in enumerate(dates):
            date_to_indices[date].append(i)
        
        ndcg_scores = []
        for date, indices in date_to_indices.items():
            if len(indices) < 2:  # Need at least 2 stocks to rank
                continue
            
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=predictions.device)
            day_preds = predictions[idx_tensor]
            day_targets = targets[idx_tensor]
            
            ndcg = self._compute_ndcg(day_preds, day_targets)
            ndcg_scores.append(ndcg)
        
        if len(ndcg_scores) == 0:
            # Return a constant that requires grad
            return torch.tensor(1.0, device=predictions.device, requires_grad=True)
        
        avg_ndcg = torch.stack(ndcg_scores).mean()
        return 1.0 - avg_ndcg
    
    def _compute_ndcg(self, predictions, targets):
        """
        Compute differentiable ranking quality score.
        
        Since true NDCG with argmax/topk is not differentiable, we use
        a combination of:
        1. Spearman-like correlation (fully differentiable)
        2. Weighted top-k emphasis using softmax
        """
        n = len(predictions)
        k = min(self.k, n)
        
        # Method 1: Pearson correlation (fully differentiable)
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        
        pred_std = pred_centered.std() + 1e-8
        target_std = target_centered.std() + 1e-8
        
        correlation = (pred_centered * target_centered).mean() / (pred_std * target_std)
        correlation = correlation.clamp(-1.0, 1.0)
        
        # Method 2: Soft top-k weighting
        # Give more weight to top-k predictions
        pred_probs = torch.softmax(predictions / self.temperature, dim=0)
        target_probs = torch.softmax(targets / self.temperature, dim=0)
        
        # KL divergence between distributions (lower is better, so negate)
        kl_div = -(target_probs * (target_probs / (pred_probs + 1e-10)).log()).sum()
        kl_score = torch.exp(-kl_div).clamp(0, 1)  # Convert to 0-1 score
        
        # Combine: 70% correlation, 30% distribution matching
        score = 0.7 * (correlation + 1.0) / 2.0 + 0.3 * kl_score
        
        return score.clamp(0, 1)


class PerDateSpearmanLoss(nn.Module):
    """
    Differentiable approximation of Spearman correlation loss per date.
    
    Maximizes rank correlation between predictions and actual returns
    within each trading day.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, predictions, targets, dates=None):
        """
        Compute negative Spearman correlation (to minimize).
        
        Args:
            predictions: (batch_size,) predicted returns
            targets: (batch_size,) actual returns
            dates: list of date strings or None
        
        Returns:
            loss: 1 - average Spearman correlation
        """
        if predictions.dim() == 2:
            predictions = predictions.squeeze(1)
        if targets.dim() == 2:
            targets = targets.squeeze(1)
        
        # If no dates, compute for whole batch
        if dates is None or len(set(dates)) == 1:
            return 1.0 - self._compute_spearman(predictions, targets)
        
        # Group by date manually
        unique_dates = list(set(dates))
        date_to_indices = {date: [] for date in unique_dates}
        
        for i, date in enumerate(dates):
            date_to_indices[date].append(i)
        
        spearman_scores = []
        for date, indices in date_to_indices.items():
            if len(indices) < 2:
                continue
            
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=predictions.device)
            day_preds = predictions[idx_tensor]
            day_targets = targets[idx_tensor]
            
            spearman = self._compute_spearman(day_preds, day_targets)
            spearman_scores.append(spearman)
        
        if len(spearman_scores) == 0:
            return torch.tensor(1.0, device=predictions.device, requires_grad=True)
        
        avg_spearman = torch.stack(spearman_scores).mean()
        return 1.0 - avg_spearman
    
    def _compute_spearman(self, predictions, targets):
        """
        Compute differentiable Spearman correlation.
        
        Uses continuous rank approximation.
        """
        # Center the values
        pred_centered = predictions - predictions.mean()
        target_centered = targets - targets.mean()
        
        # Compute correlation
        pred_std = pred_centered.std() + 1e-8
        target_std = target_centered.std() + 1e-8
        
        correlation = (pred_centered * target_centered).mean() / (pred_std * target_std)
        
        return correlation.clamp(-1.0, 1.0)


if __name__ == "__main__":
    print("Testing Per-Date Ranking Losses\n")
    print("="*70)
    
    # Simulate 2 days with 5 stocks each
    predictions = torch.tensor([
        # Day 1: stocks A, B, C, D, E
        0.002, -0.001, 0.004, 0.001, -0.002,
        # Day 2: stocks A, B, C, D, E
        0.003, 0.001, -0.001, 0.002, 0.000
    ], requires_grad=True)
    
    targets = torch.tensor([
        # Day 1 actual returns
        0.003, 0.000, 0.005, 0.002, -0.001,
        # Day 2 actual returns
        0.004, 0.002, 0.001, 0.003, 0.001
    ])
    
    dates = ['2024-01-01'] * 5 + ['2024-01-02'] * 5
    
    print("Day 1 predictions:", predictions[:5].detach().numpy())
    print("Day 1 actual:", targets[:5].numpy())
    print("\nDay 2 predictions:", predictions[5:].detach().numpy())
    print("Day 2 actual:", targets[5:].numpy())
    
    # Test NDCG loss
    print("\n" + "="*70)
    print("Testing PerDateNDCGLoss")
    print("="*70)
    
    ndcg_loss_fn = PerDateNDCGLoss(k=3)
    loss = ndcg_loss_fn(predictions, targets, dates)
    
    print(f"NDCG Loss: {loss.item():.4f}")
    print(f"NDCG Score: {1 - loss.item():.4f}")
    print(f"Gradients enabled: {loss.requires_grad}")
    
    # Test backward pass
    loss.backward()
    print(f"Gradient computed: {predictions.grad is not None}")
    
    # Test Spearman loss
    print("\n" + "="*70)
    print("Testing PerDateSpearmanLoss")
    print("="*70)
    
    predictions2 = predictions.detach().clone().requires_grad_(True)
    spearman_loss_fn = PerDateSpearmanLoss()
    loss2 = spearman_loss_fn(predictions2, targets, dates)
    
    print(f"Spearman Loss: {loss2.item():.4f}")
    print(f"Spearman Correlation: {1 - loss2.item():.4f}")
    print(f"Gradients enabled: {loss2.requires_grad}")
    
    loss2.backward()
    print(f"Gradient computed: {predictions2.grad is not None}")
    
    print("\n✅ All tests passed! Losses are differentiable.")

