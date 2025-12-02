"""
Model evaluator for comprehensive performance assessment.
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from neural_nets.evaluation.metrics import (
    compute_classification_metrics,
    compute_ranking_metrics,
    compute_trading_metrics,
    compute_daily_ic
)


class Evaluator:
    """Evaluator for neural network models."""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()
    
    def predict(self) -> pd.DataFrame:
        """
        Generate predictions on test set.
        
        Returns:
            DataFrame with predictions and metadata
        """
        all_preds = []
        all_targets = []
        all_dates = []
        all_tickers = []
        
        with torch.no_grad():
            for batch in self.test_loader:
                x_price = batch['X_price'].to(self.device)
                x_sentiment = batch['X_sentiment'].to(self.device)
                y = batch['y']
                
                # Forward pass
                outputs = self.model(x_price, x_sentiment)
                
                # Store predictions
                if self.model.task == 'classification':
                    proba = torch.softmax(outputs, dim=1)[:, 1]  # Probability of class 1
                    preds = proba.cpu().numpy()
                else:  # regression
                    preds = outputs.squeeze().cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(y.numpy())
                all_dates.extend(batch['date'])
                all_tickers.extend(batch['ticker'])
        
        # Create DataFrame
        results_df = pd.DataFrame({
            'date': all_dates,
            'ticker': all_tickers,
            'pred': all_preds,
            'actual': all_targets
        })
        
        return results_df
    
    def evaluate_classification(self, results_df: pd.DataFrame) -> Dict:
        """
        Evaluate classification model.
        
        Args:
            results_df: DataFrame with predictions
        
        Returns:
            Dictionary of metrics
        """
        y_true = results_df['actual'].values
        y_proba = results_df['pred'].values
        y_pred = (y_proba >= 0.5).astype(int)
        
        metrics = compute_classification_metrics(y_true, y_pred, y_proba)
        
        return metrics
    
    def evaluate_ranking(self, results_df: pd.DataFrame) -> Dict:
        """
        Evaluate ranking model.
        
        Args:
            results_df: DataFrame with predictions
        
        Returns:
            Dictionary of metrics
        """
        # For ranking, we need actual returns, not binary targets
        # Assume the test_loader has future_return in the dataset
        y_true = []
        y_pred = []
        
        # Reconstruct from test dataset
        for batch in self.test_loader:
            # Get actual future returns from dataset
            if hasattr(self.test_loader.dataset, 'df'):
                batch_indices = range(len(y_true), len(y_true) + len(batch['y']))
                # This is simplified - in practice, need proper indexing
        
        # Simplified: use the prediction as is
        ranking_metrics = compute_ranking_metrics(
            results_df['actual'].values,
            results_df['pred'].values
        )
        
        # Compute trading metrics
        # Need to add actual returns column
        # For now, use 'actual' as a proxy
        results_for_trading = results_df.copy()
        results_for_trading['actual_return'] = results_df['actual']
        
        trading_metrics, returns_df = compute_trading_metrics(results_for_trading, k=5)
        
        # Combine all metrics
        all_metrics = {**ranking_metrics, **trading_metrics}
        
        return all_metrics, returns_df
    
    def save_predictions(self, results_df: pd.DataFrame, filename: str):
        """Save predictions to CSV."""
        output_path = Path('neural_nets/results') / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        print(f"Saved predictions to {output_path}")


def evaluate_all_models(
    models_info: List[Dict],
    test_loader,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    Evaluate multiple models and compare.
    
    Args:
        models_info: List of dicts with 'model', 'name', 'type' keys
        test_loader: Test data loader
        device: Device to run on
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for info in models_info:
        model = info['model']
        name = info['name']
        model_type = info['type']  # 'classification' or 'ranking'
        
        print(f"\nEvaluating {name}...")
        
        evaluator = Evaluator(model, test_loader, device)
        predictions = evaluator.predict()
        
        if model_type == 'classification':
            metrics = evaluator.evaluate_classification(predictions)
        else:  # ranking
            metrics, _ = evaluator.evaluate_ranking(predictions)
        
        # Add model name to metrics
        metrics['model'] = name
        results.append(metrics)
        
        # Save predictions
        evaluator.save_predictions(predictions, f"{name}_predictions.csv")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns to put model name first
    cols = ['model'] + [c for c in comparison_df.columns if c != 'model']
    comparison_df = comparison_df[cols]
    
    return comparison_df


if __name__ == "__main__":
    print("Evaluator module defined successfully!")

