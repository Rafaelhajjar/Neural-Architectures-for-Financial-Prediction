"""
Evaluation metrics for stock prediction models.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss
)
from scipy.stats import spearmanr, kendalltau
from typing import Dict


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC, log loss)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['log_loss'] = log_loss(y_true, y_proba)
        except:
            pass
    
    return metrics


def compute_ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute ranking quality metrics.
    
    Args:
        y_true: True returns
        y_pred: Predicted returns
    
    Returns:
        Dictionary of metrics
    """
    # Spearman correlation
    spearman, spearman_p = spearmanr(y_true, y_pred)
    
    # Kendall's tau
    kendall, kendall_p = kendalltau(y_true, y_pred)
    
    # Information Coefficient (Spearman but called IC in finance)
    ic = spearman
    
    # MSE and MAE
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    metrics = {
        'spearman': spearman,
        'spearman_pvalue': spearman_p,
        'kendall': kendall,
        'kendall_pvalue': kendall_p,
        'ic': ic,
        'mse': mse,
        'mae': mae
    }
    
    return metrics


def compute_trading_metrics(
    predictions_df: pd.DataFrame,
    k: int = 5
) -> Dict[str, float]:
    """
    Compute trading strategy metrics.
    
    Implements long/short strategy: long top-k, short bottom-k stocks.
    
    Args:
        predictions_df: DataFrame with columns [date, ticker, pred, actual_return]
        k: Number of stocks to long/short
    
    Returns:
        Dictionary of trading metrics
    """
    predictions_df = predictions_df.copy()
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    daily_returns = []
    
    for date, group in predictions_df.groupby('date'):
        if len(group) < 2 * k:
            continue
        
        # Sort by predictions
        group = group.sort_values('pred', ascending=False)
        
        # Long top-k, short bottom-k
        long_stocks = group.head(k)
        short_stocks = group.tail(k)
        
        # Portfolio return (equal weight)
        long_return = long_stocks['actual_return'].mean()
        short_return = short_stocks['actual_return'].mean()
        port_return = long_return - short_return
        
        daily_returns.append({
            'date': date,
            'port_return': port_return,
            'long_return': long_return,
            'short_return': short_return
        })
    
    returns_df = pd.DataFrame(daily_returns)
    
    # Cumulative return
    returns_df['cum_return'] = (1 + returns_df['port_return']).cumprod() - 1
    
    # Sharpe ratio (annualized)
    mean_return = returns_df['port_return'].mean()
    std_return = returns_df['port_return'].std()
    sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    
    # Max drawdown
    cum_returns = (1 + returns_df['port_return']).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns_df['port_return'] > 0).mean()
    
    metrics = {
        'sharpe_ratio': sharpe,
        'total_return': returns_df['cum_return'].iloc[-1],
        'mean_daily_return': mean_return,
        'volatility': std_return,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_days': len(returns_df)
    }
    
    return metrics, returns_df


def compute_daily_ic(
    predictions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Information Coefficient (IC) for each day.
    
    Args:
        predictions_df: DataFrame with [date, ticker, pred, actual_return]
    
    Returns:
        DataFrame with daily IC values
    """
    predictions_df = predictions_df.copy()
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])
    
    daily_ic = []
    
    for date, group in predictions_df.groupby('date'):
        if len(group) > 1:
            ic, _ = spearmanr(group['pred'], group['actual_return'])
            daily_ic.append({'date': date, 'ic': ic})
    
    ic_df = pd.DataFrame(daily_ic)
    
    return ic_df


if __name__ == "__main__":
    print("Metrics module defined successfully!")
    
    # Test classification metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9])
    
    clf_metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    print("\nClassification Metrics:")
    for k, v in clf_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # Test ranking metrics
    y_true_rank = np.array([0.01, -0.02, 0.03, 0.00, 0.02])
    y_pred_rank = np.array([0.015, -0.015, 0.025, 0.005, 0.018])
    
    rank_metrics = compute_ranking_metrics(y_true_rank, y_pred_rank)
    print("\nRanking Metrics:")
    for k, v in rank_metrics.items():
        if 'pvalue' not in k:
            print(f"  {k}: {v:.4f}")

