"""
Evaluation script for controlled early vs late fusion comparison.

Evaluates both models on the test set and computes comprehensive metrics.
"""
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models.controlled_fusion import EarlyFusion100K, LateFusion100K
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model and return predictions."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_price = batch['X_price'].to(device)
            x_sentiment = batch['X_sentiment'].to(device)
            y = batch['y'].to(device)
            
            # Predict
            y_pred = model(x_price, x_sentiment)
            
            predictions.extend(y_pred.cpu().numpy().flatten())
            actuals.extend(y.cpu().numpy().flatten())
    
    return np.array(predictions), np.array(actuals)


def compute_metrics(predictions, actuals, dates, tickers):
    """Compute comprehensive trading metrics."""
    
    # Create dataframe
    df = pd.DataFrame({
        'date': dates,
        'ticker': tickers,
        'prediction': predictions,
        'actual': actuals
    })
    
    # Correlation metrics
    spearman, spearman_p = stats.spearmanr(predictions, actuals)
    kendall, kendall_p = stats.kendalltau(predictions, actuals)
    pearson = np.corrcoef(predictions, actuals)[0, 1]
    
    # Error metrics
    mse = np.mean((predictions - actuals) ** 2)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(mse)
    
    # Long-short strategy (top 5 vs bottom 5 each day)
    daily_returns = []
    
    for date in df['date'].unique():
        day_df = df[df['date'] == date].copy()
        
        if len(day_df) < 10:  # Need at least 10 stocks
            continue
        
        # Sort by predictions
        day_df = day_df.sort_values('prediction', ascending=False)
        
        # Long top 5, short bottom 5
        long_return = day_df.head(5)['actual'].mean()
        short_return = day_df.tail(5)['actual'].mean()
        
        daily_return = long_return - short_return
        daily_returns.append(daily_return)
    
    daily_returns = np.array(daily_returns)
    
    # Trading metrics
    mean_daily_return = np.mean(daily_returns)
    volatility = np.std(daily_returns)
    sharpe_ratio = (mean_daily_return / volatility) * np.sqrt(252) if volatility > 0 else 0
    
    total_return = np.sum(daily_returns)
    cumulative_returns = np.cumsum(daily_returns)
    max_drawdown = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))
    
    win_rate = np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
    
    metrics = {
        # Correlation
        'spearman': spearman,
        'spearman_pvalue': spearman_p,
        'kendall': kendall,
        'kendall_pvalue': kendall_p,
        'pearson': pearson,
        
        # Error
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        
        # Trading
        'sharpe_ratio': sharpe_ratio,
        'total_return': total_return,
        'mean_daily_return': mean_daily_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'num_days': len(daily_returns)
    }
    
    return metrics, daily_returns


def main():
    """Evaluate both models."""
    print("="*80)
    print("CONTROLLED FUSION EVALUATION")
    print("="*80)
    print()
    
    device = 'cpu'
    
    # Load data
    print("Loading test data...")
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_expanded_100stocks_with_sentiment.parquet'
    
    train_df, val_df, test_df = load_and_prepare_data(
        data_path=str(data_path),
        train_end='2013-12-31',
        val_end='2015-06-30'
    )
    
    print(f"Test samples: {len(test_df):,}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    print()
    
    # Features
    price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
    sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']
    
    # Create test loader
    loaders = create_data_loaders(
        train_df, val_df, test_df,
        price_features, sentiment_features,
        task='regression',
        batch_size=256
    )
    test_loader = loaders['test']
    
    results = {}
    
    # ========================================
    # EVALUATE EARLY FUSION
    # ========================================
    print("="*80)
    print("EVALUATING: EARLY FUSION")
    print("="*80)
    
    # Load model
    early_model = EarlyFusion100K(task='regression').to(device)
    checkpoint = torch.load(
        Path(__file__).parent / 'trained_models' / 'early_fusion_100k_best.pt',
        map_location=device
    )
    early_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Parameters: {checkpoint['parameters']:,}")
    print(f"Best val loss: {checkpoint['train_history']['best_val_loss']:.6f}")
    print()
    
    # Evaluate
    print("Computing predictions...")
    early_preds, actuals = evaluate_model(early_model, test_loader, device)
    
    # Get dates and tickers from test_df
    dates = test_df['date'].values
    tickers = test_df['ticker'].values
    
    print("Computing metrics...")
    early_metrics, early_daily_returns = compute_metrics(early_preds, actuals, dates, tickers)
    
    print(f"\n📊 EARLY FUSION RESULTS:")
    print(f"  Spearman:     {early_metrics['spearman']:.4f} (p={early_metrics['spearman_pvalue']:.4e})")
    print(f"  Kendall:      {early_metrics['kendall']:.4f}")
    print(f"  Pearson:      {early_metrics['pearson']:.4f}")
    print(f"  MSE:          {early_metrics['mse']:.6f}")
    print(f"  MAE:          {early_metrics['mae']:.6f}")
    print(f"  Sharpe Ratio: {early_metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return: {early_metrics['total_return']:.4f} ({early_metrics['total_return']*100:.2f}%)")
    print(f"  Max Drawdown: {early_metrics['max_drawdown']:.4f} ({early_metrics['max_drawdown']*100:.2f}%)")
    print(f"  Win Rate:     {early_metrics['win_rate']:.4f} ({early_metrics['win_rate']*100:.1f}%)")
    print()
    
    results['early'] = {
        'predictions': early_preds,
        'metrics': early_metrics,
        'daily_returns': early_daily_returns
    }
    
    # ========================================
    # EVALUATE LATE FUSION
    # ========================================
    print("="*80)
    print("EVALUATING: LATE FUSION")
    print("="*80)
    
    # Load model
    late_model = LateFusion100K(task='regression').to(device)
    checkpoint = torch.load(
        Path(__file__).parent / 'trained_models' / 'late_fusion_100k_best.pt',
        map_location=device
    )
    late_model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Parameters: {checkpoint['parameters']:,}")
    print(f"Best val loss: {checkpoint['train_history']['best_val_loss']:.6f}")
    print()
    
    # Evaluate
    print("Computing predictions...")
    late_preds, actuals = evaluate_model(late_model, test_loader, device)
    
    print("Computing metrics...")
    late_metrics, late_daily_returns = compute_metrics(late_preds, actuals, dates, tickers)
    
    print(f"\n📊 LATE FUSION RESULTS:")
    print(f"  Spearman:     {late_metrics['spearman']:.4f} (p={late_metrics['spearman_pvalue']:.4e})")
    print(f"  Kendall:      {late_metrics['kendall']:.4f}")
    print(f"  Pearson:      {late_metrics['pearson']:.4f}")
    print(f"  MSE:          {late_metrics['mse']:.6f}")
    print(f"  MAE:          {late_metrics['mae']:.6f}")
    print(f"  Sharpe Ratio: {late_metrics['sharpe_ratio']:.4f}")
    print(f"  Total Return: {late_metrics['total_return']:.4f} ({late_metrics['total_return']*100:.2f}%)")
    print(f"  Max Drawdown: {late_metrics['max_drawdown']:.4f} ({late_metrics['max_drawdown']*100:.2f}%)")
    print(f"  Win Rate:     {late_metrics['win_rate']:.4f} ({late_metrics['win_rate']*100:.1f}%)")
    print()
    
    results['late'] = {
        'predictions': late_preds,
        'metrics': late_metrics,
        'daily_returns': late_daily_returns
    }
    
    # ========================================
    # COMPARISON
    # ========================================
    print("="*80)
    print("CONTROLLED COMPARISON RESULTS")
    print("="*80)
    
    print("\nKey Metrics Comparison:")
    print(f"{'Metric':<20} {'Early Fusion':<15} {'Late Fusion':<15} {'Winner':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('Sharpe Ratio', 'sharpe_ratio', 'higher'),
        ('Total Return', 'total_return', 'higher'),
        ('Spearman', 'spearman', 'higher'),
        ('MSE', 'mse', 'lower'),
        ('Max Drawdown', 'max_drawdown', 'higher'),
        ('Win Rate', 'win_rate', 'higher'),
    ]
    
    early_wins = 0
    late_wins = 0
    
    for name, key, direction in metrics_to_compare:
        early_val = early_metrics[key]
        late_val = late_metrics[key]
        
        if direction == 'higher':
            winner = 'Early' if early_val > late_val else 'Late' if late_val > early_val else 'Tie'
            if winner == 'Early':
                early_wins += 1
            elif winner == 'Late':
                late_wins += 1
        else:  # lower is better
            winner = 'Early' if early_val < late_val else 'Late' if late_val < early_val else 'Tie'
            if winner == 'Early':
                early_wins += 1
            elif winner == 'Late':
                late_wins += 1
        
        print(f"{name:<20} {early_val:>14.4f} {late_val:>14.4f} {winner:>14}")
    
    print("-" * 70)
    print(f"{'Score':<20} {early_wins:>14} {late_wins:>14}")
    print()
    
    if early_wins > late_wins:
        print(f"🏆 WINNER: Early Fusion ({early_wins} vs {late_wins})")
    elif late_wins > early_wins:
        print(f"🏆 WINNER: Late Fusion ({late_wins} vs {early_wins})")
    else:
        print(f"🤝 TIE ({early_wins} - {late_wins})")
    
    # Save results
    results_df = pd.DataFrame({
        'model': ['Early Fusion', 'Late Fusion'],
        **{key: [early_metrics[key], late_metrics[key]] for key in early_metrics.keys()}
    })
    
    results_path = Path(__file__).parent / 'results' / 'controlled_fusion_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'date': dates,
        'ticker': tickers,
        'actual': actuals,
        'early_fusion_pred': early_preds,
        'late_fusion_pred': late_preds
    })
    
    preds_path = Path(__file__).parent / 'results' / 'controlled_fusion_predictions.csv'
    predictions_df.to_csv(preds_path, index=False)
    print(f"💾 Predictions saved to: {preds_path}")
    
    print("\n" + "="*80)
    print("✅ Evaluation complete!")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()

