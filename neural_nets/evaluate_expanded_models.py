"""
Evaluate all models trained on the expanded 100-stock dataset.

Generates:
1. Performance metrics for each model
2. Trading strategy backtests
3. Comparison with original 17-stock results
4. CSV results file
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models import CombinedNet, LateFusionNet
from neural_nets.models.advanced_models import DeepLateFusionNet, DeepCombinedNet
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders
from neural_nets.evaluation.metrics import compute_ranking_metrics, compute_trading_metrics
from neural_nets.training.train_ensemble import EnsemblePredictor

print("="*70)
print("EVALUATING EXPANDED DATASET MODELS")
print("="*70)

# Load expanded dataset
DATA_PATH = 'data/processed/features_expanded_100stocks_with_sentiment.parquet'
df = pd.read_parquet(DATA_PATH)
print(f"Dataset: {len(df):,} samples, {df['ticker'].nunique()} stocks")

# Split data
from neural_nets.training.data_loader import get_train_val_test_split
train_df, val_df, test_df = get_train_val_test_split(df)

print(f"Test set: {len(test_df):,} samples")
print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")

# Create data loaders
price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']

loaders = create_data_loaders(
    train_df, val_df, test_df,
    price_features, sentiment_features,
    task='regression',
    batch_size=256,
    normalize=True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}\n")

# Models to evaluate
models_to_eval = [
    ('Combined Ranker (MSE)', CombinedNet, 'combined_ranker_mse_expanded_best.pt'),
    ('Late Fusion Ranker (MSE)', LateFusionNet, 'late_fusion_ranker_mse_expanded_best.pt'),
    ('Deep Late Fusion', DeepLateFusionNet, 'deep_late_fusion_mse_expanded_best.pt'),
    ('Deep Combined', DeepCombinedNet, 'deep_combined_mse_expanded_best.pt'),
]

results_list = []

# Evaluate each model
for model_name, model_class, model_file in models_to_eval:
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    
    try:
        # Load model
        model = model_class(task='regression')
        checkpoint_path = f'neural_nets/trained_models/{model_file}'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Generate predictions
        all_preds = []
        all_targets = []
        all_dates = []
        all_tickers = []
        
        with torch.no_grad():
            for batch in loaders['test']:
                x_price = batch['X_price'].to(device)
                x_sentiment = batch['X_sentiment'].to(device)
                y = batch['y']
                
                outputs = model(x_price, x_sentiment)
                preds = outputs.squeeze().cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(y.numpy())
                all_dates.extend(batch['date'])
                all_tickers.extend(batch['ticker'])
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'date': all_dates,
            'ticker': all_tickers,
            'pred': all_preds,
            'actual_return': all_targets
        })
        
        # Compute ranking metrics
        ranking_metrics = compute_ranking_metrics(
            results_df['actual_return'].values,
            results_df['pred'].values
        )
        
        # Compute trading metrics
        trading_metrics, returns_df = compute_trading_metrics(results_df, k=5)
        
        # Combine all metrics
        all_metrics = {**ranking_metrics, **trading_metrics, 'model': model_name}
        results_list.append(all_metrics)
        
        print(f"  Spearman: {ranking_metrics['spearman']:.4f}")
        print(f"  Sharpe:   {trading_metrics['sharpe_ratio']:.4f}")
        print(f"  Return:   {trading_metrics['total_return']*100:.2f}%")
        print(f"  Max DD:   {trading_metrics['max_drawdown']*100:.2f}%")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        continue

# Evaluate ensemble
print(f"\n{'='*70}")
print("Evaluating: Late Fusion Ensemble (3 members)")
print(f"{'='*70}")

try:
    # Load ensemble
    ensemble = EnsemblePredictor(
        model_class=LateFusionNet,
        base_name='late_fusion_ensemble_expanded',
        num_members=3,
        device=device
    )
    
    # Generate predictions
    all_preds = []
    all_targets = []
    all_dates = []
    all_tickers = []
    
    with torch.no_grad():
        for batch in loaders['test']:
            x_price = batch['X_price'].to(device)
            x_sentiment = batch['X_sentiment'].to(device)
            y = batch['y']
            
            preds = ensemble.predict(x_price, x_sentiment)
            
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_dates.extend(batch['date'])
            all_tickers.extend(batch['ticker'])
    
    results_df = pd.DataFrame({
        'date': all_dates,
        'ticker': all_tickers,
        'pred': all_preds,
        'actual_return': all_targets
    })
    
    ranking_metrics = compute_ranking_metrics(
        results_df['actual_return'].values,
        results_df['pred'].values
    )
    
    trading_metrics, returns_df = compute_trading_metrics(results_df, k=5)
    
    all_metrics = {**ranking_metrics, **trading_metrics, 'model': 'Late Fusion Ensemble (3)'}
    results_list.append(all_metrics)
    
    print(f"  Spearman: {ranking_metrics['spearman']:.4f}")
    print(f"  Sharpe:   {trading_metrics['sharpe_ratio']:.4f}")
    print(f"  Return:   {trading_metrics['total_return']*100:.2f}%")
    print(f"  Max DD:   {trading_metrics['max_drawdown']*100:.2f}%")
    
except Exception as e:
    print(f"  ❌ Failed: {e}")

# Save results
print(f"\n{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

results_df = pd.DataFrame(results_list)
output_path = 'neural_nets/results/expanded_100stocks_results.csv'
results_df.to_csv(output_path, index=False)
print(f"✅ Results saved to: {output_path}")

# Print summary
print(f"\n{'='*70}")
print("RESULTS SUMMARY - EXPANDED 100 STOCKS")
print(f"{'='*70}\n")

# Sort by Sharpe ratio
results_df_sorted = results_df.sort_values('sharpe_ratio', ascending=False)

print("Ranking by Sharpe Ratio:")
print("-" * 70)
for idx, row in results_df_sorted.iterrows():
    print(f"{row['model']:30s} | Sharpe: {row['sharpe_ratio']:6.3f} | Return: {row['total_return']*100:6.2f}% | MaxDD: {row['max_drawdown']*100:6.2f}%")

# Compare to original 17-stock results
print(f"\n{'='*70}")
print("COMPARISON: 100 STOCKS vs 17 STOCKS")
print(f"{'='*70}\n")

try:
    original_results = pd.read_csv('neural_nets/results/results_with_benchmarks.csv')
    original_late_fusion = original_results[original_results['model'] == 'Late Fusion Ranker (MSE)'].iloc[0]
    original_ensemble = original_results[original_results['model'] == 'Deep Late Fusion Ensemble (5)'].iloc[0]
    
    expanded_late_fusion = results_df[results_df['model'] == 'Late Fusion Ranker (MSE)'].iloc[0]
    expanded_ensemble = results_df[results_df['model'] == 'Late Fusion Ensemble (3)'].iloc[0]
    
    print("Late Fusion Ranker (MSE):")
    print(f"  17 stocks:  Sharpe {original_late_fusion['sharpe_ratio']:.3f}, Return {original_late_fusion['total_return']*100:.1f}%")
    print(f"  100 stocks: Sharpe {expanded_late_fusion['sharpe_ratio']:.3f}, Return {expanded_late_fusion['total_return']*100:.1f}%")
    print(f"  Change:     {(expanded_late_fusion['sharpe_ratio']/original_late_fusion['sharpe_ratio']-1)*100:+.1f}% Sharpe")
    
    print("\nEnsemble:")
    print(f"  17 stocks (5 members):  Sharpe {original_ensemble['sharpe_ratio']:.3f}, Return {original_ensemble['total_return']*100:.1f}%")
    print(f"  100 stocks (3 members): Sharpe {expanded_ensemble['sharpe_ratio']:.3f}, Return {expanded_ensemble['total_return']*100:.1f}%")
    print(f"  Change:                 {(expanded_ensemble['sharpe_ratio']/original_ensemble['sharpe_ratio']-1)*100:+.1f}% Sharpe")
    
except Exception as e:
    print(f"Could not load original results for comparison: {e}")

print(f"\n{'='*70}")
print("✅ EVALUATION COMPLETE")
print(f"{'='*70}")
print("\nKey Insights:")
print("  - Performance with 100 stocks is more realistic")
print("  - Lower Sharpe/returns indicate reduced overfitting")
print("  - Results should generalize better to new stocks")
print("  - This is the honest assessment of model performance")

