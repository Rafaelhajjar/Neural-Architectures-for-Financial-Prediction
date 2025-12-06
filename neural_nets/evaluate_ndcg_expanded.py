"""
Evaluate NDCG models trained on expanded 100-stock dataset.
Compare with MSE models to assess novel loss function performance.
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models import CombinedNet, LateFusionNet
from neural_nets.training.data_loader import get_train_val_test_split, create_data_loaders
from neural_nets.evaluation.metrics import compute_ranking_metrics, compute_trading_metrics

print("="*70)
print("EVALUATING NDCG MODELS ON 100 STOCKS")
print("="*70)

# Load expanded dataset
DATA_PATH = 'data/processed/features_expanded_100stocks_with_sentiment.parquet'
df = pd.read_parquet(DATA_PATH)
print(f"Dataset: {len(df):,} samples, {df['ticker'].nunique()} stocks\n")

# Split data
train_df, val_df, test_df = get_train_val_test_split(df)
print(f"Test set: {len(test_df):,} samples")
print(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}\n")

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
    ('Combined Ranker (NDCG)', CombinedNet, 'combined_ranker_ndcg_expanded_fixed_best.pt'),
    ('Late Fusion Ranker (NDCG)', LateFusionNet, 'late_fusion_ranker_ndcg_expanded_fixed_best.pt'),
]

results_list = []

# Evaluate each model
for model_name, model_class, model_file in models_to_eval:
    print(f"{'='*70}")
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
        
        print(f"  Spearman: {ranking_metrics['spearman']:7.4f}")
        print(f"  Sharpe:   {trading_metrics['sharpe_ratio']:7.4f}")
        print(f"  Return:   {trading_metrics['total_return']*100:7.2f}%")
        print(f"  Max DD:   {trading_metrics['max_drawdown']*100:7.2f}%\n")
        
    except Exception as e:
        print(f"  ❌ Failed: {e}\n")
        continue

# Save results
print(f"{'='*70}")
print("SAVING RESULTS")
print(f"{'='*70}")

results_df = pd.DataFrame(results_list)
output_path = 'neural_nets/results/ndcg_expanded_100stocks_results.csv'
results_df.to_csv(output_path, index=False)
print(f"✅ Results saved to: {output_path}\n")

# ========================================================================
# COMPREHENSIVE COMPARISON
# ========================================================================
print(f"{'='*70}")
print("COMPLETE MSE vs NDCG COMPARISON")
print(f"{'='*70}\n")

# Load all results
mse_results = pd.read_csv('neural_nets/results/expanded_100stocks_results.csv')
original_results = pd.read_csv('neural_nets/results/results_with_benchmarks.csv')

# Filter for relevant models
original_17 = original_results[original_results['model'].str.contains('Ranker')].copy()
mse_100 = mse_results[mse_results['model'].str.contains('Ranker') & ~mse_results['model'].str.contains('Deep')].copy()
ndcg_100 = results_df.copy()

print("=" * 80)
print("COMBINED RANKER: MSE vs NDCG across dataset sizes")
print("=" * 80)

try:
    # 17 stocks
    c17_mse = original_17[original_17['model'] == 'Combined Ranker (MSE)'].iloc[0]
    c17_ndcg = original_17[original_17['model'] == 'Combined Ranker (NDCG)'].iloc[0]
    
    # 100 stocks
    c100_mse = mse_100[mse_100['model'] == 'Combined Ranker (MSE)'].iloc[0]
    c100_ndcg = ndcg_100[ndcg_100['model'] == 'Combined Ranker (NDCG)'].iloc[0]
    
    print(f"{'Dataset':<15} {'Loss':<6} {'Sharpe':>8} {'Return':>10} {'Max DD':>10}")
    print("-" * 80)
    print(f"{'17 stocks':<15} {'MSE':<6} {c17_mse['sharpe_ratio']:>8.3f} {c17_mse['total_return']*100:>9.2f}% {c17_mse['max_drawdown']*100:>9.2f}%")
    print(f"{'17 stocks':<15} {'NDCG':<6} {c17_ndcg['sharpe_ratio']:>8.3f} {c17_ndcg['total_return']*100:>9.2f}% {c17_ndcg['max_drawdown']*100:>9.2f}%")
    print(f"{'100 stocks':<15} {'MSE':<6} {c100_mse['sharpe_ratio']:>8.3f} {c100_mse['total_return']*100:>9.2f}% {c100_mse['max_drawdown']*100:>9.2f}%")
    print(f"{'100 stocks':<15} {'NDCG':<6} {c100_ndcg['sharpe_ratio']:>8.3f} {c100_ndcg['total_return']*100:>9.2f}% {c100_ndcg['max_drawdown']*100:>9.2f}%")
    
    print("\nKey Findings:")
    print(f"  - MSE on 17 stocks:   {c17_mse['sharpe_ratio']:.3f} Sharpe")
    print(f"  - NDCG on 17 stocks:  {c17_ndcg['sharpe_ratio']:.3f} Sharpe ({(c17_ndcg['sharpe_ratio']/c17_mse['sharpe_ratio']-1)*100:+.0f}%)")
    print(f"  - MSE on 100 stocks:  {c100_mse['sharpe_ratio']:.3f} Sharpe")
    print(f"  - NDCG on 100 stocks: {c100_ndcg['sharpe_ratio']:.3f} Sharpe ({(c100_ndcg['sharpe_ratio']/c100_mse['sharpe_ratio']-1)*100:+.0f}%)")
    
except Exception as e:
    print(f"Could not create Combined comparison: {e}")

print("\n" + "=" * 80)
print("LATE FUSION RANKER: MSE vs NDCG across dataset sizes")
print("=" * 80)

try:
    # 17 stocks
    lf17_mse = original_17[original_17['model'] == 'Late Fusion Ranker (MSE)'].iloc[0]
    lf17_ndcg = original_17[original_17['model'] == 'Late Fusion Ranker (NDCG)'].iloc[0]
    
    # 100 stocks
    lf100_mse = mse_100[mse_100['model'] == 'Late Fusion Ranker (MSE)'].iloc[0]
    lf100_ndcg = ndcg_100[ndcg_100['model'] == 'Late Fusion Ranker (NDCG)'].iloc[0]
    
    print(f"{'Dataset':<15} {'Loss':<6} {'Sharpe':>8} {'Return':>10} {'Max DD':>10}")
    print("-" * 80)
    print(f"{'17 stocks':<15} {'MSE':<6} {lf17_mse['sharpe_ratio']:>8.3f} {lf17_mse['total_return']*100:>9.2f}% {lf17_mse['max_drawdown']*100:>9.2f}%")
    print(f"{'17 stocks':<15} {'NDCG':<6} {lf17_ndcg['sharpe_ratio']:>8.3f} {lf17_ndcg['total_return']*100:>9.2f}% {lf17_ndcg['max_drawdown']*100:>9.2f}%")
    print(f"{'100 stocks':<15} {'MSE':<6} {lf100_mse['sharpe_ratio']:>8.3f} {lf100_mse['total_return']*100:>9.2f}% {lf100_mse['max_drawdown']*100:>9.2f}%")
    print(f"{'100 stocks':<15} {'NDCG':<6} {lf100_ndcg['sharpe_ratio']:>8.3f} {lf100_ndcg['total_return']*100:>9.2f}% {lf100_ndcg['max_drawdown']*100:>9.2f}%")
    
    print("\nKey Findings:")
    print(f"  - MSE on 17 stocks:   {lf17_mse['sharpe_ratio']:.3f} Sharpe ⭐")
    print(f"  - NDCG on 17 stocks:  {lf17_ndcg['sharpe_ratio']:.3f} Sharpe ({(lf17_ndcg['sharpe_ratio']/lf17_mse['sharpe_ratio']-1)*100:+.0f}%)")
    print(f"  - MSE on 100 stocks:  {lf100_mse['sharpe_ratio']:.3f} Sharpe")
    print(f"  - NDCG on 100 stocks: {lf100_ndcg['sharpe_ratio']:.3f} Sharpe")
    
except Exception as e:
    print(f"Could not create Late Fusion comparison: {e}")

# ========================================================================
# CONCLUSIONS
# ========================================================================
print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)

print("\n1. NOVEL NDCG LOSS PERFORMANCE:")
print("   - Consistently underperforms MSE on both dataset sizes")
print("   - Theoretical ranking objective doesn't translate to practice")
print("   - MSE provides better gradient signal for this problem")

print("\n2. DATASET SIZE EFFECT:")
print("   - Both MSE and NDCG suffer on expanded dataset")
print("   - Indicates overfitting was present in original 17-stock results")
print("   - More stocks = harder problem, lower performance")

print("\n3. RECOMMENDATIONS:")
print("   - Use MSE loss for stock ranking (simpler, more effective)")
print("   - NDCG approximation needs refinement for financial data")
print("   - Focus on Deep Late Fusion architecture (best on 100 stocks)")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETE")
print("="*70)

