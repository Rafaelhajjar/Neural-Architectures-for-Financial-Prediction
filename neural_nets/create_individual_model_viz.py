"""
Create detailed visualizations for individual models.
Each model gets its own folder with comprehensive analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import sys

sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models import CombinedNet, LateFusionNet
from neural_nets.models.advanced_models import DeepLateFusionNet, DeepCombinedNet
from neural_nets.training.data_loader import get_train_val_test_split, create_data_loaders

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

print("="*70)
print("CREATING INDIVIDUAL MODEL VISUALIZATIONS")
print("="*70)

# Load expanded dataset
DATA_PATH = 'data/processed/features_expanded_100stocks_with_sentiment.parquet'
df = pd.read_parquet(DATA_PATH)
train_df, val_df, test_df = get_train_val_test_split(df)

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

device = 'cpu'

# Models to visualize
models_config = [
    {
        'name': 'Combined Ranker (NDCG)',
        'folder': 'combined_ranker_ndcg',
        'model_class': CombinedNet,
        'checkpoint': 'neural_nets/trained_models/combined_ranker_ndcg_expanded_fixed_best.pt',
        'color': 'coral',
        'description': 'Best NDCG Model - Simple architecture with novel ranking loss'
    },
    {
        'name': 'Deep Late Fusion',
        'folder': 'deep_late_fusion',
        'model_class': DeepLateFusionNet,
        'checkpoint': 'neural_nets/trained_models/deep_late_fusion_mse_expanded_best.pt',
        'color': 'steelblue',
        'description': 'Best MSE Model - Deep 6-layer network with batch normalization'
    },
    {
        'name': 'Deep Combined',
        'folder': 'deep_combined',
        'model_class': DeepCombinedNet,
        'checkpoint': 'neural_nets/trained_models/deep_combined_mse_expanded_best.pt',
        'color': 'forestgreen',
        'description': 'Deep early fusion model with 95K parameters'
    }
]

def generate_predictions(model, test_loader, device):
    """Generate predictions for test set."""
    model.eval()
    all_preds = []
    all_targets = []
    all_dates = []
    all_tickers = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_price = batch['X_price'].to(device)
            x_sentiment = batch['X_sentiment'].to(device)
            y = batch['y']
            
            outputs = model(x_price, x_sentiment)
            preds = outputs.squeeze().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(y.numpy())
            all_dates.extend(batch['date'])
            all_tickers.extend(batch['ticker'])
    
    return pd.DataFrame({
        'date': all_dates,
        'ticker': all_tickers,
        'prediction': all_preds,
        'actual_return': all_targets
    })

def compute_strategy_returns(predictions_df, k=5):
    """Compute long-short strategy returns."""
    daily_returns = []
    
    for date in predictions_df['date'].unique():
        day_data = predictions_df[predictions_df['date'] == date].copy()
        
        if len(day_data) < k * 2:
            continue
        
        # Rank by prediction
        day_data = day_data.sort_values('prediction', ascending=False)
        
        # Long top k, short bottom k
        long_stocks = day_data.head(k)
        short_stocks = day_data.tail(k)
        
        # Equal-weighted returns
        long_return = long_stocks['actual_return'].mean()
        short_return = short_stocks['actual_return'].mean()
        daily_return = long_return - short_return
        
        daily_returns.append({
            'date': date,
            'return': daily_return,
            'long_return': long_return,
            'short_return': short_return
        })
    
    return pd.DataFrame(daily_returns)

def create_model_visualizations(config):
    """Create comprehensive visualizations for a single model."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {config['name']}")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir = Path(f"neural_nets/{config['folder']}_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = config['model_class'](task='regression')
    checkpoint = torch.load(config['checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Generate predictions
    print("Generating predictions...")
    predictions_df = generate_predictions(model, loaders['test'], device)
    
    # Compute strategy returns
    print("Computing strategy returns...")
    returns_df = compute_strategy_returns(predictions_df, k=5)
    returns_df['cumulative'] = (1 + returns_df['return']).cumprod()
    returns_df['drawdown'] = returns_df['cumulative'] / returns_df['cumulative'].cummax() - 1
    
    # Compute metrics
    sharpe = returns_df['return'].mean() / returns_df['return'].std() * np.sqrt(252)
    total_return = returns_df['cumulative'].iloc[-1] - 1
    max_dd = returns_df['drawdown'].min()
    volatility = returns_df['return'].std() * np.sqrt(252)
    
    color = config['color']
    
    # ========================================================================
    # PLOT 1: Equity Curve
    # ========================================================================
    print("  Creating equity curve...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(returns_df['date'], returns_df['cumulative'], 
            linewidth=2.5, color=color, label='Strategy')
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
    
    ax.fill_between(returns_df['date'], 1, returns_df['cumulative'], 
                     where=(returns_df['cumulative'] >= 1), alpha=0.3, color='green')
    ax.fill_between(returns_df['date'], 1, returns_df['cumulative'], 
                     where=(returns_df['cumulative'] < 1), alpha=0.3, color='red')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=12, fontweight='bold')
    ax.set_title(f'{config["name"]}: Equity Curve\n{config["description"]}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    
    # Add performance text
    perf_text = f'Sharpe: {sharpe:.2f} | Return: {total_return*100:.1f}% | Max DD: {max_dd*100:.1f}%'
    ax.text(0.5, 0.95, perf_text, transform=ax.transAxes, 
            ha='center', va='top', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "1_equity_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 2: Drawdown Analysis
    # ========================================================================
    print("  Creating drawdown analysis...")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Equity curve
    ax = axes[0]
    ax.plot(returns_df['date'], returns_df['cumulative'], 
            linewidth=2, color=color, label='Cumulative Return')
    ax.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Drawdown
    ax = axes[1]
    ax.fill_between(returns_df['date'], 0, returns_df['drawdown']*100, 
                     color='red', alpha=0.5, label='Drawdown')
    ax.plot(returns_df['date'], returns_df['drawdown']*100, 
            color='darkred', linewidth=1.5)
    ax.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    fig.suptitle(f'{config["name"]}: Drawdown Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "2_drawdown_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 3: Returns Distribution
    # ========================================================================
    print("  Creating returns distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax = axes[0]
    daily_returns_pct = returns_df['return'] * 100
    ax.hist(daily_returns_pct, bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(x=daily_returns_pct.mean(), color='green', linestyle='--', 
               linewidth=2, label=f'Mean: {daily_returns_pct.mean():.2f}%')
    ax.set_xlabel('Daily Return (%)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Box plot
    ax = axes[1]
    bp = ax.boxplot([daily_returns_pct], vert=True, patch_artist=True, widths=0.5)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Daily Return (%)', fontsize=11, fontweight='bold')
    ax.set_xticklabels(['Strategy'])
    ax.set_title('Returns Box Plot', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add statistics
    stats_text = f"""Mean: {daily_returns_pct.mean():.3f}%
Median: {daily_returns_pct.median():.3f}%
Std: {daily_returns_pct.std():.3f}%
Skew: {daily_returns_pct.skew():.3f}
Kurt: {daily_returns_pct.kurtosis():.3f}"""
    ax.text(1.35, 0.5, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    fig.suptitle(f'{config["name"]}: Returns Distribution Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "3_returns_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 4: Long vs Short Performance
    # ========================================================================
    print("  Creating long/short analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cumulative long returns
    ax = axes[0, 0]
    long_cum = (1 + returns_df['long_return']).cumprod()
    ax.plot(returns_df['date'], long_cum, color='green', linewidth=2)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Cumulative Return', fontsize=10, fontweight='bold')
    ax.set_title('Long Portfolio (Top 5)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Cumulative short returns
    ax = axes[0, 1]
    short_cum = (1 + returns_df['short_return']).cumprod()
    ax.plot(returns_df['date'], short_cum, color='red', linewidth=2)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylabel('Cumulative Return', fontsize=10, fontweight='bold')
    ax.set_title('Short Portfolio (Bottom 5)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Long returns distribution
    ax = axes[1, 0]
    ax.hist(returns_df['long_return']*100, bins=40, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=returns_df['long_return'].mean()*100, color='darkgreen', 
               linestyle='--', linewidth=2)
    ax.set_xlabel('Daily Return (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'Long Returns (Mean: {returns_df["long_return"].mean()*100:.2f}%)', 
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Short returns distribution
    ax = axes[1, 1]
    ax.hist(returns_df['short_return']*100, bins=40, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=returns_df['short_return'].mean()*100, color='darkred', 
               linestyle='--', linewidth=2)
    ax.set_xlabel('Daily Return (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'Short Returns (Mean: {returns_df["short_return"].mean()*100:.2f}%)', 
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    fig.suptitle(f'{config["name"]}: Long/Short Portfolio Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "4_long_short_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 5: Prediction Quality
    # ========================================================================
    print("  Creating prediction quality analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Scatter: Prediction vs Actual
    ax = axes[0, 0]
    sample = predictions_df.sample(min(5000, len(predictions_df)))
    ax.scatter(sample['prediction'], sample['actual_return'], 
               alpha=0.3, s=10, color=color)
    # Add trend line
    z = np.polyfit(predictions_df['prediction'], predictions_df['actual_return'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(predictions_df['prediction'].min(), 
                         predictions_df['prediction'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    ax.set_xlabel('Predicted Return', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual Return', fontsize=10, fontweight='bold')
    ax.set_title('Prediction vs Actual', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Prediction distribution
    ax = axes[0, 1]
    ax.hist(predictions_df['prediction'], bins=50, color=color, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted Return', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Prediction Distribution', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Residuals
    ax = axes[1, 0]
    residuals = predictions_df['actual_return'] - predictions_df['prediction']
    ax.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title(f'Residuals (Mean: {residuals.mean():.5f})', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Quantile performance
    ax = axes[1, 1]
    predictions_df['pred_quantile'] = pd.qcut(predictions_df['prediction'], 
                                                q=10, labels=False, duplicates='drop')
    quantile_returns = predictions_df.groupby('pred_quantile')['actual_return'].mean() * 100
    ax.bar(range(len(quantile_returns)), quantile_returns, color=color, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('Prediction Quantile (0=Lowest, 9=Highest)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mean Actual Return (%)', fontsize=10, fontweight='bold')
    ax.set_title('Returns by Prediction Quantile', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    fig.suptitle(f'{config["name"]}: Prediction Quality Analysis', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "5_prediction_quality.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 6: Monthly Performance
    # ========================================================================
    print("  Creating monthly performance...")
    fig, ax = plt.subplots(figsize=(14, 7))
    
    returns_df['month'] = pd.to_datetime(returns_df['date']).dt.to_period('M')
    monthly_returns = returns_df.groupby('month')['return'].sum() * 100
    
    colors_monthly = ['green' if x > 0 else 'red' for x in monthly_returns]
    ax.bar(range(len(monthly_returns)), monthly_returns, color=colors_monthly, 
           alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Month', fontsize=11, fontweight='bold')
    ax.set_ylabel('Monthly Return (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{config["name"]}: Monthly Returns', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(monthly_returns)))
    ax.set_xticklabels([str(m) for m in monthly_returns.index], rotation=45, ha='right')
    ax.grid(alpha=0.3, axis='y')
    
    # Add win rate
    win_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100
    ax.text(0.02, 0.98, f'Win Rate: {win_rate:.1f}%', transform=ax.transAxes,
            va='top', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "6_monthly_performance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 7: Summary Dashboard
    # ========================================================================
    print("  Creating summary dashboard...")
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.text(0.5, 0.7, config['name'], ha='center', va='top',
                  fontsize=24, fontweight='bold', transform=title_ax.transAxes)
    title_ax.text(0.5, 0.3, config['description'], ha='center', va='top',
                  fontsize=14, transform=title_ax.transAxes, style='italic')
    title_ax.axis('off')
    
    # Equity curve
    ax = fig.add_subplot(gs[1, :])
    ax.plot(returns_df['date'], returns_df['cumulative'], 
            linewidth=2.5, color=color)
    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.fill_between(returns_df['date'], 1, returns_df['cumulative'], 
                     where=(returns_df['cumulative'] >= 1), alpha=0.3, color='green')
    ax.fill_between(returns_df['date'], 1, returns_df['cumulative'], 
                     where=(returns_df['cumulative'] < 1), alpha=0.3, color='red')
    ax.set_ylabel('Cumulative Return', fontsize=11, fontweight='bold')
    ax.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Key metrics
    ax = fig.add_subplot(gs[2, 0])
    metrics_text = f"""
PERFORMANCE METRICS

Sharpe Ratio:    {sharpe:.3f}
Total Return:    {total_return*100:.2f}%
Max Drawdown:    {max_dd*100:.2f}%
Volatility:      {volatility*100:.2f}%

Daily Mean:      {returns_df['return'].mean()*100:.3f}%
Daily Std:       {returns_df['return'].std()*100:.3f}%
Win Rate:        {(returns_df['return'] > 0).mean()*100:.1f}%
"""
    ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.axis('off')
    
    # Returns distribution
    ax = fig.add_subplot(gs[2, 1])
    ax.hist(returns_df['return']*100, bins=40, color=color, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Daily Return (%)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax.set_title('Returns Distribution', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Drawdown
    ax = fig.add_subplot(gs[2, 2])
    ax.fill_between(returns_df['date'], 0, returns_df['drawdown']*100, 
                     color='red', alpha=0.5)
    ax.plot(returns_df['date'], returns_df['drawdown']*100, 
            color='darkred', linewidth=1.5)
    ax.set_xlabel('Date', fontsize=10, fontweight='bold')
    ax.set_ylabel('Drawdown (%)', fontsize=10, fontweight='bold')
    ax.set_title('Drawdown', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.savefig(output_dir / "7_summary_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # Create README
    # ========================================================================
    print("  Creating README...")
    readme = f"""# {config['name']}: Detailed Analysis

**Model:** {config['name']}
**Description:** {config['description']}
**Test Period:** July 2015 - December 2016 (18 months)
**Dataset:** 100 stocks, 37,900 test samples

## Performance Summary

- **Sharpe Ratio:** {sharpe:.3f}
- **Total Return:** {total_return*100:.2f}%
- **Max Drawdown:** {max_dd*100:.2f}%
- **Volatility:** {volatility*100:.2f}%
- **Win Rate:** {(returns_df['return'] > 0).mean()*100:.1f}%

## Visualizations

1. **Equity Curve** - Cumulative performance over time
2. **Drawdown Analysis** - Underwater periods
3. **Returns Distribution** - Statistical analysis of daily returns
4. **Long/Short Analysis** - Individual portfolio performance
5. **Prediction Quality** - Model accuracy and calibration
6. **Monthly Performance** - Month-by-month breakdown
7. **Summary Dashboard** - Complete overview

## Files Generated

All visualizations are 300 DPI PNG files suitable for presentations and reports.
"""
    
    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)
    
    print(f"  ✅ Complete! Saved to: {output_dir}")
    return sharpe, total_return, max_dd

# ============================================================================
# Process all models
# ============================================================================

results_summary = []

for config in models_config:
    try:
        sharpe, total_return, max_dd = create_model_visualizations(config)
        results_summary.append({
            'model': config['name'],
            'sharpe': sharpe,
            'return': total_return,
            'max_dd': max_dd,
            'status': 'SUCCESS'
        })
    except Exception as e:
        print(f"\n❌ Failed to process {config['name']}: {e}")
        import traceback
        traceback.print_exc()
        results_summary.append({
            'model': config['name'],
            'sharpe': None,
            'return': None,
            'max_dd': None,
            'status': f'FAILED: {str(e)}'
        })

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✅ INDIVIDUAL MODEL VISUALIZATIONS COMPLETE")
print("="*70)

for result in results_summary:
    if result['status'] == 'SUCCESS':
        print(f"\n{result['model']}:")
        print(f"  Sharpe:      {result['sharpe']:.3f}")
        print(f"  Return:      {result['return']*100:.2f}%")
        print(f"  Max DD:      {result['max_dd']*100:.2f}%")
    else:
        print(f"\n{result['model']}: {result['status']}")

print("\n" + "="*70)
print("Folders created:")
print("  - neural_nets/combined_ranker_ndcg_visualizations/")
print("  - neural_nets/deep_late_fusion_visualizations/")
print("  - neural_nets/deep_combined_visualizations/")
print("="*70)

