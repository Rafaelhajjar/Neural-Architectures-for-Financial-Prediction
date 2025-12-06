"""
Time Series Cross-Validation for Controlled Fusion Comparison.

Evaluates both models on multiple non-overlapping time periods to test
if performance differences are consistent or period-specific.
"""
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from neural_nets.models.controlled_fusion import EarlyFusion100K, LateFusion100K
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (16, 10)


def evaluate_model_on_period(model, data_df, price_features, sentiment_features, device='cpu'):
    """Evaluate model on a specific time period."""
    model.eval()
    
    predictions = []
    actuals = []
    dates = []
    tickers = []
    
    # Process in batches
    with torch.no_grad():
        for idx in range(len(data_df)):
            row = data_df.iloc[idx]
            
            # Convert to float arrays explicitly
            price_vals = [float(row[f]) for f in price_features]
            sentiment_vals = [float(row[f]) for f in sentiment_features]
            
            x_price = torch.tensor(price_vals, dtype=torch.float32).unsqueeze(0).to(device)
            x_sentiment = torch.tensor(sentiment_vals, dtype=torch.float32).unsqueeze(0).to(device)
            y_actual = float(row['future_return'])
            
            y_pred = model(x_price, x_sentiment)
            
            predictions.append(y_pred.cpu().numpy().flatten()[0])
            actuals.append(y_actual)
            dates.append(row['date'])
            tickers.append(row['ticker'])
    
    # Compute metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Long-short strategy
    df = pd.DataFrame({
        'date': dates,
        'ticker': tickers,
        'prediction': predictions,
        'actual': actuals
    })
    
    daily_returns = []
    for date in df['date'].unique():
        day_df = df[df['date'] == date].copy()
        
        if len(day_df) < 10:
            continue
        
        day_df = day_df.sort_values('prediction', ascending=False)
        long_return = day_df.head(5)['actual'].mean()
        short_return = day_df.tail(5)['actual'].mean()
        daily_returns.append(long_return - short_return)
    
    daily_returns = np.array(daily_returns)
    
    # Compute metrics
    spearman, _ = stats.spearmanr(predictions, actuals)
    mse = np.mean((predictions - actuals) ** 2)
    
    mean_return = np.mean(daily_returns)
    volatility = np.std(daily_returns)
    sharpe = (mean_return / volatility) * np.sqrt(252) if volatility > 0 else 0
    
    total_return = np.sum(daily_returns)
    cumulative = np.cumsum(daily_returns)
    max_dd = np.min(cumulative - np.maximum.accumulate(cumulative))
    win_rate = np.sum(daily_returns > 0) / len(daily_returns) if len(daily_returns) > 0 else 0
    
    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'mean_daily_return': mean_return,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'spearman': spearman,
        'mse': mse,
        'num_days': len(daily_returns),
        'daily_returns': daily_returns
    }


def timeseries_cross_validation():
    """Perform time series cross-validation."""
    print("="*80)
    print("TIME SERIES CROSS-VALIDATION: Early vs Late Fusion")
    print("="*80)
    print()
    
    device = 'cpu'
    
    # Load data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'features_expanded_100stocks_with_sentiment.parquet'
    
    train_df, val_df, test_df = load_and_prepare_data(
        data_path=str(data_path),
        train_end='2013-12-31',
        val_end='2015-06-30'
    )
    
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"Total test samples: {len(test_df):,}")
    print()
    
    # Load models
    early_model = EarlyFusion100K(task='regression').to(device)
    late_model = LateFusion100K(task='regression').to(device)
    
    early_checkpoint = torch.load(
        Path(__file__).parent / 'trained_models' / 'early_fusion_100k_best.pt',
        map_location=device
    )
    late_checkpoint = torch.load(
        Path(__file__).parent / 'trained_models' / 'late_fusion_100k_best.pt',
        map_location=device
    )
    
    early_model.load_state_dict(early_checkpoint['model_state_dict'])
    late_model.load_state_dict(late_checkpoint['model_state_dict'])
    
    print("Models loaded successfully")
    print()
    
    # Define time periods (3-month windows)
    periods = [
        ('2015-07-01', '2015-09-30', 'Q3 2015'),
        ('2015-10-01', '2015-12-31', 'Q4 2015'),
        ('2016-01-01', '2016-03-31', 'Q1 2016'),
        ('2016-04-01', '2016-06-30', 'Q2 2016'),
        ('2016-07-01', '2016-09-30', 'Q3 2016'),
        ('2016-10-01', '2016-12-31', 'Q4 2016'),
    ]
    
    features = {
        'price': ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank'],
        'sentiment': ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']
    }
    
    results = {
        'early': {'periods': [], 'metrics': []},
        'late': {'periods': [], 'metrics': []}
    }
    
    # Evaluate on each period
    print("Evaluating on time periods:")
    print("-" * 80)
    
    for start_date, end_date, label in periods:
        period_df = test_df[
            (test_df['date'] >= start_date) & 
            (test_df['date'] <= end_date)
        ].copy()
        
        print(f"\n{label} ({start_date} to {end_date})")
        print(f"  Samples: {len(period_df):,}")
        
        # Evaluate early fusion
        early_metrics = evaluate_model_on_period(
            early_model, period_df, 
            features['price'], features['sentiment'], device
        )
        
        # Evaluate late fusion
        late_metrics = evaluate_model_on_period(
            late_model, period_df,
            features['price'], features['sentiment'], device
        )
        
        # Store results
        results['early']['periods'].append(label)
        results['early']['metrics'].append(early_metrics)
        results['late']['periods'].append(label)
        results['late']['metrics'].append(late_metrics)
        
        # Print comparison
        print(f"  Early Fusion: Sharpe {early_metrics['sharpe']:.3f}, Return {early_metrics['total_return']*100:+.1f}%")
        print(f"  Late Fusion:  Sharpe {late_metrics['sharpe']:.3f}, Return {late_metrics['total_return']*100:+.1f}%")
        
        winner = "Early" if early_metrics['sharpe'] > late_metrics['sharpe'] else "Late"
        print(f"  Winner: {winner} 🏆")
    
    print()
    print("="*80)
    
    # Statistical tests
    early_sharpes = [m['sharpe'] for m in results['early']['metrics']]
    late_sharpes = [m['sharpe'] for m in results['late']['metrics']]
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(early_sharpes, late_sharpes)
    
    print("\nSTATISTICAL ANALYSIS")
    print("="*80)
    print(f"\nEarly Fusion (across {len(periods)} periods):")
    print(f"  Mean Sharpe:   {np.mean(early_sharpes):.3f}")
    print(f"  Std Sharpe:    {np.std(early_sharpes):.3f}")
    print(f"  Min Sharpe:    {np.min(early_sharpes):.3f}")
    print(f"  Max Sharpe:    {np.max(early_sharpes):.3f}")
    print(f"  Positive periods: {sum(1 for s in early_sharpes if s > 0)}/{len(early_sharpes)}")
    
    print(f"\nLate Fusion (across {len(periods)} periods):")
    print(f"  Mean Sharpe:   {np.mean(late_sharpes):.3f}")
    print(f"  Std Sharpe:    {np.std(late_sharpes):.3f}")
    print(f"  Min Sharpe:    {np.min(late_sharpes):.3f}")
    print(f"  Max Sharpe:    {np.max(late_sharpes):.3f}")
    print(f"  Positive periods: {sum(1 for s in late_sharpes if s > 0)}/{len(late_sharpes)}")
    
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value:     {p_value:.6f}")
    
    if p_value < 0.05:
        print(f"  ✅ Difference is statistically significant (p < 0.05)")
        effect_size = (np.mean(early_sharpes) - np.mean(late_sharpes)) / np.std(early_sharpes - np.array(late_sharpes))
        print(f"  Effect size (Cohen's d): {effect_size:.3f}")
    else:
        print(f"  ⚠️ Difference not statistically significant (p >= 0.05)")
    
    # Win rate
    wins_early = sum(1 for e, l in zip(early_sharpes, late_sharpes) if e > l)
    wins_late = sum(1 for e, l in zip(early_sharpes, late_sharpes) if l > e)
    
    print(f"\nPeriod-by-period wins:")
    print(f"  Early Fusion: {wins_early}/{len(periods)}")
    print(f"  Late Fusion:  {wins_late}/{len(periods)}")
    
    # Save results
    results_df = pd.DataFrame({
        'period': results['early']['periods'],
        'early_sharpe': [m['sharpe'] for m in results['early']['metrics']],
        'late_sharpe': [m['sharpe'] for m in results['late']['metrics']],
        'early_return': [m['total_return'] for m in results['early']['metrics']],
        'late_return': [m['total_return'] for m in results['late']['metrics']],
        'early_win_rate': [m['win_rate'] for m in results['early']['metrics']],
        'late_win_rate': [m['win_rate'] for m in results['late']['metrics']],
    })
    
    save_path = Path(__file__).parent / 'results' / 'timeseries_cv_results.csv'
    results_df.to_csv(save_path, index=False)
    print(f"\n💾 Results saved to: {save_path}")
    
    return results, results_df


def create_visualizations(results, results_df):
    """Create comprehensive time series CV visualizations."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    print()
    
    save_dir = Path(__file__).parent / 'controlled_fusion_visualizations'
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Time Series Cross-Validation: Early vs Late Fusion', 
                 fontsize=18, fontweight='bold')
    
    periods = results_df['period'].values
    early_sharpes = results_df['early_sharpe'].values
    late_sharpes = results_df['late_sharpe'].values
    early_returns = results_df['early_return'].values * 100
    late_returns = results_df['late_return'].values * 100
    
    # 1. Period-by-period Sharpe comparison
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(periods))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, early_sharpes, width, label='Early Fusion', 
                    color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, late_sharpes, width, label='Late Fusion',
                    color='orange', alpha=0.7, edgecolor='black')
    
    ax1.set_xlabel('Time Period', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12)
    ax1.set_title('Sharpe Ratio by Time Period', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, rotation=45, ha='right')
    ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=9)
    
    # 2. Cumulative performance over periods
    ax2 = fig.add_subplot(gs[0, 2])
    cumulative_early = np.cumsum(early_returns)
    cumulative_late = np.cumsum(late_returns)
    
    ax2.plot(periods, cumulative_early, marker='o', linewidth=2, 
            color='blue', label='Early Fusion', markersize=8)
    ax2.plot(periods, cumulative_late, marker='s', linewidth=2,
            color='orange', label='Late Fusion', markersize=8)
    ax2.set_xlabel('Time Period', fontsize=11)
    ax2.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax2.set_title('Cumulative Returns', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Distribution of Sharpe ratios
    ax3 = fig.add_subplot(gs[1, 0])
    
    bp = ax3.boxplot([early_sharpes, late_sharpes], 
                     labels=['Early Fusion', 'Late Fusion'],
                     patch_artist=True, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('orange')
    bp['boxes'][1].set_alpha(0.5)
    
    ax3.set_ylabel('Sharpe Ratio', fontsize=11)
    ax3.set_title('Sharpe Distribution Across Periods', fontsize=13, fontweight='bold')
    ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add mean values as text
    ax3.text(1, np.mean(early_sharpes), f'μ={np.mean(early_sharpes):.3f}',
            ha='center', va='bottom', fontweight='bold')
    ax3.text(2, np.mean(late_sharpes), f'μ={np.mean(late_sharpes):.3f}',
            ha='center', va='bottom', fontweight='bold')
    
    # 4. Scatter: Early vs Late Sharpe
    ax4 = fig.add_subplot(gs[1, 1])
    
    ax4.scatter(early_sharpes, late_sharpes, s=150, alpha=0.7, 
               edgecolors='black', linewidths=2)
    
    # Add diagonal line (equal performance)
    min_val = min(early_sharpes.min(), late_sharpes.min())
    max_val = max(early_sharpes.max(), late_sharpes.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
            alpha=0.5, label='Equal Performance')
    
    # Label points
    for i, period in enumerate(periods):
        ax4.annotate(period, (early_sharpes[i], late_sharpes[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax4.set_xlabel('Early Fusion Sharpe', fontsize=11)
    ax4.set_ylabel('Late Fusion Sharpe', fontsize=11)
    ax4.set_title('Period-by-Period Comparison', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(0, color='black', linestyle='--', alpha=0.2)
    ax4.axvline(0, color='black', linestyle='--', alpha=0.2)
    
    # 5. Win rate by period
    ax5 = fig.add_subplot(gs[1, 2])
    
    early_win_rates = results_df['early_win_rate'].values * 100
    late_win_rates = results_df['late_win_rate'].values * 100
    
    bars1 = ax5.bar(x - width/2, early_win_rates, width, label='Early Fusion',
                   color='blue', alpha=0.7, edgecolor='black')
    bars2 = ax5.bar(x + width/2, late_win_rates, width, label='Late Fusion',
                   color='orange', alpha=0.7, edgecolor='black')
    
    ax5.set_xlabel('Time Period', fontsize=11)
    ax5.set_ylabel('Win Rate (%)', fontsize=11)
    ax5.set_title('Win Rate by Period', fontsize=13, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(periods, rotation=45, ha='right')
    ax5.axhline(50, color='black', linestyle='--', alpha=0.3, label='50% (Random)')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Statistical summary
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    
    # Perform statistical tests
    t_stat, p_value = stats.ttest_rel(early_sharpes, late_sharpes)
    wins_early = sum(1 for e, l in zip(early_sharpes, late_sharpes) if e > l)
    
    summary_text = f"""
    STATISTICAL SUMMARY (Time Series Cross-Validation)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Early Fusion Performance:                    Late Fusion Performance:
      Mean Sharpe:     {np.mean(early_sharpes):>7.3f}                      Mean Sharpe:     {np.mean(late_sharpes):>7.3f}
      Std Dev:         {np.std(early_sharpes):>7.3f}                      Std Dev:         {np.std(late_sharpes):>7.3f}
      Min/Max:         {np.min(early_sharpes):>7.3f} / {np.max(early_sharpes):<7.3f}              Min/Max:         {np.min(late_sharpes):>7.3f} / {np.max(late_sharpes):<7.3f}
      Positive periods: {sum(1 for s in early_sharpes if s > 0)}/{len(early_sharpes)}                          Positive periods: {sum(1 for s in late_sharpes if s > 0)}/{len(late_sharpes)}
    
    Period-by-Period Comparison:
      Early Fusion wins: {wins_early}/{len(periods)} periods ({wins_early/len(periods)*100:.1f}%)
      Late Fusion wins:  {len(periods)-wins_early}/{len(periods)} periods ({(len(periods)-wins_early)/len(periods)*100:.1f}%)
    
    Statistical Significance (Paired t-test):
      t-statistic: {t_stat:.4f}
      p-value:     {p_value:.6f}  {'✅ SIGNIFICANT (p < 0.05)' if p_value < 0.05 else '⚠️ NOT SIGNIFICANT (p >= 0.05)'}
      
    {'Effect size (Cohen\'s d): ' + f'{(np.mean(early_sharpes) - np.mean(late_sharpes)) / np.std(early_sharpes - late_sharpes):.3f}' if p_value < 0.05 else ''}
    
    Conclusion:
      {'Early Fusion CONSISTENTLY outperforms Late Fusion across time periods.' if wins_early > len(periods)/2 and p_value < 0.05 else 'Results are mixed or not statistically robust across periods.'}
      {'The performance difference is statistically significant and reproducible.' if p_value < 0.05 else 'More data needed to establish statistical significance.'}
    """
    
    ax6.text(0.05, 0.5, summary_text, fontsize=11, fontfamily='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.savefig(save_dir / '6_timeseries_cross_validation.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Time series cross-validation plot saved")
    
    # Create second figure: detailed period analysis
    fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig2.suptitle('Detailed Period-by-Period Analysis', fontsize=16, fontweight='bold')
    
    for idx, (period, ax) in enumerate(zip(periods, axes.flat)):
        early_metrics = results['early']['metrics'][idx]
        late_metrics = results['late']['metrics'][idx]
        
        # Plot cumulative returns for this period
        ax.plot(np.cumsum(early_metrics['daily_returns']), 
               label='Early Fusion', linewidth=2, color='blue')
        ax.plot(np.cumsum(late_metrics['daily_returns']),
               label='Late Fusion', linewidth=2, color='orange')
        
        ax.set_title(f"{period}\nEarly: {early_metrics['sharpe']:.2f}, Late: {late_metrics['sharpe']:.2f}",
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('Trading Day')
        ax.set_ylabel('Cumulative Return')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / '7_period_details.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Period details plot saved")
    
    print(f"\n💾 All visualizations saved to: {save_dir}/")


if __name__ == "__main__":
    # Run time series cross-validation
    results, results_df = timeseries_cross_validation()
    
    # Create visualizations
    create_visualizations(results, results_df)
    
    print("\n" + "="*80)
    print("✅ TIME SERIES CROSS-VALIDATION COMPLETE!")
    print("="*80)

