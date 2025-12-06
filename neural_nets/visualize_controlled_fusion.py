"""
Comprehensive visualization comparing Early vs Late Fusion models.

Creates detailed plots showing:
1. Training curves comparison
2. Performance metrics comparison
3. Equity curves and drawdowns
4. Prediction quality analysis
5. Summary dashboard
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10


def plot_training_curves(early_history, late_history, save_dir):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves Comparison: Early vs Late Fusion', fontsize=16, fontweight='bold')
    
    # Training loss
    axes[0, 0].plot(early_history['train_loss'], label='Early Fusion', linewidth=2, color='blue')
    axes[0, 0].plot(late_history['train_loss'], label='Late Fusion', linewidth=2, color='orange')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Training Loss (MSE)')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    axes[0, 1].plot(early_history['val_loss'], label='Early Fusion', linewidth=2, color='blue')
    axes[0, 1].plot(late_history['val_loss'], label='Late Fusion', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Validation Loss (MSE)')
    axes[0, 1].set_title('Validation Loss Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Mark best epochs
    axes[0, 1].axvline(early_history['best_epoch'], color='blue', linestyle='--', alpha=0.5, label='Early Best')
    axes[0, 1].axvline(late_history['best_epoch'], color='orange', linestyle='--', alpha=0.5, label='Late Best')
    
    # Learning rates
    axes[1, 0].plot(early_history.get('learning_rates', []), label='Early Fusion', linewidth=2, color='blue')
    axes[1, 0].plot(late_history.get('learning_rates', []), label='Late Fusion', linewidth=2, color='orange')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Train vs Val comparison
    axes[1, 1].plot(range(len(early_history['train_loss'])), 
                   np.array(early_history['val_loss']) - np.array(early_history['train_loss']),
                   label='Early Fusion (Val - Train)', linewidth=2, color='blue')
    axes[1, 1].plot(range(len(late_history['train_loss'])),
                   np.array(late_history['val_loss']) - np.array(late_history['train_loss']),
                   label='Late Fusion (Val - Train)', linewidth=2, color='orange')
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Generalization Gap')
    axes[1, 1].set_title('Overfitting Analysis (Val - Train Loss)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / '1_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Training curves saved")


def plot_metrics_comparison(results_df, save_dir):
    """Plot bar chart comparing key metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Metrics Comparison: Early vs Late Fusion', fontsize=16, fontweight='bold')
    
    metrics = [
        ('Sharpe Ratio', 'sharpe_ratio', 'higher'),
        ('Total Return (%)', 'total_return', 'higher', 100),
        ('Spearman Correlation', 'spearman', 'higher'),
        ('MSE', 'mse', 'lower'),
        ('Max Drawdown (%)', 'max_drawdown', 'higher', 100),
        ('Win Rate (%)', 'win_rate', 'higher', 100)
    ]
    
    for idx, metric_info in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        metric_name = metric_info[0]
        metric_key = metric_info[1]
        better = metric_info[2]
        scale = metric_info[3] if len(metric_info) > 3 else 1
        
        early_val = results_df.loc[results_df['model'] == 'Early Fusion', metric_key].values[0] * scale
        late_val = results_df.loc[results_df['model'] == 'Late Fusion', metric_key].values[0] * scale
        
        bars = ax.bar(['Early Fusion', 'Late Fusion'], [early_val, late_val], 
                     color=['blue', 'orange'], alpha=0.7, edgecolor='black', linewidth=2)
        
        # Highlight winner
        if better == 'higher':
            if early_val > late_val:
                bars[0].set_edgecolor('gold')
                bars[0].set_linewidth(3)
            elif late_val > early_val:
                bars[1].set_edgecolor('gold')
                bars[1].set_linewidth(3)
        else:  # lower is better
            if early_val < late_val:
                bars[0].set_edgecolor('gold')
                bars[0].set_linewidth(3)
            elif late_val < early_val:
                bars[1].set_edgecolor('gold')
                bars[1].set_linewidth(3)
        
        ax.set_title(metric_name, fontweight='bold')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}' if abs(height) < 10 else f'{height:.2f}',
                   ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_dir / '2_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Metrics comparison saved")


def plot_equity_curves(predictions_df, save_dir):
    """Plot equity curves for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Trading Performance: Early vs Late Fusion', fontsize=16, fontweight='bold')
    
    # Compute daily returns for each model
    daily_returns = {}
    
    for model_name, pred_col in [('Early Fusion', 'early_fusion_pred'), ('Late Fusion', 'late_fusion_pred')]:
        model_returns = []
        
        for date in predictions_df['date'].unique():
            day_df = predictions_df[predictions_df['date'] == date].copy()
            
            if len(day_df) < 10:
                continue
            
            # Sort by predictions
            day_df = day_df.sort_values(pred_col, ascending=False)
            
            # Long top 5, short bottom 5
            long_return = day_df.head(5)['actual'].mean()
            short_return = day_df.tail(5)['actual'].mean()
            
            model_returns.append(long_return - short_return)
        
        daily_returns[model_name] = np.array(model_returns)
    
    dates = pd.to_datetime(predictions_df['date'].unique())[:len(daily_returns['Early Fusion'])]
    
    # Equity curves
    axes[0, 0].plot(dates, np.cumsum(daily_returns['Early Fusion']), 
                   label='Early Fusion', linewidth=2, color='blue')
    axes[0, 0].plot(dates, np.cumsum(daily_returns['Late Fusion']), 
                   label='Late Fusion', linewidth=2, color='orange')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Cumulative Return')
    axes[0, 0].set_title('Equity Curves (Long Top-5, Short Bottom-5)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Drawdown
    for model_name, color in [('Early Fusion', 'blue'), ('Late Fusion', 'orange')]:
        cumulative = np.cumsum(daily_returns[model_name])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        
        axes[0, 1].plot(dates, drawdown, label=model_name, linewidth=2, color=color)
    
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Drawdown')
    axes[0, 1].set_title('Drawdown Analysis')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].fill_between(dates, 0, drawdown, alpha=0.3)
    
    # Daily returns distribution
    axes[1, 0].hist(daily_returns['Early Fusion'], bins=50, alpha=0.5, label='Early Fusion', color='blue', edgecolor='black')
    axes[1, 0].hist(daily_returns['Late Fusion'], bins=50, alpha=0.5, label='Late Fusion', color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Daily Returns Distribution')
    axes[1, 0].legend()
    axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rolling Sharpe
    window = 60  # 60-day rolling window
    for model_name, color in [('Early Fusion', 'blue'), ('Late Fusion', 'orange')]:
        rolling_sharpe = pd.Series(daily_returns[model_name]).rolling(window).apply(
            lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
        )
        axes[1, 1].plot(dates, rolling_sharpe, label=model_name, linewidth=2, color=color, alpha=0.7)
    
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel(f'{window}-Day Rolling Sharpe')
    axes[1, 1].set_title('Rolling Sharpe Ratio')
    axes[1, 1].legend()
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / '3_equity_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Equity curves saved")


def plot_prediction_quality(predictions_df, save_dir):
    """Plot prediction quality analysis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Prediction Quality Analysis: Early vs Late Fusion', fontsize=16, fontweight='bold')
    
    # Scatter plots: Predictions vs Actuals
    for idx, (model_name, pred_col, color) in enumerate([
        ('Early Fusion', 'early_fusion_pred', 'blue'),
        ('Late Fusion', 'late_fusion_pred', 'orange')
    ]):
        ax = axes[0, idx]
        
        # Sample for visibility
        sample_df = predictions_df.sample(min(5000, len(predictions_df)), random_state=42)
        
        ax.scatter(sample_df[pred_col], sample_df['actual'], 
                  alpha=0.3, s=10, color=color)
        
        # Add diagonal line (perfect predictions)
        min_val = min(sample_df[pred_col].min(), sample_df['actual'].min())
        max_val = max(sample_df[pred_col].max(), sample_df['actual'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Compute correlation
        corr = np.corrcoef(predictions_df[pred_col], predictions_df['actual'])[0, 1]
        spearman, _ = stats.spearmanr(predictions_df[pred_col], predictions_df['actual'])
        
        ax.set_xlabel('Predicted Return')
        ax.set_ylabel('Actual Return')
        ax.set_title(f'{model_name}\nPearson: {corr:.4f}, Spearman: {spearman:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Prediction error distributions
    for idx, (model_name, pred_col, color) in enumerate([
        ('Early Fusion', 'early_fusion_pred', 'blue'),
        ('Late Fusion', 'late_fusion_pred', 'orange')
    ]):
        ax = axes[1, idx]
        
        errors = predictions_df['actual'] - predictions_df[pred_col]
        
        ax.hist(errors, bins=100, alpha=0.7, color=color, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(errors.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: {errors.mean():.6f}')
        
        ax.set_xlabel('Prediction Error (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name} Error Distribution\nStd: {errors.std():.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # Direct comparison scatter
    ax = axes[0, 2]
    sample_df = predictions_df.sample(min(5000, len(predictions_df)), random_state=42)
    ax.scatter(sample_df['early_fusion_pred'], sample_df['late_fusion_pred'], 
              alpha=0.3, s=10, c=sample_df['actual'], cmap='RdYlGn')
    
    # Diagonal
    min_val = min(sample_df['early_fusion_pred'].min(), sample_df['late_fusion_pred'].min())
    max_val = max(sample_df['early_fusion_pred'].max(), sample_df['late_fusion_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Early Fusion Prediction')
    ax.set_ylabel('Late Fusion Prediction')
    ax.set_title('Model Agreement\n(Color = Actual Return)')
    ax.grid(True, alpha=0.3)
    
    # Error comparison boxplot
    ax = axes[1, 2]
    early_errors = np.abs(predictions_df['actual'] - predictions_df['early_fusion_pred'])
    late_errors = np.abs(predictions_df['actual'] - predictions_df['late_fusion_pred'])
    
    bp = ax.boxplot([early_errors, late_errors], labels=['Early Fusion', 'Late Fusion'],
                    patch_artist=True, showmeans=True)
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.5)
    bp['boxes'][1].set_facecolor('orange')
    bp['boxes'][1].set_alpha(0.5)
    
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / '4_prediction_quality.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Prediction quality saved")


def plot_summary_dashboard(results_df, early_history, late_history, save_dir):
    """Create comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('Controlled Fusion Comparison - Summary Dashboard', fontsize=18, fontweight='bold')
    
    # Extract metrics
    early_metrics = results_df[results_df['model'] == 'Early Fusion'].iloc[0]
    late_metrics = results_df[results_df['model'] == 'Late Fusion'].iloc[0]
    
    # 1. Model Architecture Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    arch_text = f"""
    EARLY FUSION (112K params)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Input(7) → 256 → 256 → 128 
    → 64 → 32 → Output(1)
    
    Strategy: Concatenate at input
    Fusion Point: Layer 0 (immediate)
    
    
    LATE FUSION (118K params)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Price: 4 → 180 → 180 → 90
    Sentiment: 3 → 180 → 180 → 90
    Fusion: 180 → 80 → 32 → Output
    
    Strategy: Separate branches
    Fusion Point: After processing
    """
    ax1.text(0.1, 0.5, arch_text, fontsize=9, fontfamily='monospace',
            verticalalignment='center')
    ax1.set_title('Architecture Comparison', fontweight='bold', fontsize=12)
    
    # 2. Winner board
    ax2 = fig.add_subplot(gs[0, 1:3])
    ax2.axis('off')
    
    winner_metrics = {
        'Sharpe Ratio': (early_metrics['sharpe_ratio'], late_metrics['sharpe_ratio'], 'higher'),
        'Total Return': (early_metrics['total_return'], late_metrics['total_return'], 'higher'),
        'Spearman': (early_metrics['spearman'], late_metrics['spearman'], 'higher'),
        'MSE': (early_metrics['mse'], late_metrics['mse'], 'lower'),
    }
    
    early_wins = 0
    late_wins = 0
    
    for metric, (early_val, late_val, direction) in winner_metrics.items():
        if direction == 'higher':
            if early_val > late_val:
                early_wins += 1
            elif late_val > early_val:
                late_wins += 1
        else:
            if early_val < late_val:
                early_wins += 1
            elif late_val < early_val:
                late_wins += 1
    
    overall_winner = "EARLY FUSION" if early_wins > late_wins else "LATE FUSION" if late_wins > early_wins else "TIE"
    winner_color = 'blue' if overall_winner == "EARLY FUSION" else 'orange' if overall_winner == "LATE FUSION" else 'gray'
    
    winner_text = f"""
    🏆 OVERALL WINNER: {overall_winner} 🏆
    
    Score: Early {early_wins} - {late_wins} Late
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Early Fusion:
      Sharpe: {early_metrics['sharpe_ratio']:.4f}
      Return: {early_metrics['total_return']*100:.2f}%
      Spearman: {early_metrics['spearman']:.4f}
    
    Late Fusion:
      Sharpe: {late_metrics['sharpe_ratio']:.4f}
      Return: {late_metrics['total_return']*100:.2f}%
      Spearman: {late_metrics['spearman']:.4f}
    """
    
    ax2.text(0.5, 0.5, winner_text, fontsize=11, fontfamily='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=winner_color, alpha=0.2))
    ax2.set_title('Winner Board', fontweight='bold', fontsize=12)
    
    # 3. Training summary
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.axis('off')
    training_text = f"""
    TRAINING SUMMARY
    ━━━━━━━━━━━━━━━━━━━━━
    
    Early Fusion:
      Best epoch: {early_history['best_epoch']}
      Best val loss: {early_history['best_val_loss']:.6f}
      Total epochs: {len(early_history['train_loss'])}
    
    Late Fusion:
      Best epoch: {late_history['best_epoch']}
      Best val loss: {late_history['best_val_loss']:.6f}
      Total epochs: {len(late_history['train_loss'])}
    """
    ax3.text(0.1, 0.5, training_text, fontsize=9, fontfamily='monospace',
            verticalalignment='center')
    ax3.set_title('Training Info', fontweight='bold', fontsize=12)
    
    # 4-7. Metric comparison bars
    metrics_to_plot = [
        ('Sharpe Ratio', 'sharpe_ratio', 1),
        ('Total Return (%)', 'total_return', 100),
        ('Spearman', 'spearman', 1),
        ('Win Rate (%)', 'win_rate', 100),
    ]
    
    for idx, (name, key, scale) in enumerate(metrics_to_plot):
        ax = fig.add_subplot(gs[1, idx])
        early_val = early_metrics[key] * scale
        late_val = late_metrics[key] * scale
        
        bars = ax.barh(['Late', 'Early'], [late_val, early_val], color=['orange', 'blue'], alpha=0.7)
        ax.set_xlabel('Value')
        ax.set_title(name, fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2.,
                   f'{width:.4f}' if abs(width) < 10 else f'{width:.2f}',
                   ha='left', va='center', fontweight='bold')
    
    # 8. Validation loss comparison
    ax8 = fig.add_subplot(gs[2, :2])
    ax8.plot(early_history['val_loss'], label='Early Fusion', linewidth=2, color='blue')
    ax8.plot(late_history['val_loss'], label='Late Fusion', linewidth=2, color='orange')
    ax8.axvline(early_history['best_epoch'], color='blue', linestyle='--', alpha=0.5)
    ax8.axvline(late_history['best_epoch'], color='orange', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Epoch')
    ax8.set_ylabel('Validation Loss')
    ax8.set_title('Validation Loss Comparison', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Key insights
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis('off')
    
    sharpe_diff = ((late_metrics['sharpe_ratio'] - early_metrics['sharpe_ratio']) / 
                   abs(early_metrics['sharpe_ratio']) * 100 if early_metrics['sharpe_ratio'] != 0 else 0)
    
    insights_text = f"""
    KEY INSIGHTS
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    • Sharpe Ratio Difference: {sharpe_diff:+.1f}%
      {'Late Fusion performs better' if sharpe_diff > 5 else 'Early Fusion performs better' if sharpe_diff < -5 else 'Similar performance'}
    
    • Parameter Efficiency: {'Early Fusion' if early_wins > late_wins else 'Late Fusion'}
      wins with {abs(118489 - 112577):,} {'fewer' if early_wins > late_wins else 'more'} parameters
    
    • Training Stability: 
      Early best at epoch {early_history['best_epoch']}, Late at {late_history['best_epoch']}
    
    • Prediction Quality:
      Both achieve Spearman ≈ {np.mean([early_metrics['spearman'], late_metrics['spearman']]):.4f}
    
    • Conclusion: {overall_winner} provides {'marginally' if abs(early_wins - late_wins) <= 1 else 'clearly'} 
      better results for this task
    """
    
    ax9.text(0.05, 0.5, insights_text, fontsize=10, fontfamily='monospace',
            verticalalignment='center')
    ax9.set_title('Analysis', fontweight='bold', fontsize=12)
    
    plt.savefig(save_dir / '5_summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Summary dashboard saved")


def main():
    """Create all visualizations."""
    print("="*80)
    print("CONTROLLED FUSION VISUALIZATION")
    print("="*80)
    print()
    
    # Create output directory
    save_dir = Path(__file__).parent / 'controlled_fusion_visualizations'
    save_dir.mkdir(exist_ok=True)
    
    print(f"Output directory: {save_dir}")
    print()
    
    # Load results
    results_path = Path(__file__).parent / 'results' / 'controlled_fusion_comparison.csv'
    predictions_path = Path(__file__).parent / 'results' / 'controlled_fusion_predictions.csv'
    
    if not results_path.exists() or not predictions_path.exists():
        print("❌ Error: Results files not found!")
        print("   Run evaluate_controlled_fusion.py first")
        return
    
    results_df = pd.read_csv(results_path)
    predictions_df = pd.read_csv(predictions_path)
    
    print(f"Loaded {len(results_df)} model results")
    print(f"Loaded {len(predictions_df):,} predictions")
    print()
    
    # Load training histories
    import torch
    
    early_checkpoint = torch.load(
        Path(__file__).parent / 'trained_models' / 'early_fusion_100k_best.pt',
        map_location='cpu'
    )
    late_checkpoint = torch.load(
        Path(__file__).parent / 'trained_models' / 'late_fusion_100k_best.pt',
        map_location='cpu'
    )
    
    early_history = early_checkpoint['train_history']
    late_history = late_checkpoint['train_history']
    
    # Create visualizations
    print("Creating visualizations...")
    print()
    
    plot_training_curves(early_history, late_history, save_dir)
    plot_metrics_comparison(results_df, save_dir)
    plot_equity_curves(predictions_df, save_dir)
    plot_prediction_quality(predictions_df, save_dir)
    plot_summary_dashboard(results_df, early_history, late_history, save_dir)
    
    # Create README
    readme_text = f"""# Controlled Fusion Comparison Visualizations

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Models:** Early Fusion (112K params) vs Late Fusion (118K params)
**Dataset:** 100 stocks, 38,000 test samples

## Results Summary

### Early Fusion
- Sharpe Ratio: {results_df.loc[0, 'sharpe_ratio']:.4f}
- Total Return: {results_df.loc[0, 'total_return']*100:.2f}%
- Spearman: {results_df.loc[0, 'spearman']:.4f}

### Late Fusion  
- Sharpe Ratio: {results_df.loc[1, 'sharpe_ratio']:.4f}
- Total Return: {results_df.loc[1, 'total_return']*100:.2f}%
- Spearman: {results_df.loc[1, 'spearman']:.4f}

## Visualizations

1. **training_curves.png** - Training and validation loss over epochs
2. **metrics_comparison.png** - Bar charts comparing key metrics
3. **equity_curves.png** - Trading performance and drawdowns
4. **prediction_quality.png** - Prediction accuracy analysis
5. **summary_dashboard.png** - Comprehensive overview

## Conclusion

This controlled experiment definitively compares early vs late fusion strategies
with matched parameter counts (~115K params each) on the same dataset.
"""
    
    (save_dir / 'README.md').write_text(readme_text)
    
    print()
    print("="*80)
    print("✅ All visualizations complete!")
    print("="*80)
    print(f"\nSaved to: {save_dir}/")
    print("\nFiles created:")
    for f in sorted(save_dir.glob('*.png')):
        print(f"  • {f.name}")
    print(f"  • README.md")


if __name__ == "__main__":
    main()

