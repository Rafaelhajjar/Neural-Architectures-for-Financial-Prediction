"""
Comprehensive visualizations for 100-stock neural network experiment.
Focuses exclusively on the expanded dataset results.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

OUTPUT_DIR = Path("neural_nets/100visualisation")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("CREATING 100-STOCK EXPERIMENT VISUALIZATIONS")
print("="*70)

# Load results
mse_results = pd.read_csv('neural_nets/results/expanded_100stocks_results.csv')
ndcg_results = pd.read_csv('neural_nets/results/ndcg_expanded_100stocks_results.csv')

# Combine all results
all_results = pd.concat([mse_results, ndcg_results], ignore_index=True)

# Add loss type column
all_results['loss_type'] = all_results['model'].apply(lambda x: 'NDCG' if 'NDCG' in x else 'MSE')
all_results['base_model'] = all_results['model'].str.replace(' (MSE)', '').str.replace(' (NDCG)', '')

print(f"\nTotal models: {len(all_results)}")
print(f"MSE models: {(all_results['loss_type'] == 'MSE').sum()}")
print(f"NDCG models: {(all_results['loss_type'] == 'NDCG').sum()}")

# ============================================================================
# PLOT 1: Model Performance Comparison (Sharpe Ratio)
# ============================================================================
print("\n1. Creating Sharpe Ratio comparison...")

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by Sharpe ratio
plot_data = all_results.sort_values('sharpe_ratio', ascending=True)

# Color by performance
colors = ['red' if x < 0 else 'orange' if x < 0.5 else 'green' 
          for x in plot_data['sharpe_ratio']]

bars = ax.barh(range(len(plot_data)), plot_data['sharpe_ratio'], color=colors, alpha=0.7)

# Add value labels
for i, (idx, row) in enumerate(plot_data.iterrows()):
    value = row['sharpe_ratio']
    ax.text(value + 0.05 if value > 0 else value - 0.05, i, 
            f"{value:.2f}", va='center', fontweight='bold', fontsize=10)

ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data['model'], fontsize=11)
ax.set_xlabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('100-Stock Experiment: Model Performance Comparison', 
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_sharpe_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 1_sharpe_comparison.png")

# ============================================================================
# PLOT 2: Return vs Risk Scatter
# ============================================================================
print("2. Creating return vs risk scatter...")

fig, ax = plt.subplots(figsize=(12, 8))

for loss_type in ['MSE', 'NDCG']:
    data = all_results[all_results['loss_type'] == loss_type]
    ax.scatter(data['volatility']*100, data['total_return']*100, 
               s=200, alpha=0.7, label=loss_type,
               marker='o' if loss_type == 'MSE' else 's')
    
    # Add labels
    for _, row in data.iterrows():
        ax.annotate(row['base_model'], 
                   (row['volatility']*100, row['total_return']*100),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Volatility (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Return (%)', fontsize=12, fontweight='bold')
ax.set_title('100-Stock Experiment: Risk-Return Profile', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "2_risk_return_scatter.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 2_risk_return_scatter.png")

# ============================================================================
# PLOT 3: Performance Metrics Heatmap
# ============================================================================
print("3. Creating performance metrics heatmap...")

# Select key metrics
metrics_cols = ['sharpe_ratio', 'total_return', 'max_drawdown', 'spearman', 'mse', 'mae']
heatmap_data = all_results[['model'] + metrics_cols].set_index('model')

# Normalize for better visualization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
heatmap_normalized = pd.DataFrame(
    scaler.fit_transform(heatmap_data),
    index=heatmap_data.index,
    columns=heatmap_data.columns
)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_normalized, annot=heatmap_data.round(3), fmt='', 
            cmap='RdYlGn', center=0.5, linewidths=0.5,
            cbar_kws={'label': 'Normalized Score'}, ax=ax)
ax.set_title('100-Stock Experiment: Performance Metrics Overview', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('')
ax.set_ylabel('Model', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "3_metrics_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 3_metrics_heatmap.png")

# ============================================================================
# PLOT 4: MSE vs NDCG Loss Comparison
# ============================================================================
print("4. Creating MSE vs NDCG comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Only compare models that have both MSE and NDCG versions
comparable_models = ['Combined Ranker', 'Late Fusion Ranker']
comparison_data = all_results[all_results['base_model'].isin(comparable_models)]

metrics_to_plot = [
    ('sharpe_ratio', 'Sharpe Ratio', axes[0, 0]),
    ('total_return', 'Total Return (%)', axes[0, 1]),
    ('max_drawdown', 'Max Drawdown (%)', axes[1, 0]),
    ('spearman', 'Spearman Correlation', axes[1, 1])
]

for metric, title, ax in metrics_to_plot:
    pivot_data = comparison_data.pivot(index='base_model', columns='loss_type', values=metric)
    
    if 'return' in metric or 'drawdown' in metric:
        pivot_data = pivot_data * 100
    
    x = np.arange(len(pivot_data))
    width = 0.35
    
    ax.bar(x - width/2, pivot_data['MSE'], width, label='MSE', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, pivot_data['NDCG'], width, label='NDCG', alpha=0.8, color='coral')
    
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_data.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, model in enumerate(pivot_data.index):
        for j, loss in enumerate(['MSE', 'NDCG']):
            value = pivot_data.loc[model, loss]
            x_pos = i + (-width/2 if j == 0 else width/2)
            y_pos = value + (abs(value) * 0.05)
            ax.text(x_pos, y_pos, f'{value:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('100-Stock Experiment: MSE vs NDCG Loss Comparison', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "4_mse_vs_ndcg.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 4_mse_vs_ndcg.png")

# ============================================================================
# PLOT 5: Top Models Detailed Comparison
# ============================================================================
print("5. Creating top models detailed comparison...")

# Get top 3 models by Sharpe
top_models = all_results.nlargest(3, 'sharpe_ratio')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Sharpe Ratio
ax = axes[0]
bars = ax.bar(range(len(top_models)), top_models['sharpe_ratio'], 
              color=['gold', 'silver', '#CD7F32'], alpha=0.7)
ax.set_ylabel('Sharpe Ratio', fontsize=11, fontweight='bold')
ax.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(top_models)))
ax.set_xticklabels([m[:20] for m in top_models['model']], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, (_, row) in enumerate(top_models.iterrows()):
    ax.text(i, row['sharpe_ratio'] + 0.03, f"{row['sharpe_ratio']:.2f}", 
           ha='center', fontweight='bold')

# Total Return
ax = axes[1]
bars = ax.bar(range(len(top_models)), top_models['total_return']*100, 
              color=['gold', 'silver', '#CD7F32'], alpha=0.7)
ax.set_ylabel('Total Return (%)', fontsize=11, fontweight='bold')
ax.set_title('Total Return', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(top_models)))
ax.set_xticklabels([m[:20] for m in top_models['model']], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
for i, (_, row) in enumerate(top_models.iterrows()):
    ax.text(i, row['total_return']*100 + 1, f"{row['total_return']*100:.1f}%", 
           ha='center', fontweight='bold')

# Max Drawdown
ax = axes[2]
bars = ax.bar(range(len(top_models)), top_models['max_drawdown']*100, 
              color=['gold', 'silver', '#CD7F32'], alpha=0.7)
ax.set_ylabel('Max Drawdown (%)', fontsize=11, fontweight='bold')
ax.set_title('Max Drawdown', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(top_models)))
ax.set_xticklabels([m[:20] for m in top_models['model']], rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
for i, (_, row) in enumerate(top_models.iterrows()):
    ax.text(i, row['max_drawdown']*100 - 2, f"{row['max_drawdown']*100:.1f}%", 
           ha='center', fontweight='bold')

fig.suptitle('100-Stock Experiment: Top 3 Models Detailed Analysis', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "5_top3_models.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 5_top3_models.png")

# ============================================================================
# PLOT 6: Model Complexity vs Performance
# ============================================================================
print("6. Creating complexity vs performance analysis...")

# Add parameter counts (from known architectures)
param_counts = {
    'Combined Ranker (MSE)': 11393,
    'Combined Ranker (NDCG)': 11393,
    'Late Fusion Ranker (MSE)': 6849,
    'Late Fusion Ranker (NDCG)': 6849,
    'Deep Late Fusion': 71297,
    'Deep Combined': 95873
}

all_results['parameters'] = all_results['model'].map(param_counts)

fig, ax = plt.subplots(figsize=(12, 8))

# Color by loss type
colors_map = {'MSE': 'steelblue', 'NDCG': 'coral'}
for loss_type in ['MSE', 'NDCG']:
    data = all_results[all_results['loss_type'] == loss_type].dropna(subset=['parameters'])
    ax.scatter(data['parameters'], data['sharpe_ratio'], 
               s=300, alpha=0.7, label=loss_type,
               color=colors_map[loss_type],
               marker='o' if loss_type == 'MSE' else 's',
               edgecolors='black', linewidth=1.5)
    
    for _, row in data.iterrows():
        ax.annotate(row['base_model'], 
                   (row['parameters'], row['sharpe_ratio']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=10, fontweight='bold')

ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Break-even')
ax.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
ax.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_title('100-Stock Experiment: Model Complexity vs Performance', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xscale('log')
ax.legend(fontsize=11, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "6_complexity_vs_performance.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 6_complexity_vs_performance.png")

# ============================================================================
# PLOT 7: Summary Dashboard
# ============================================================================
print("7. Creating summary dashboard...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Top left: Best model highlight
ax1 = fig.add_subplot(gs[0, :2])
best_model = all_results.loc[all_results['sharpe_ratio'].idxmax()]
info_text = f"""
🏆 BEST MODEL: {best_model['model']}

Sharpe Ratio:      {best_model['sharpe_ratio']:.3f}
Total Return:      {best_model['total_return']*100:.2f}%
Max Drawdown:      {best_model['max_drawdown']*100:.2f}%
Volatility:        {best_model['volatility']*100:.2f}%
Spearman Corr:     {best_model['spearman']:.4f}
MSE:               {best_model['mse']:.6f}
Parameters:        {param_counts.get(best_model['model'], 'N/A'):,}
"""
ax1.text(0.05, 0.5, info_text, transform=ax1.transAxes, 
         fontsize=13, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='gold', alpha=0.3))
ax1.axis('off')

# Top right: Dataset info
ax2 = fig.add_subplot(gs[0, 2])
dataset_text = f"""
📊 DATASET INFO

Stocks:        100
Total Samples: 205,283
Train Period:  2008-2013
Val Period:    2014-2015
Test Period:   2015-2016
Test Samples:  37,900
"""
ax2.text(0.05, 0.5, dataset_text, transform=ax2.transAxes, 
         fontsize=11, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
ax2.axis('off')

# Middle: Sharpe ratio comparison
ax3 = fig.add_subplot(gs[1, :])
sorted_data = all_results.sort_values('sharpe_ratio', ascending=True)
colors = ['red' if x < 0 else 'orange' if x < 0.5 else 'green' 
          for x in sorted_data['sharpe_ratio']]
ax3.barh(range(len(sorted_data)), sorted_data['sharpe_ratio'], color=colors, alpha=0.7)
ax3.set_yticks(range(len(sorted_data)))
ax3.set_yticklabels(sorted_data['model'], fontsize=10)
ax3.set_xlabel('Sharpe Ratio', fontsize=11, fontweight='bold')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax3.grid(axis='x', alpha=0.3)
for i, (_, row) in enumerate(sorted_data.iterrows()):
    value = row['sharpe_ratio']
    ax3.text(value + 0.05 if value > 0 else value - 0.05, i, 
            f"{value:.2f}", va='center', fontsize=9, fontweight='bold')

# Bottom left: Return distribution
ax4 = fig.add_subplot(gs[2, 0])
returns = all_results['total_return'] * 100
ax4.hist(returns, bins=6, color='steelblue', alpha=0.7, edgecolor='black')
ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
ax4.set_xlabel('Total Return (%)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Count', fontsize=10, fontweight='bold')
ax4.set_title('Return Distribution', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# Bottom middle: Loss type comparison
ax5 = fig.add_subplot(gs[2, 1])
loss_summary = all_results.groupby('loss_type')['sharpe_ratio'].agg(['mean', 'std'])
x = np.arange(len(loss_summary))
ax5.bar(x, loss_summary['mean'], yerr=loss_summary['std'], 
        color=['steelblue', 'coral'], alpha=0.7, capsize=5,
        edgecolor='black', linewidth=1.5)
ax5.set_xticks(x)
ax5.set_xticklabels(loss_summary.index, fontsize=11, fontweight='bold')
ax5.set_ylabel('Mean Sharpe Ratio', fontsize=10, fontweight='bold')
ax5.set_title('Loss Function Comparison', fontsize=11, fontweight='bold')
ax5.axhline(y=0, color='red', linestyle='--', linewidth=1)
ax5.grid(axis='y', alpha=0.3)
for i, (loss, row) in enumerate(loss_summary.iterrows()):
    ax5.text(i, row['mean'] + 0.05, f"{row['mean']:.2f}", 
            ha='center', fontweight='bold')

# Bottom right: Key statistics
ax6 = fig.add_subplot(gs[2, 2])
stats_text = f"""
📈 STATISTICS

Models Tested:    {len(all_results)}
MSE Models:       {(all_results['loss_type'] == 'MSE').sum()}
NDCG Models:      {(all_results['loss_type'] == 'NDCG').sum()}

Profitable:       {(all_results['sharpe_ratio'] > 0).sum()}
Unprofitable:     {(all_results['sharpe_ratio'] < 0).sum()}

Best Sharpe:      {all_results['sharpe_ratio'].max():.3f}
Worst Sharpe:     {all_results['sharpe_ratio'].min():.3f}
Mean Sharpe:      {all_results['sharpe_ratio'].mean():.3f}
"""
ax6.text(0.05, 0.5, stats_text, transform=ax6.transAxes, 
         fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
ax6.axis('off')

fig.suptitle('100-Stock Neural Network Experiment: Complete Overview', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(OUTPUT_DIR / "7_summary_dashboard.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 7_summary_dashboard.png")

# ============================================================================
# PLOT 8: Model Architecture Comparison
# ============================================================================
print("8. Creating architecture comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Group by base architecture
architecture_groups = {
    'Simple': ['Combined Ranker (MSE)', 'Combined Ranker (NDCG)'],
    'Late Fusion': ['Late Fusion Ranker (MSE)', 'Late Fusion Ranker (NDCG)'],
    'Deep': ['Deep Late Fusion', 'Deep Combined']
}

metrics = ['sharpe_ratio', 'total_return', 'max_drawdown', 'volatility']
titles = ['Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)', 'Volatility (%)']
axes_flat = axes.flatten()

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes_flat[idx]
    
    arch_data = []
    arch_labels = []
    arch_colors = []
    
    for arch_name, models in architecture_groups.items():
        arch_models = all_results[all_results['model'].isin(models)]
        if len(arch_models) > 0:
            values = arch_models[metric]
            if 'return' in metric or 'drawdown' in metric or 'volatility' in metric:
                values = values * 100
            arch_data.extend(values)
            # Extract loss type if present
            labels = []
            for m in arch_models['model']:
                if '(' in m:
                    loss = m.split('(')[1].strip(')')
                    labels.append(f"{arch_name}\n{loss}")
                else:
                    labels.append(arch_name)
            arch_labels.extend(labels)
            
            # Color by architecture
            if 'Simple' in arch_name:
                arch_colors.extend(['lightblue'] * len(values))
            elif 'Late Fusion' in arch_name:
                arch_colors.extend(['lightcoral'] * len(values))
            else:
                arch_colors.extend(['lightgreen'] * len(values))
    
    bars = ax.bar(range(len(arch_data)), arch_data, color=arch_colors, 
                   alpha=0.7, edgecolor='black', linewidth=1)
    ax.set_xticks(range(len(arch_labels)))
    ax.set_xticklabels(arch_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(title, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    if 'drawdown' in metric.lower():
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    
    # Add value labels
    for i, value in enumerate(arch_data):
        ax.text(i, value + (abs(value) * 0.03), f'{value:.1f}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')

fig.suptitle('100-Stock Experiment: Architecture Comparison', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "8_architecture_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: 8_architecture_comparison.png")

# ============================================================================
# Create README
# ============================================================================
print("\n9. Creating README...")

readme_content = f"""# 100-Stock Neural Network Experiment: Visualizations

**Generated:** {pd.Timestamp.now().strftime('%B %d, %Y')}
**Dataset:** 100 stocks, 205,283 samples, 2008-2016
**Test Period:** July 2015 - December 2016 (18 months)

## 🏆 Best Model

**{best_model['model']}**
- Sharpe Ratio: {best_model['sharpe_ratio']:.3f}
- Total Return: {best_model['total_return']*100:.2f}%
- Max Drawdown: {best_model['max_drawdown']*100:.2f}%
- Parameters: {param_counts.get(best_model['model'], 'N/A'):,}

## 📊 Visualizations

### 1. Sharpe Ratio Comparison (`1_sharpe_comparison.png`)
Bar chart comparing Sharpe ratios across all models tested.

### 2. Risk-Return Scatter (`2_risk_return_scatter.png`)
Scatter plot showing the trade-off between volatility and returns for each model.

### 3. Performance Metrics Heatmap (`3_metrics_heatmap.png`)
Comprehensive heatmap showing normalized scores across multiple performance metrics.

### 4. MSE vs NDCG Comparison (`4_mse_vs_ndcg.png`)
Direct comparison of MSE and NDCG loss functions on the same architectures.

### 5. Top 3 Models Analysis (`5_top3_models.png`)
Detailed breakdown of the three best-performing models.

### 6. Complexity vs Performance (`6_complexity_vs_performance.png`)
Analysis of model parameter count versus achieved Sharpe ratio.

### 7. Summary Dashboard (`7_summary_dashboard.png`)
Complete overview with key statistics and visualizations in one view.

### 8. Architecture Comparison (`8_architecture_comparison.png`)
Comparison across different neural network architectures (Simple, Late Fusion, Deep).

## 📈 Key Findings

1. **Two models tied for best performance** at 0.76 Sharpe ratio:
   - Deep Late Fusion (MSE) - 71,297 parameters
   - Combined Ranker (NDCG) - 11,393 parameters (84% fewer!)

2. **NDCG loss showed surprising effectiveness** on the Combined Ranker architecture

3. **Model complexity didn't guarantee better results** - simpler models with proper loss functions matched deep networks

4. **Test set performance was realistic** - no signs of severe overfitting

## 📁 Files

All visualizations are saved as high-resolution PNG files (300 DPI) suitable for presentations and reports.

## 🔗 Related Files

- Results: `../results/expanded_100stocks_results.csv`
- NDCG Results: `../results/ndcg_expanded_100stocks_results.csv`
- Complete Analysis: `../FINAL_100_STOCKS_SUMMARY.md`
- NDCG Deep Dive: `../NDCG_COMPLETE_ANALYSIS.md`
"""

with open(OUTPUT_DIR / "README.md", 'w') as f:
    f.write(readme_content)
print("   ✓ Saved: README.md")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("✅ VISUALIZATION GENERATION COMPLETE")
print("="*70)
print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"\nGenerated {8} visualizations + README")
print("\nFiles created:")
print("  1. 1_sharpe_comparison.png")
print("  2. 2_risk_return_scatter.png")
print("  3. 3_metrics_heatmap.png")
print("  4. 4_mse_vs_ndcg.png")
print("  5. 5_top3_models.png")
print("  6. 6_complexity_vs_performance.png")
print("  7. 7_summary_dashboard.png")
print("  8. 8_architecture_comparison.png")
print("  9. README.md")
print("\n" + "="*70)

