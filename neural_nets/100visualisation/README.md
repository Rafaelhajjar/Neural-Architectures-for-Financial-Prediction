# 100-Stock Neural Network Experiment: Visualizations

**Generated:** December 03, 2025
**Dataset:** 100 stocks, 205,283 samples, 2008-2016
**Test Period:** July 2015 - December 2016 (18 months)

## 🏆 Best Model

**Combined Ranker (NDCG)**
- Sharpe Ratio: 0.764
- Total Return: 42.92%
- Max Drawdown: -46.06%
- Parameters: 11,393

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
