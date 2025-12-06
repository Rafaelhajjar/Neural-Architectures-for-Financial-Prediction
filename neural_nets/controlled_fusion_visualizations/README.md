# Controlled Fusion Comparison Visualizations

**Date:** 2025-12-04 18:07:12
**Models:** Early Fusion (112K params) vs Late Fusion (118K params)
**Dataset:** 100 stocks, 38,000 test samples

## Results Summary

### Early Fusion
- Sharpe Ratio: 0.9634
- Total Return: 69.13%
- Spearman: -0.0205

### Late Fusion  
- Sharpe Ratio: -0.7097
- Total Return: -52.34%
- Spearman: 0.0393

## Visualizations

1. **training_curves.png** - Training and validation loss over epochs
2. **metrics_comparison.png** - Bar charts comparing key metrics
3. **equity_curves.png** - Trading performance and drawdowns
4. **prediction_quality.png** - Prediction accuracy analysis
5. **summary_dashboard.png** - Comprehensive overview

## Conclusion

This controlled experiment definitively compares early vs late fusion strategies
with matched parameter counts (~115K params each) on the same dataset.
