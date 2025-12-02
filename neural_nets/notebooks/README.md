# Neural Network Analysis Notebooks

This folder will contain Jupyter notebooks for analyzing neural network results.

## Recommended Notebooks to Create

### 1. `01_data_exploration.ipynb`
- Load and explore the dataset
- Visualize feature distributions
- Show train/val/test split
- Correlation analysis between features

### 2. `02_training_results.ipynb`
- Load training histories from saved models
- Plot training/validation loss curves
- Show learning rate schedules
- Analyze early stopping behavior

### 3. `03_model_comparison.ipynb`
- Compare all 7 models side-by-side
- Metrics table (accuracy, Spearman, Sharpe)
- ROC curves for classification models
- Ranking quality visualizations

### 4. `04_trading_performance.ipynb`
- Backtest all ranking models
- Plot equity curves
- Compare long/short strategies
- Analyze NDCG vs MSE performance

### 5. `05_ndcg_analysis.ipynb`
- Deep dive into NDCG loss behavior
- Compare top-k selection accuracy
- Show example daily rankings
- NDCG advantage visualization

## Quick Start

```python
import sys
from pathlib import Path
sys.path.append(str(Path().absolute().parent.parent))

import torch
import pandas as pd
from neural_nets.models import LateFusionNet
from neural_nets.training.data_loader import load_and_prepare_data, create_data_loaders
from neural_nets.evaluation import Evaluator

# Load data
train_df, val_df, test_df = load_and_prepare_data()

# Create test loader
price_features = ['ret_1d', 'momentum_126d', 'vol_20d', 'mom_rank']
sentiment_features = ['market_sentiment_mean', 'market_sentiment_std', 'market_news_count']

loaders = create_data_loaders(
    train_df, val_df, test_df,
    price_features, sentiment_features,
    task='classification',
    batch_size=256
)

# Load trained model
model = LateFusionNet(task='classification')
checkpoint = torch.load('neural_nets/trained_models/late_fusion_classifier_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
evaluator = Evaluator(model, loaders['test'], device='cpu')
predictions = evaluator.predict()
metrics = evaluator.evaluate_classification(predictions)

print(metrics)
```

## Tips

- Use `%matplotlib inline` for plots
- Save figures to `neural_nets/results/` for report
- Compare with XGBoost results from earlier experiments
- Focus on NDCG vs MSE comparison for your key finding

