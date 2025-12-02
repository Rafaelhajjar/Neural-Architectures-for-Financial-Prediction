# Neural Networks for Stock Prediction with NDCG Loss

Complete neural network implementation for multimodal stock prediction using price and sentiment features.

## 🎯 Overview

This module implements 7 neural network models for stock prediction:

### Classification Models (Binary Up/Down)
1. **Price-Only Classifier** - Uses only technical price features
2. **Combined Classifier** - Simple concatenation of price + sentiment
3. **Late Fusion Classifier** - Separate branches for each modality

### Ranking Models (Predict Returns)
4. **Combined Ranker (MSE)** - Baseline with MSE loss
5. **Combined Ranker (NDCG)** - Novel NDCG ranking loss
6. **Late Fusion Ranker (MSE)** - Baseline with MSE loss
7. **Late Fusion Ranker (NDCG)** - Novel NDCG ranking loss

## 📊 Dataset

**Source**: `data/processed/features_with_sentiment.parquet`

**Features**:
- **Price (4)**: ret_1d, momentum_126d, vol_20d, mom_rank
- **Sentiment (3)**: market_sentiment_mean, market_sentiment_std, market_news_count
- **Total**: 7 input features

**Size**: 34,612 samples (17 stocks × 2,036 days)

**Split**:
- Train: 60% (pre-2014)
- Validation: 20% (2014-2015)
- Test: 20% (2015-2016)

## 🚀 Quick Start

### Train All Models

```bash
python neural_nets/run_experiments.py
```

This trains all 7 models and saves checkpoints to `neural_nets/trained_models/`.

### Train Individual Models

**Classification:**
```bash
python neural_nets/training/train_classifier.py
```

**Ranking:**
```bash
python neural_nets/training/train_ranker.py
```

## 🏗️ Architecture

### Price-Only Net
```
Input(4) → Dense(64) → ReLU → Dropout(0.3)
        → Dense(32) → ReLU → Dropout(0.2)
        → Output(2 or 1)
```

### Combined Net
```
Input(7) → Dense(128) → ReLU → Dropout(0.3)
        → Dense(64) → ReLU → Dropout(0.2)
        → Dense(32) → ReLU
        → Output(2 or 1)
```

### Late Fusion Net
```
Price(4) → Dense(64) → ReLU → Dense(32)
                                ↓
Sentiment(3) → Dense(64) → ReLU → Dense(32)
                                ↓
                        Concat(64) → Dense(32) → Dropout(0.3)
                                              → Output(2 or 1)
```

## 💡 Novel NDCG Loss

**NDCG (Normalized Discounted Cumulative Gain)** directly optimizes ranking quality:

- Emphasizes correct ordering at top positions
- Perfect for long/short strategies (long top-5, short bottom-5)
- Formula: NDCG@k = DCG@k / IDCG@k

**Why it's better than MSE:**
- MSE: Tries to predict exact returns → cares about magnitude
- NDCG: Tries to get ranking right → cares about order

For portfolio construction, **order matters more than exact values**!

## 📁 File Structure

```
neural_nets/
├── models/
│   ├── __init__.py
│   ├── base_models.py          # Network architectures
│   └── losses.py               # NDCG and other losses
├── training/
│   ├── __init__.py
│   ├── data_loader.py          # Dataset and splits
│   ├── trainer.py              # Training loop
│   ├── train_classifier.py     # Classification training
│   └── train_ranker.py         # Ranking training
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # Evaluation metrics
│   └── evaluator.py            # Evaluation pipeline
├── notebooks/                  # Jupyter notebooks (create these)
├── trained_models/             # Saved model checkpoints
├── results/                    # Predictions and metrics
├── run_experiments.py          # Master training script
└── README.md                   # This file
```

## 📊 Evaluation Metrics

### Classification Models
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC
- Log Loss

### Ranking Models
- Spearman Correlation
- Kendall's Tau
- Information Coefficient (IC)
- MSE, MAE

### Trading Strategy
- Sharpe Ratio
- Total Return
- Max Drawdown
- Win Rate

## 🔬 Comparing NDCG vs MSE

Expected results:

| Model | Loss | Spearman | Sharpe | Notes |
|-------|------|----------|--------|-------|
| Combined Ranker | MSE | 0.15-0.20 | 0.4-0.5 | Baseline |
| Combined Ranker | NDCG | 0.18-0.25 | 0.5-0.6 | Better ranking |
| Late Fusion | MSE | 0.16-0.22 | 0.45-0.55 | Multimodal baseline |
| Late Fusion | NDCG | 0.20-0.27 | 0.55-0.65 | Best performance |

NDCG should improve:
- Top-k selection accuracy
- Sharpe ratio
- Portfolio returns

## 💻 Requirements

All dependencies are in the main `requirements.txt`:
- PyTorch >= 2.1.0
- pandas >= 2.2.0
- numpy >= 1.26.0
- scikit-learn >= 1.4.0
- scipy

## 🎓 For Your Report

This implementation demonstrates:

1. **Ablation Study**: Price-only vs Sentiment vs Combined
2. **Multimodal Learning**: Early fusion vs Late fusion
3. **Novel Loss Function**: NDCG for ranking
4. **Practical Application**: Long/short trading strategy
5. **Comprehensive Evaluation**: Multiple metrics and visualizations

## 📝 Training Configuration

**Hyperparameters:**
- Learning rate: 0.001 (Adam optimizer)
- Batch size: 256
- Max epochs: 100
- Early stopping: 20 epochs patience
- L2 regularization: 1e-5
- Learning rate scheduler: ReduceLROnPlateau

**Training time** (on CPU):
- Per model: 5-10 minutes
- All 7 models: ~60 minutes

## 🚀 Next Steps

1. **Train models**: `python neural_nets/run_experiments.py`
2. **Create evaluation notebooks**: Analyze results in Jupyter
3. **Compare with XGBoost**: Add to comparison table
4. **Generate visualizations**: For final report

## 📈 Expected Insights

- **Sentiment helps**: Combined > Price-only
- **Late fusion works**: Late fusion ≥ Early fusion
- **NDCG improves ranking**: NDCG > MSE for trading
- **Multimodal advantage**: Neural nets can leverage both modalities

Good luck with your experiments! 🎉
