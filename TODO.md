# CIS 5200 Project TODO List
## Project: Get Rich Quick - Stock Movement Prediction with Multimodal Data
**Team: Schuylkill River Trading**

---

## 📅 Timeline Overview

### Phase 1: Proposal & Setup (Oct 20 - Oct 31)
### Phase 2: Data & Feature Engineering (Nov 1 - Nov 5)
### Phase 3: Model Development (Nov 6 - Nov 15)
### Phase 4: Advanced Models & Tuning (Nov 16 - Nov 20)
### Phase 5: Checkpoint (Nov 21 - Nov 24)
### Phase 6: Analysis & Interpretation (Nov 25 - Nov 30)
### Phase 7: Final Deliverables (Dec 1 - Dec 8)


---


## 👤 Rafael's Tasks

### Data Pipeline & Infrastructure
- [x] Set up yfinance data collection for DJIA stocks (using tech stock universe)
- [ ] Implement Twitter/X API scraper for financial keywords
- [ ] Set up FinBERT model and sentiment scoring pipeline
- [x] Ensure point-in-time data integrity (no look-ahead bias)
- [x] Create data preprocessing and feature alignment system
- [ ] Merge sentiment features with market data by date

### Evaluation Framework
- [x] Implement walk-forward validation splits
- [x] Build metric calculation system (Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss)
- [ ] Implement ranking metrics (Spearman, Kendall-τ, NDCG)
- [x] Create backtesting framework for top-k trading strategy
- [x] Build system to track cumulative returns, Sharpe ratio, max drawdown
- [x] Set up evaluation pipeline for all models

### Neural Network
- [x] Design initial multimodal neural network architecture
- [x] Implement separate subnetworks for numerical and textual features (Late Fusion)
- [x] Implement differentiable ranking loss (ListNet/NDCG)
- [x] Add auxiliary BCE head with light weighting (classification models)
- [x] Implement L2 regularization and early stopping
- [x] Add daily feature normalization (z-scoring)
- [x] Tune hyperparameters and validate on ranking metrics

### Writing (Final Report)
- [ ] Write Dataset section
- [ ] Write Evaluation section
- [ ] Write Conclusion section
- [ ] Contribute to Abstract
- [ ] Contribute to Motivation
- [ ] Final editing pass

---

## 👤 Monica's Tasks

### Model Development - Baselines
- [ ] Implement Logistic Regression baseline with L2 regularization
- [ ] Implement binary cross-entropy loss function
- [ ] Train and tune Logistic Regression model
- [ ] Evaluate Logistic Regression on test set
- [ ] Implement Random Forest classifier
- [ ] Tune Random Forest hyperparameters (n_estimators, max_depth, min_samples_leaf)
- [ ] Train and evaluate Random Forest model
- [ ] Compare RF vs Logistic Regression performance

### Loss Functions & Metrics Design
- [ ] Help design custom loss functions for classification task
- [ ] Collaborate on defining model comparison metrics
- [ ] Assist with probability calibration analysis
- [ ] Create calibration plots and Brier score calculations

### Neural Network Collaboration
- [ ] Test and validate neural network outputs
- [ ] Help tune neural network hyperparameters
- [ ] Contribute to model ensemble strategy

### Writing (Final Report)
- [ ] Write Problem Formulation section
- [ ] Write Methods section (Logistic Regression + Random Forest)
- [ ] Contribute to Abstract
- [ ] Contribute to Motivation
- [ ] Final editing pass

---

## 👤 Kylie's Tasks

### Model Development - Advanced Models
- [x] Implement XGBoost classifier with binary:logistic objective
- [ ] Configure early stopping on validation set
- [ ] Tune XGBoost hyperparameters (initial defaults set, needs tuning)
- [ ] Map XGBoost probabilities to confidence buckets (Strong Up, Up, Neutral, Down, Strong Down)
- [x] Train and evaluate XGBoost model
- [x] Implement ranking model formulation
- [x] Train ranking model with Spearman/Kendall loss (using rank:pairwise objective)
- [x] Evaluate ranking model performance

### Interpretability & Analysis
- [x] Compute feature importances for tree-based models
- [ ] Generate SHAP values for XGBoost and Random Forest
- [ ] Create visualizations of feature importance
- [ ] Plot correlations between sentiment and next-day returns (waiting on sentiment data)
- [ ] Analyze keyword-level effects (e.g., "beat", "miss", "recall")
- [ ] Create ranking stability plots (Spearman correlation over time)
- [x] Generate ROC and precision-recall curves (ROC done, PR curve pending)
- [ ] Create calibration plots

### Backtesting Support
- [x] Help Rafael with backtesting analysis
- [x] Analyze strategy performance metrics
- [x] Create performance visualization plots (equity curve, daily returns, confusion matrix, ROC)

### Neural Network Collaboration
- [ ] Test and validate neural network outputs
- [ ] Help tune neural network hyperparameters
- [ ] Contribute to model ensemble strategy

### Writing (Final Report)
- [ ] Write Methods section (XGBoost + Ranking Model)
- [ ] Write Related Work section
- [ ] Create all visualizations for report
- [ ] Contribute to Abstract
- [ ] Contribute to Motivation
- [ ] Final editing pass

---

## 👥 Shared Tasks (All Team Members)

### Model Ensemble & Integration
- [ ] Collaborate on improving the neural network architecture
- [ ] Implement model output combining strategies
- [ ] Test static ensembling (weighted average, stacking)
- [ ] Implement adaptive weighting scheme
- [ ] Compare ensemble performance

### Ablation Studies
- [x] Run models with numerical features only (XGBoost classifier & ranker complete)
- [ ] Run models with textual (FinBERT) sentiment only
- [ ] Run models with combined multimodal features
- [ ] Compare and analyze ablation results

### Checkpoint Report (Due Nov 21-24)
- [ ] Draft checkpoint report collaboratively
- [ ] Include preliminary results and analysis
- [ ] Review and edit checkpoint together
- [ ] Submit checkpoint report

### Final Presentation (Dec 4-5)
- [ ] Create presentation slides (Dec 1-3)
- [ ] Prepare demo/visualizations
- [ ] Rehearse presentation
- [ ] Present during recitation session

### Final Report & Code (Due Dec 6-8)
- [ ] Integrate all written sections
- [ ] Proofread and edit entire report
- [ ] Create final Jupyter notebook with all experiments
- [ ] Clean and comment code
- [ ] Write comprehensive README
- [ ] Submit final report and notebook

---

## 📊 Milestones Checklist

- [x] **Oct 27**: Proposal submitted ✓
- [x] **Oct 31**: TA check-in completed, data pipeline initialized ✓
- [x] **Nov 5**: Feature engineering complete ✓ (technical features from price data)
- [ ] **Nov 10**: Baseline models (LR, RF) trained
- [ ] **Nov 15**: Neural network baseline complete, all metrics tested
- [x] **Nov 20**: Model tuning and ranking experiments complete ✓ (XGBoost models done)
- [x] **Nov 24**: Checkpoint report submitted ✓
- [ ] **Nov 30**: Interpretability analysis complete (in progress)
- [ ] **Dec 3**: Presentation slides ready
- [ ] **Dec 5**: Presentation delivered
- [ ] **Dec 8**: Final report and code submitted

---

## 📝 Notes

### Dataset Sources
- Yahoo Finance (yfinance) - DJIA daily market data (2008-2016)
- Kaggle "Daily News for Stock Prediction" - Top 25 headlines per day
- Twitter/X API - Financial keywords and sentiment
- FinBERT - Pre-trained financial sentiment model

### Key Technical Details
- Use walk-forward validation (train on earlier, test on later)
- No look-ahead bias in feature engineering
- Daily feature normalization for cross-sectional patterns
- Binary classification: predict up (1) or down (0)
- Ranking: order stocks by expected performance

### Communication
- Regular team check-ins to sync progress
- Share code via Git repository
- Document experiments and findings
- Collaborate on writing using shared LaTeX/Overleaf document

---

## 📈 Current Status (as of Dec 2, 2025)

### ✅ Completed
- **Data Infrastructure**: Price data collection via yfinance with caching system
- **Feature Engineering**: Technical indicators (returns, momentum, volatility, cross-sectional ranking)
- **Sentiment Analysis**: FinBERT processing of 123K news headlines (2008-2016)
- **Multimodal Dataset**: Combined price + sentiment features (34,612 samples, 17 stocks)
- **XGBoost Classifier**: Binary classification model predicting up/down movements
- **XGBoost Ranker**: Ranking model using pairwise objective for stock selection
- **Neural Networks**: 7 models (3 classifiers + 4 rankers) with different architectures
  - Price-Only, Combined, Late Fusion architectures
  - Cross-Entropy, MSE, and NDCG/ListNet losses
  - **Best result: 61.9% return, 1.58 Sharpe ratio (Late Fusion MSE)**
- **Evaluation Pipeline**: Time-based validation, comprehensive metrics
- **Backtesting Framework**: Long/short strategy with Sharpe ratio and cumulative returns
- **Visualizations**: ROC curve, confusion matrix, equity curve, daily return distribution
- **Analysis Pipeline**: Complete analysis script with metrics and plots

### 🚧 In Progress
- Hyperparameter tuning for XGBoost models
- Additional interpretability analysis (SHAP values)
- Baseline models (Logistic Regression, Random Forest)
- Neural network visualization notebooks

### ⏳ Next Steps
- **Add sentiment data**: Twitter/X API scraper and FinBERT sentiment scoring
- **Multimodal features**: Combine technical indicators with textual sentiment
- **Neural network**: Implement multimodal architecture
- **Ablation studies**: Compare numerical-only vs sentiment-only vs combined
- **Final report**: Complete all sections and create comprehensive documentation

### ⚠️ Note
Currently working with **financial price data only** (no sentiment/textual data yet).

