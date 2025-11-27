from __future__ import annotations
import pandas as pd
from typing import List, Tuple, Dict, Any
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)


def time_based_train_test_split(
    df: pd.DataFrame,
    date_col: str,
    split_date: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: rows with date < split_date
    Test:  rows with date >= split_date
    """
    df = df.sort_values(date_col)
    split_ts = pd.to_datetime(split_date)
    train = df[df[date_col] < split_ts].copy()
    test = df[df[date_col] >= split_ts].copy()
    return train, test


def train_xgb_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
    date_col: str = "date",
    split_date: str = "2018-01-01",
    xgb_params: Dict[str, Any] | None = None,
):
    """
    Train an XGBoost classifier to predict up/down.

    Returns:
        model
        metrics dict
        test_df with columns: date, ticker, y_true, y_proba
    """
    if xgb_params is None:
        xgb_params = dict(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
        )

    train_df, test_df = time_based_train_test_split(df, date_col, split_date)

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    model = XGBClassifier(**xgb_params)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba)),
    }

    out = test_df.copy()
    out["y_true"] = y_test
    out["y_proba"] = y_proba

    return model, metrics, out
