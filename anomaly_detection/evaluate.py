"""
Evaluation module for anomaly detection models.
Computes reconstruction errors, thresholds, and metrics against synthetic anomalies.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
from anomaly_detection.config import DEFAULT_THRESHOLD_PERCENTILE


def compute_reconstruction_errors(model, X: np.ndarray, is_cnn: bool = False) -> np.ndarray:
    """Compute per-sample MSE reconstruction error for an autoencoder."""
    if is_cnn:
        X_input = X.reshape(X.shape[0], X.shape[1], 1)
        X_reconstructed = model.predict(X_input, verbose=0)
        X_reconstructed = X_reconstructed.reshape(X.shape)
    else:
        X_reconstructed = model.predict(X, verbose=0)
    mse = np.mean((X - X_reconstructed) ** 2, axis=1)
    mse = np.nan_to_num(mse, nan=0.0, posinf=1e6, neginf=0.0)
    return mse


def compute_anomaly_scores_traditional(model, X: np.ndarray) -> np.ndarray:
    """
    Compute anomaly scores for traditional models.
    Returns scores where higher = more anomalous (invert decision_function).
    """
    scores = -model.decision_function(X)  # Negate: sklearn returns lower = more anomalous
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
    return scores


def determine_threshold(errors: np.ndarray, percentile: float = DEFAULT_THRESHOLD_PERCENTILE) -> float:
    """Set anomaly threshold at given percentile of reconstruction errors."""
    return np.percentile(errors, percentile)


def evaluate_at_thresholds(
    anomaly_scores: np.ndarray,
    y_true: np.ndarray,
    normal_scores: np.ndarray,
    thresholds_pct: list[float] = None,
) -> pd.DataFrame:
    """
    Evaluate anomaly detection at multiple thresholds.
    Thresholds are computed as percentiles of normal_scores (validation/train errors).
    """
    if thresholds_pct is None:
        thresholds_pct = [90.0, 95.0, 97.5, 99.0]

    results = []
    for pct in thresholds_pct:
        threshold = np.percentile(normal_scores, pct)
        clean_scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=0.0)
        y_pred = (clean_scores > threshold).astype(int)
        results.append({
            "Threshold %": pct,
            "Threshold Value": threshold,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1 Score": f1_score(y_true, y_pred, zero_division=0),
            "Flagged": int(y_pred.sum()),
            "True Positives": int(((y_pred == 1) & (y_true == 1)).sum()),
            "False Positives": int(((y_pred == 1) & (y_true == 0)).sum()),
        })

    return pd.DataFrame(results)


def compute_auc_pr(anomaly_scores: np.ndarray, y_true: np.ndarray) -> float:
    """Compute Area Under Precision-Recall Curve."""
    scores = np.nan_to_num(anomaly_scores, nan=0.0, posinf=1e6, neginf=0.0)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
    return auc(recall_vals, precision_vals)


def build_model_comparison(all_results: dict, y_true: np.ndarray, normal_scores_dict: dict) -> pd.DataFrame:
    """
    Build a comparison table across all models at the default threshold.

    Args:
        all_results: {model_name: anomaly_scores_array}
        y_true: ground truth labels
        normal_scores_dict: {model_name: scores_on_normal_data}
    """
    rows = []
    for model_name, scores in all_results.items():
        normal_scores = normal_scores_dict[model_name]
        threshold = np.percentile(normal_scores, DEFAULT_THRESHOLD_PERCENTILE)
        y_pred = (scores > threshold).astype(int)
        auc_pr = compute_auc_pr(scores, y_true)

        rows.append({
            "Model": model_name,
            "AUC-PR": auc_pr,
            f"Precision@{DEFAULT_THRESHOLD_PERCENTILE}%": precision_score(y_true, y_pred, zero_division=0),
            f"Recall@{DEFAULT_THRESHOLD_PERCENTILE}%": recall_score(y_true, y_pred, zero_division=0),
            f"F1@{DEFAULT_THRESHOLD_PERCENTILE}%": f1_score(y_true, y_pred, zero_division=0),
            "Anomalies Flagged": int(y_pred.sum()),
        })

    return pd.DataFrame(rows).sort_values("AUC-PR", ascending=False)
