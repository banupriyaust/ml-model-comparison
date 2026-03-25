"""
Traditional (non-deep-learning) anomaly detection baselines.
"""

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from anomaly_detection.config import CONTAMINATION, RANDOM_STATE


def get_traditional_models() -> dict:
    """Return dict of traditional anomaly detection models."""
    return {
        "Isolation Forest": IsolationForest(
            n_estimators=200,
            contamination=CONTAMINATION,
            max_samples=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "One-Class SVM": OneClassSVM(
            kernel="rbf",
            gamma="scale",
            nu=CONTAMINATION,
        ),
    }
