"""
Synthetic anomaly injection for evaluation.
Creates 5 types of anomalies modeled after real healthcare fraud patterns.
"""

import numpy as np
from anomaly_detection.config import SYNTHETIC_ANOMALY_FRACTION, RANDOM_STATE


# Feature name -> index mapping (must match feature_engineering.py output order)
FEATURE_INDEX = {
    "clm_pmt_amt_filled": 0,
    "approved_amount_filled": 1,
    "payment_approved_ratio": 2,
    "payment_approved_diff": 3,
    "is_negative_payment": 4,
    "claim_duration_days": 5,
    "submission_to_service_days": 6,
    "review_duration_days": 7,
    "submission_to_resolution_days": 8,
    "processing_days": 9,
    "diagnosis_count": 13,
    "primary_diag_frequency": 14,
    "provider_claim_volume": 17,
    "provider_avg_payment": 19,
    "member_claim_frequency": 22,
}


def inject_synthetic_anomalies(
    X_test: np.ndarray,
    feature_names: list[str],
    anomaly_fraction: float = SYNTHETIC_ANOMALY_FRACTION,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Inject synthetic anomalies into the test set.

    Returns:
        X_contaminated: test data with anomalies appended
        y_labels: 0=normal, 1=anomaly
        anomaly_types: string labels for each anomaly type (empty for normal)
    """
    rng = np.random.RandomState(random_state)
    n_anomalies = int(len(X_test) * anomaly_fraction)

    # Build index map from actual feature names
    fidx = {name: i for i, name in enumerate(feature_names)}

    # Select random rows to clone and perturb
    indices = rng.choice(len(X_test), n_anomalies, replace=False)
    X_anomalies = X_test[indices].copy()

    # Distribute across 5 types
    type_counts = _split_counts(n_anomalies, [0.25, 0.20, 0.15, 0.20, 0.20])
    type_labels = []
    offset = 0

    # Type 1: Extreme Payment (25%)
    n = type_counts[0]
    subset = X_anomalies[offset : offset + n]
    col = fidx.get("clm_pmt_amt_filled", 0)
    multipliers = rng.uniform(5, 20, size=n)
    subset[:, col] = np.abs(subset[:, col]) * multipliers + rng.uniform(1000, 5000, size=n)
    if "payment_approved_ratio" in fidx:
        subset[:, fidx["payment_approved_ratio"]] = subset[:, col] / (np.abs(subset[:, fidx.get("approved_amount_filled", 1)]) + 1)
    if "payment_approved_diff" in fidx:
        subset[:, fidx["payment_approved_diff"]] = subset[:, col] - subset[:, fidx.get("approved_amount_filled", 1)]
    type_labels.extend(["extreme_payment"] * n)
    offset += n

    # Type 2: Payment-Approval Mismatch (20%)
    n = type_counts[1]
    subset = X_anomalies[offset : offset + n]
    if "payment_approved_diff" in fidx:
        subset[:, fidx["payment_approved_diff"]] = rng.uniform(500, 5000, size=n)
    if "payment_approved_ratio" in fidx:
        subset[:, fidx["payment_approved_ratio"]] = rng.uniform(3, 15, size=n)
    type_labels.extend(["payment_mismatch"] * n)
    offset += n

    # Type 3: Impossible Duration (15%)
    n = type_counts[2]
    subset = X_anomalies[offset : offset + n]
    if "claim_duration_days" in fidx:
        subset[:, fidx["claim_duration_days"]] = rng.choice(
            [0, rng.randint(300, 730)], size=n, p=[0.5, 0.5]
        ).astype(np.float32)
    if "submission_to_resolution_days" in fidx:
        subset[:, fidx["submission_to_resolution_days"]] = rng.uniform(200, 500, size=n)
    type_labels.extend(["impossible_duration"] * n)
    offset += n

    # Type 4: Diagnosis Stuffing (20%)
    n = type_counts[3]
    subset = X_anomalies[offset : offset + n]
    if "diagnosis_count" in fidx:
        subset[:, fidx["diagnosis_count"]] = 10
    if "diag_code_rarity_score" in fidx:
        subset[:, fidx["diag_code_rarity_score"]] = rng.uniform(0.5, 1.0, size=n)
    if "primary_diag_frequency" in fidx:
        subset[:, fidx["primary_diag_frequency"]] = rng.uniform(0, 5, size=n)
    type_labels.extend(["diagnosis_stuffing"] * n)
    offset += n

    # Type 5: Provider Anomaly (20%)
    n = type_counts[4]
    subset = X_anomalies[offset : offset + n]
    if "provider_claim_volume" in fidx:
        subset[:, fidx["provider_claim_volume"]] = rng.uniform(1, 3, size=n)
    if "provider_avg_payment" in fidx:
        subset[:, fidx["provider_avg_payment"]] = rng.uniform(5000, 20000, size=n)
    if "clm_pmt_amt_filled" in fidx:
        subset[:, fidx["clm_pmt_amt_filled"]] = rng.uniform(3000, 15000, size=n)
    type_labels.extend(["provider_anomaly"] * n)

    # Combine
    X_contaminated = np.vstack([X_test, X_anomalies])
    y_labels = np.concatenate([np.zeros(len(X_test)), np.ones(n_anomalies)])

    all_types = ["normal"] * len(X_test) + type_labels
    anomaly_types = np.array(all_types)

    return X_contaminated, y_labels, anomaly_types


def _split_counts(total: int, fractions: list[float]) -> list[int]:
    """Split total into counts per fraction, ensuring they sum to total."""
    counts = [int(total * f) for f in fractions]
    counts[-1] = total - sum(counts[:-1])
    return counts
