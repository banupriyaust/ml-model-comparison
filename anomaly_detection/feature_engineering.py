"""
Feature engineering for anomaly detection.
Transforms raw claims DataFrame into 28 numeric features across 6 groups:
  A. Payment (5)   B. Temporal (8)   C. Diagnosis (4)
  D. Provider (5)  E. Member (3)     F. Categorical (3)
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Transform raw claims DataFrame into feature matrix.
    Returns: (X_array, feature_names, df_with_features)
    """
    df = df.copy()

    # --- Parse dates ---
    for col in ["clm_from_dt", "clm_thru_dt"]:
        df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
    for col in ["submission_date", "review_start_date", "resolution_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # ===== GROUP A: Payment Features (5) =====
    df["clm_pmt_amt_filled"] = pd.to_numeric(df["clm_pmt_amt"], errors="coerce").fillna(0)
    df["approved_amount_filled"] = pd.to_numeric(df["approved_amount"], errors="coerce").fillna(0)
    df["payment_approved_ratio"] = df["clm_pmt_amt_filled"] / (df["approved_amount_filled"].abs() + 1)
    df["payment_approved_diff"] = df["clm_pmt_amt_filled"] - df["approved_amount_filled"]
    df["is_negative_payment"] = (df["clm_pmt_amt_filled"] < 0).astype(np.float32)

    # ===== GROUP B: Temporal Features (8) =====
    df["claim_duration_days"] = (df["clm_thru_dt"] - df["clm_from_dt"]).dt.days.fillna(0).astype(np.float32)
    df["submission_to_service_days"] = (
        (df["submission_date"] - df["clm_from_dt"]).dt.days.fillna(0).astype(np.float32)
    )
    df["review_duration_days"] = (
        (df["resolution_date"] - df["review_start_date"]).dt.days.fillna(0).astype(np.float32)
    )
    df["submission_to_resolution_days"] = (
        (df["resolution_date"] - df["submission_date"]).dt.days.fillna(0).astype(np.float32)
    )
    df["processing_days"] = pd.to_numeric(df["estimated_processing_days"], errors="coerce").fillna(14)

    month = df["clm_from_dt"].dt.month.fillna(6)
    df["claim_month_sin"] = np.sin(2 * np.pi * month / 12).astype(np.float32)
    df["claim_month_cos"] = np.cos(2 * np.pi * month / 12).astype(np.float32)
    df["claim_day_of_week"] = df["submission_date"].dt.dayofweek.fillna(2).astype(np.float32)

    # ===== GROUP C: Diagnosis Complexity Features (4) =====
    diag_cols = [f"icd9_dgns_cd_{i}" for i in range(1, 11)]
    existing_diag_cols = [c for c in diag_cols if c in df.columns]
    df["diagnosis_count"] = df[existing_diag_cols].notna().sum(axis=1).astype(np.float32)
    df["has_secondary_diag"] = df["icd9_dgns_cd_2"].notna().astype(np.float32) if "icd9_dgns_cd_2" in df.columns else 0

    # Frequency encoding for primary diagnosis
    if "icd9_dgns_cd_1" in df.columns:
        icd9_freq = df["icd9_dgns_cd_1"].value_counts()
        df["primary_diag_frequency"] = df["icd9_dgns_cd_1"].map(icd9_freq).fillna(0).astype(np.float32)

        # Rarity score: average inverse-frequency of first 5 diagnosis codes
        rarity_cols = []
        for i in range(1, 6):
            col = f"icd9_dgns_cd_{i}"
            if col in df.columns:
                freq = df[col].value_counts()
                rc = f"_diag_inv_freq_{i}"
                df[rc] = df[col].map(lambda x, f=freq: 1.0 / f.get(x, 1) if pd.notna(x) else 0).astype(np.float32)
                rarity_cols.append(rc)
        df["diag_code_rarity_score"] = df[rarity_cols].mean(axis=1) if rarity_cols else 0
    else:
        df["primary_diag_frequency"] = 0
        df["diag_code_rarity_score"] = 0

    # ===== GROUP D: Provider/Physician Pattern Features (5) =====
    if "prvdr_num" in df.columns:
        prvdr_vol = df["prvdr_num"].value_counts()
        df["provider_claim_volume"] = df["prvdr_num"].map(prvdr_vol).fillna(0).astype(np.float32)

        # Provider average payment
        prvdr_avg_pay = df.groupby("prvdr_num")["clm_pmt_amt_filled"].transform("mean")
        df["provider_avg_payment"] = prvdr_avg_pay.fillna(0).astype(np.float32)

        # Provider category entropy
        def _category_entropy(group):
            probs = group.value_counts(normalize=True)
            return -(probs * np.log2(probs + 1e-10)).sum()

        if "claim_category" in df.columns:
            prvdr_entropy = df.groupby("prvdr_num")["claim_category"].transform(
                lambda g: _category_entropy(g)
            )
            df["provider_category_entropy"] = prvdr_entropy.fillna(0).astype(np.float32)
        else:
            df["provider_category_entropy"] = 0
    else:
        df["provider_claim_volume"] = 0
        df["provider_avg_payment"] = 0
        df["provider_category_entropy"] = 0

    if "at_physn_npi" in df.columns:
        phys_vol = df["at_physn_npi"].value_counts()
        df["physician_claim_volume"] = df["at_physn_npi"].map(phys_vol).fillna(0).astype(np.float32)

        # Physician-provider ratio
        phys_prov = df.groupby("at_physn_npi")["prvdr_num"].transform("nunique") if "prvdr_num" in df.columns else 1
        df["physician_provider_ratio"] = pd.to_numeric(phys_prov, errors="coerce").fillna(1).astype(np.float32)
    else:
        df["physician_claim_volume"] = 0
        df["physician_provider_ratio"] = 1

    # ===== GROUP E: Member/Patient Behavior Features (3) =====
    if "desynpuf_id" in df.columns:
        member_freq = df["desynpuf_id"].value_counts()
        df["member_claim_frequency"] = df["desynpuf_id"].map(member_freq).fillna(0).astype(np.float32)

        member_avg_pay = df.groupby("desynpuf_id")["clm_pmt_amt_filled"].transform("mean")
        df["member_avg_payment"] = member_avg_pay.fillna(0).astype(np.float32)

        if "prvdr_num" in df.columns:
            member_providers = df.groupby("desynpuf_id")["prvdr_num"].transform("nunique")
            df["member_unique_providers"] = pd.to_numeric(member_providers, errors="coerce").fillna(1).astype(np.float32)
        else:
            df["member_unique_providers"] = 1
    else:
        df["member_claim_frequency"] = 0
        df["member_avg_payment"] = 0
        df["member_unique_providers"] = 1

    # ===== GROUP F: Categorical Encoded Features (3) =====
    urgency_map = {"routine": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    df["urgency_ordinal"] = df["urgency_level"].map(urgency_map).fillna(1).astype(np.float32) if "urgency_level" in df.columns else 0

    # One-hot encode claim_category
    if "claim_category" in df.columns:
        dummies = pd.get_dummies(df["claim_category"], prefix="cat").astype(np.float32)
        for expected_col in ["cat_carrier", "cat_inpatient", "cat_outpatient", "cat_prescription"]:
            if expected_col not in dummies.columns:
                dummies[expected_col] = 0
        df = pd.concat([df, dummies[["cat_carrier", "cat_inpatient", "cat_outpatient", "cat_prescription"]]], axis=1)
    else:
        for c in ["cat_carrier", "cat_inpatient", "cat_outpatient", "cat_prescription"]:
            df[c] = 0

    df["appeal_flag_binary"] = (df["appeal_flag"].astype(str).str.lower() == "true").astype(np.float32) if "appeal_flag" in df.columns else 0

    # ===== Assemble Feature Matrix =====
    feature_names = [
        # A: Payment (5)
        "clm_pmt_amt_filled", "approved_amount_filled",
        "payment_approved_ratio", "payment_approved_diff", "is_negative_payment",
        # B: Temporal (8)
        "claim_duration_days", "submission_to_service_days",
        "review_duration_days", "submission_to_resolution_days",
        "processing_days", "claim_month_sin", "claim_month_cos", "claim_day_of_week",
        # C: Diagnosis (4)
        "diagnosis_count", "primary_diag_frequency",
        "has_secondary_diag", "diag_code_rarity_score",
        # D: Provider (5)
        "provider_claim_volume", "physician_claim_volume",
        "provider_avg_payment", "provider_category_entropy", "physician_provider_ratio",
        # E: Member (3)
        "member_claim_frequency", "member_avg_payment", "member_unique_providers",
        # F: Categorical (3)
        "urgency_ordinal", "appeal_flag_binary",
        "cat_carrier", "cat_inpatient", "cat_outpatient", "cat_prescription",
    ]

    X = df[feature_names].values.astype(np.float32)

    # Replace any remaining NaN or inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"Engineered {len(feature_names)} features from {len(df)} claims")
    return X, feature_names, df
