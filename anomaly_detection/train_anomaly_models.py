"""
Main training orchestrator for anomaly detection.
Trains all models, evaluates with synthetic anomalies, and exports results.

Usage: python -m anomaly_detection.train_anomaly_models
"""

import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve

from anomaly_detection.config import (
    BATCH_SIZE, ENCODING_DIM, EPOCHS, LATENT_DIM, PATIENCE,
    RANDOM_STATE, RESULTS_DIR, TEST_SIZE, VALIDATION_SIZE, COLORS,
    DEFAULT_THRESHOLD_PERCENTILE,
)
from anomaly_detection.data_loader import load_claims_sample
from anomaly_detection.evaluate import (
    build_model_comparison, compute_anomaly_scores_traditional,
    compute_auc_pr, compute_reconstruction_errors,
    determine_threshold, evaluate_at_thresholds,
)
from anomaly_detection.feature_engineering import engineer_features
from anomaly_detection.models.autoencoder import build_autoencoder
from anomaly_detection.models.cnn_autoencoder import build_cnn_autoencoder
from anomaly_detection.models.traditional import get_traditional_models
from anomaly_detection.models.vae import build_vae
from anomaly_detection.synthetic_anomalies import inject_synthetic_anomalies

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")


def main():
    print("=" * 70)
    print("HEALTHCARE CLAIMS ANOMALY DETECTION")
    print("=" * 70)

    # ─── Step 1: Load Data ───
    print("\n[1/8] Loading claims data...")
    df = load_claims_sample()
    print(f"  Loaded {len(df):,} claims")

    # ─── Step 2: Feature Engineering ───
    print("\n[2/8] Engineering features...")
    X, feature_names, df_features = engineer_features(df)
    print(f"  {len(feature_names)} features: {feature_names}")

    # Save feature names
    with open(RESULTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # ─── Step 3: Scale and Split ───
    print("\n[3/8] Scaling and splitting data...")
    scaler = StandardScaler()

    X_train_full, X_test, idx_train, idx_test = train_test_split(
        X, np.arange(len(X)), test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    X_train, X_val, idx_tr, idx_val = train_test_split(
        X_train_full, idx_train, test_size=VALIDATION_SIZE, random_state=RANDOM_STATE
    )

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, RESULTS_DIR / "scaler.joblib")
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # Callbacks for DL models
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=0
        ),
    ]

    input_dim = X_train_scaled.shape[1]
    all_scores_test = {}
    all_scores_val = {}
    training_times = {}

    # ─── Step 4: Train Deep Learning Models ───
    print("\n[4/8] Training deep learning models...")

    # 4a: Vanilla Autoencoder
    print("\n  --- Autoencoder ---")
    t0 = time.time()
    ae_model = build_autoencoder(input_dim, ENCODING_DIM)
    ae_model.fit(
        X_train_scaled, X_train_scaled,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val_scaled, X_val_scaled),
        callbacks=callbacks, verbose=0,
    )
    training_times["Autoencoder"] = time.time() - t0
    ae_model.save(RESULTS_DIR / "autoencoder.keras")

    ae_val_errors = compute_reconstruction_errors(ae_model, X_val_scaled)
    ae_test_errors = compute_reconstruction_errors(ae_model, X_test_scaled)
    all_scores_val["Autoencoder"] = ae_val_errors
    all_scores_test["Autoencoder"] = ae_test_errors
    print(f"  Trained in {training_times['Autoencoder']:.1f}s | Val MSE: {ae_val_errors.mean():.6f}")

    # 4b: Variational Autoencoder
    print("\n  --- VAE ---")
    t0 = time.time()
    vae_model, vae_encoder, vae_decoder = build_vae(input_dim, LATENT_DIM)
    vae_model.fit(
        X_train_scaled, None,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val_scaled, None),
        callbacks=callbacks, verbose=0,
    )
    training_times["VAE"] = time.time() - t0
    vae_model.save(RESULTS_DIR / "vae.keras")

    vae_val_errors = compute_reconstruction_errors(vae_model, X_val_scaled)
    vae_test_errors = compute_reconstruction_errors(vae_model, X_test_scaled)
    all_scores_val["VAE"] = vae_val_errors
    all_scores_test["VAE"] = vae_test_errors
    print(f"  Trained in {training_times['VAE']:.1f}s | Val MSE: {vae_val_errors.mean():.6f}")

    # 4c: CNN Autoencoder
    print("\n  --- CNN Autoencoder ---")
    t0 = time.time()
    cnn_model = build_cnn_autoencoder(input_dim, ENCODING_DIM)
    X_train_cnn = X_train_scaled.reshape(-1, input_dim, 1)
    X_val_cnn = X_val_scaled.reshape(-1, input_dim, 1)
    cnn_model.fit(
        X_train_cnn, X_train_cnn,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_val_cnn, X_val_cnn),
        callbacks=callbacks, verbose=0,
    )
    training_times["CNN Autoencoder"] = time.time() - t0
    cnn_model.save(RESULTS_DIR / "cnn_autoencoder.keras")

    cnn_val_errors = compute_reconstruction_errors(cnn_model, X_val_scaled, is_cnn=True)
    cnn_test_errors = compute_reconstruction_errors(cnn_model, X_test_scaled, is_cnn=True)
    all_scores_val["CNN Autoencoder"] = cnn_val_errors
    all_scores_test["CNN Autoencoder"] = cnn_test_errors
    print(f"  Trained in {training_times['CNN Autoencoder']:.1f}s | Val MSE: {cnn_val_errors.mean():.6f}")

    # ─── Step 5: Train Traditional Models ───
    print("\n[5/8] Training traditional models...")
    traditional_models = get_traditional_models()

    for name, model in traditional_models.items():
        print(f"\n  --- {name} ---")
        t0 = time.time()

        if name == "One-Class SVM" and len(X_train_scaled) > 50000:
            # Subsample for SVM (memory constraint)
            rng = np.random.RandomState(RANDOM_STATE)
            svm_idx = rng.choice(len(X_train_scaled), 50000, replace=False)
            model.fit(X_train_scaled[svm_idx])
            print("  (Subsampled to 50,000 for SVM)")
        else:
            model.fit(X_train_scaled)

        training_times[name] = time.time() - t0
        joblib.dump(model, RESULTS_DIR / f"{name.lower().replace(' ', '_')}.joblib")

        val_scores = compute_anomaly_scores_traditional(model, X_val_scaled)
        test_scores = compute_anomaly_scores_traditional(model, X_test_scaled)
        all_scores_val[name] = val_scores
        all_scores_test[name] = test_scores
        print(f"  Trained in {training_times[name]:.1f}s")

    # ─── Step 6: Inject Synthetic Anomalies and Evaluate ───
    print("\n[6/8] Injecting synthetic anomalies and evaluating...")
    X_contaminated, y_true, anomaly_types = inject_synthetic_anomalies(
        X_test_scaled, feature_names
    )

    # Compute scores on contaminated set
    all_scores_contaminated = {}
    for name in all_scores_test:
        if name in ["Autoencoder", "VAE"]:
            model = ae_model if name == "Autoencoder" else vae_model
            all_scores_contaminated[name] = compute_reconstruction_errors(model, X_contaminated)
        elif name == "CNN Autoencoder":
            all_scores_contaminated[name] = compute_reconstruction_errors(
                cnn_model, X_contaminated, is_cnn=True
            )
        else:
            trad_model = joblib.load(RESULTS_DIR / f"{name.lower().replace(' ', '_')}.joblib")
            all_scores_contaminated[name] = compute_anomaly_scores_traditional(trad_model, X_contaminated)

    # Build comparison table
    comparison_df = build_model_comparison(all_scores_contaminated, y_true, all_scores_val)
    comparison_df["Training Time (s)"] = comparison_df["Model"].map(training_times)
    comparison_df.to_csv(RESULTS_DIR / "anomaly_model_metrics.csv", index=False)
    print("\n  Model Comparison:")
    print(comparison_df.to_string(index=False))

    # Per-model detailed evaluation at multiple thresholds
    for name, scores in all_scores_contaminated.items():
        detail_df = evaluate_at_thresholds(scores, y_true, all_scores_val[name])
        detail_df.to_csv(RESULTS_DIR / f"evaluation_{name.lower().replace(' ', '_')}.csv", index=False)

    # ─── Step 7: Save Reconstruction Errors and Top Anomalies ───
    print("\n[7/8] Saving results...")

    # Save reconstruction errors for dashboard histograms
    errors_dict = {name: scores for name, scores in all_scores_test.items()}
    np.savez(RESULTS_DIR / "reconstruction_errors.npz", **errors_dict)

    # Top anomalous claims (from test set, using autoencoder scores)
    ae_threshold = determine_threshold(ae_val_errors)
    anomaly_mask = ae_test_errors > ae_threshold
    test_df = df_features.iloc[idx_test].copy()
    test_df["anomaly_score_ae"] = ae_test_errors
    test_df["is_anomaly_ae"] = anomaly_mask.astype(int)

    # Add scores from all models
    for name, scores in all_scores_test.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        test_df[f"score_{safe_name}"] = scores
        threshold = determine_threshold(all_scores_val[name])
        test_df[f"flag_{safe_name}"] = (scores > threshold).astype(int)

    # Consensus: how many models flag it
    flag_cols = [c for c in test_df.columns if c.startswith("flag_")]
    test_df["consensus_count"] = test_df[flag_cols].sum(axis=1)

    # Sort by consensus and autoencoder score
    top_anomalies = (
        test_df.sort_values(["consensus_count", "anomaly_score_ae"], ascending=[False, False])
        .head(1000)
    )

    # Select useful columns for export
    export_cols = [
        "claim_reference", "claim_category", "claim_status", "clm_pmt_amt",
        "approved_amount", "urgency_level", "prvdr_num", "icd9_dgns_cd_1",
        "anomaly_score_ae", "consensus_count",
    ] + [c for c in test_df.columns if c.startswith("score_") or c.startswith("flag_")]
    export_cols = [c for c in export_cols if c in top_anomalies.columns]
    top_anomalies[export_cols].to_csv(RESULTS_DIR / "top_anomalous_claims.csv", index=False)

    # Save anomaly type breakdown for synthetic evaluation
    type_breakdown = pd.DataFrame({
        "anomaly_type": anomaly_types,
        "y_true": y_true,
    })
    for name, scores in all_scores_contaminated.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        threshold = determine_threshold(all_scores_val[name])
        type_breakdown[f"detected_{safe_name}"] = (scores > threshold).astype(int)
    type_breakdown.to_csv(RESULTS_DIR / "anomaly_type_breakdown.csv", index=False)

    # ─── Step 8: Generate Charts ───
    print("\n[8/8] Generating charts...")
    _generate_charts(comparison_df, all_scores_test, all_scores_val,
                     all_scores_contaminated, y_true, X_test_scaled, ae_test_errors)

    print("\n" + "=" * 70)
    print("COMPLETE! Results saved to:", RESULTS_DIR)
    print("=" * 70)
    print(f"\nRun dashboard: streamlit run anomaly_detection/anomaly_dashboard.py")


def _generate_charts(comparison_df, all_scores_test, all_scores_val,
                     all_scores_contaminated, y_true, X_test_scaled, ae_test_errors):
    """Generate all PNG charts."""

    # Chart 1: Model comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    models = comparison_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25
    metrics = [f"Precision@{DEFAULT_THRESHOLD_PERCENTILE}%",
               f"Recall@{DEFAULT_THRESHOLD_PERCENTILE}%",
               f"F1@{DEFAULT_THRESHOLD_PERCENTILE}%"]
    for i, metric in enumerate(metrics):
        vals = comparison_df[metric].values
        ax.bar(x + i * width, vals, width, label=metric.split("@")[0], color=COLORS[i])
        for j, v in enumerate(vals):
            ax.text(x[j] + i * width, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Anomaly Detection Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close()

    # Chart 2: Reconstruction error distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    dl_models = ["Autoencoder", "VAE", "CNN Autoencoder"]
    for i, name in enumerate(dl_models):
        if name in all_scores_test:
            ax = axes[i]
            errors = all_scores_test[name]
            threshold = determine_threshold(all_scores_val[name])
            ax.hist(errors, bins=100, alpha=0.7, color=COLORS[i], density=True)
            ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({DEFAULT_THRESHOLD_PERCENTILE}%)")
            ax.set_title(name)
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Density")
            ax.legend()
    plt.suptitle("Reconstruction Error Distributions (Test Set)")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "reconstruction_error_distribution.png", dpi=150)
    plt.close()

    # Chart 3: Precision-Recall curves
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (name, scores) in enumerate(all_scores_contaminated.items()):
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
        auc_val = compute_auc_pr(scores, y_true)
        ax.plot(recall_vals, precision_vals, label=f"{name} (AUC-PR={auc_val:.3f})", color=COLORS[i % len(COLORS)])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (Synthetic Anomaly Detection)")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "precision_recall_curves.png", dpi=150)
    plt.close()

    # Chart 4: PCA scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    pca = PCA(n_components=2, random_state=RANDOM_STATE)

    # Subsample for visualization
    n_viz = min(10000, len(X_test_scaled))
    rng = np.random.RandomState(RANDOM_STATE)
    viz_idx = rng.choice(len(X_test_scaled), n_viz, replace=False)
    X_viz = X_test_scaled[viz_idx]
    scores_viz = ae_test_errors[viz_idx]

    X_pca = pca.fit_transform(X_viz)
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=scores_viz, cmap="RdYlGn_r",
                         s=5, alpha=0.5)
    plt.colorbar(scatter, label="Anomaly Score")
    ax.set_title("PCA Projection Colored by Autoencoder Anomaly Score")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "anomaly_scatter.png", dpi=150)
    plt.close()

    # Chart 5: Training time comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(comparison_df["Model"])
    times = comparison_df["Training Time (s)"].values
    bars = ax.barh(names, times, color=[COLORS[i % len(COLORS)] for i in range(len(names))])
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", fontsize=10)
    ax.set_xlabel("Training Time (seconds)")
    ax.set_title("Model Training Time Comparison")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "training_time.png", dpi=150)
    plt.close()

    # Chart 6: Feature contribution to anomaly detection
    fig, ax = plt.subplots(figsize=(10, 8))
    threshold = determine_threshold(all_scores_val["Autoencoder"])
    anomaly_mask = ae_test_errors > threshold
    normal_mask = ~anomaly_mask

    # Compute mean absolute difference between normal and anomalous for each feature
    feature_names_json = json.loads((RESULTS_DIR / "feature_names.json").read_text())
    normal_mean = X_test_scaled[normal_mask].mean(axis=0)
    anomaly_mean = X_test_scaled[anomaly_mask].mean(axis=0)
    diff = np.abs(anomaly_mean - normal_mean)
    sorted_idx = np.argsort(diff)[::-1]

    top_n = min(15, len(feature_names_json))
    ax.barh(
        [feature_names_json[i] for i in sorted_idx[:top_n]][::-1],
        diff[sorted_idx[:top_n]][::-1],
        color=COLORS[0],
    )
    ax.set_xlabel("Mean Absolute Difference (Normal vs Anomalous)")
    ax.set_title("Top Features Distinguishing Anomalous Claims")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance_anomaly.png", dpi=150)
    plt.close()

    print("  Generated 6 charts")


if __name__ == "__main__":
    import json  # needed for chart 6
    main()
