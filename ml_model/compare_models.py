"""
Multi-Model Comparison for Claim Outcome Prediction
Trains 7 classifiers and 7 regressors, compares metrics,
generates matplotlib charts and exports CSV for BI tools.

Models (Traditional):
  1. XGBoost
  2. Random Forest
  3. Logistic Regression / Linear Regression
  4. Gradient Boosting (sklearn)

Models (Neural Networks):
  5. FNN (Feedforward Neural Network)
  6. CNN (1D Convolutional Neural Network)
  7. RNN (LSTM-based Recurrent Neural Network)

Usage: python ml_model/compare_models.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sqlite3
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
    roc_curve, precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

warnings.filterwarnings("ignore")

DB_PATH = Path(r"C:\Users\banup\Desktop\Masters thesis\chatbot\database\claims.db")
OUTPUT_DIR = Path(__file__).resolve().parent / "comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_SIZE = 500_000  # 500K for faster training of all 7 models


# ═══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK WRAPPER CLASSES (sklearn-compatible)
# ═══════════════════════════════════════════════════════════════════════════════
class KerasClassifierWrapper:
    """Base class for Keras-based sklearn-compatible classifiers."""

    def __init__(self, epochs=30, batch_size=2048, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.model = None

    def _build_model(self, input_dim):
        raise NotImplementedError

    def _preprocess(self, X):
        return X

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)
        X_scaled = self.scaler.fit_transform(X)
        X_processed = self._preprocess(X_scaled)
        self.model = self._build_model(X.shape[1])
        self.model.fit(
            X_processed, y,
            epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, verbose=self.verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, verbose=0),
            ]
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_processed = self._preprocess(X_scaled)
        proba = self.model.predict(X_processed, verbose=0).flatten()
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_processed = self._preprocess(X_scaled)
        proba = self.model.predict(X_processed, verbose=0).flatten()
        return np.column_stack([1 - proba, proba])


class KerasRegressorWrapper:
    """Base class for Keras-based sklearn-compatible regressors."""

    def __init__(self, epochs=30, batch_size=2048, verbose=0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.model = None

    def _build_model(self, input_dim):
        raise NotImplementedError

    def _preprocess(self, X):
        return X

    def fit(self, X, y):
        tf.keras.utils.set_random_seed(42)
        X_scaled = self.scaler.fit_transform(X)
        X_processed = self._preprocess(X_scaled)
        self.model = self._build_model(X.shape[1])
        self.model.fit(
            X_processed, y,
            epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, verbose=self.verbose,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, verbose=0),
            ]
        )
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_processed = self._preprocess(X_scaled)
        return self.model.predict(X_processed, verbose=0).flatten()


# ── FNN (Feedforward Neural Network) ─────────────────────────────────────────
class FNNClassifier(KerasClassifierWrapper):
    def _build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


class FNNRegressor(KerasRegressorWrapper):
    def _build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


# ── CNN (1D Convolutional Neural Network) ────────────────────────────────────
class CNNClassifier(KerasClassifierWrapper):
    def _preprocess(self, X):
        return X.reshape(X.shape[0], X.shape[1], 1)

    def _build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu',
                                   input_shape=(input_dim, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


class CNNRegressor(KerasRegressorWrapper):
    def _preprocess(self, X):
        return X.reshape(X.shape[0], X.shape[1], 1)

    def _build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu',
                                   input_shape=(input_dim, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


# ── RNN (LSTM-based Recurrent Neural Network) ────────────────────────────────
class RNNClassifier(KerasClassifierWrapper):
    def _preprocess(self, X):
        return X.reshape(X.shape[0], X.shape[1], 1)

    def _build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(input_dim, 1)),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model


class RNNRegressor(KerasRegressorWrapper):
    def _preprocess(self, X):
        return X.reshape(X.shape[0], X.shape[1], 1)

    def _build_model(self, input_dim):
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(input_dim, 1)),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & FEATURE ENGINEERING (same as train_model.py)
# ═══════════════════════════════════════════════════════════════════════════════
def load_data():
    print(f"Connecting to {DB_PATH}...")
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA cache_size=-50000")

    print(f"Sampling {SAMPLE_SIZE:,} rows (approved + denied only)...")
    query = """
        SELECT claim_category, urgency_level, icd9_dgns_cd_1, icd9_dgns_cd_2,
               icd9_dgns_cd_3, icd9_dgns_cd_4, icd9_dgns_cd_5,
               prvdr_num, clm_pmt_amt, clm_from_dt, clm_thru_dt,
               claim_status, estimated_processing_days, approved_amount
        FROM claims
        WHERE claim_status IN ('approved', 'denied')
        ORDER BY RANDOM()
        LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(SAMPLE_SIZE,))
    conn.close()
    print(f"Loaded {len(df):,} rows")
    return df


def engineer_features(df):
    print("Engineering features...")

    df["claim_category"] = df["claim_category"].fillna("unknown")
    urgency_map = {"routine": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
    df["urgency_ordinal"] = df["urgency_level"].map(urgency_map).fillna(1)

    icd9_counts = df["icd9_dgns_cd_1"].value_counts()
    top_100 = set(icd9_counts.head(100).index)
    df["icd9_freq"] = df["icd9_dgns_cd_1"].apply(
        lambda x: icd9_counts.get(x, 0) if x in top_100 else 0
    )

    prvdr_counts = df["prvdr_num"].value_counts()
    top_200 = set(prvdr_counts.head(200).index)
    df["prvdr_freq"] = df["prvdr_num"].apply(
        lambda x: prvdr_counts.get(x, 0) if x in top_200 else 0
    )

    df["clm_from_dt"] = pd.to_datetime(df["clm_from_dt"], errors="coerce")
    df["clm_thru_dt"] = pd.to_datetime(df["clm_thru_dt"], errors="coerce")
    df["claim_duration"] = (df["clm_thru_dt"] - df["clm_from_dt"]).dt.days.fillna(0)
    df["clm_pmt_amt"] = pd.to_numeric(df["clm_pmt_amt"], errors="coerce").fillna(0)

    diag_cols = [f"icd9_dgns_cd_{i}" for i in range(1, 6)]
    df["diagnosis_count"] = df[diag_cols].notna().sum(axis=1)
    df["has_secondary_diag"] = df["icd9_dgns_cd_2"].notna().astype(int)

    cat_dummies = pd.get_dummies(df["claim_category"], prefix="cat")

    df["target_approved"] = (df["claim_status"] == "approved").astype(int)
    df["processing_days"] = pd.to_numeric(
        df["estimated_processing_days"], errors="coerce"
    ).fillna(14)

    feature_cols = [
        "urgency_ordinal", "icd9_freq", "prvdr_freq",
        "claim_duration", "clm_pmt_amt", "diagnosis_count", "has_secondary_diag",
    ]

    X = pd.concat([df[feature_cols], cat_dummies], axis=1)
    y_class = df["target_approved"]
    y_reg = df["processing_days"]
    feature_names = list(X.columns)

    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
    print(f"Class balance: Approved={y_class.sum():,} ({y_class.mean()*100:.1f}%), "
          f"Denied={len(y_class)-y_class.sum():,} ({(1-y_class.mean())*100:.1f}%)")

    return X.values, y_class.values, y_reg.values, feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════
def get_classifiers():
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_split=10,
            n_jobs=-1, random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=42, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        "FNN (Neural Net)": FNNClassifier(epochs=30, batch_size=2048),
        "CNN (1D Conv)": CNNClassifier(epochs=30, batch_size=2048),
        "RNN (LSTM)": RNNClassifier(epochs=30, batch_size=2048),
    }


def get_regressors():
    return {
        "XGBoost": XGBRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42, verbosity=0,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=12, min_samples_split=10,
            n_jobs=-1, random_state=42,
        ),
        "Linear Regression": LinearRegression(n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=42,
        ),
        "FNN (Neural Net)": FNNRegressor(epochs=30, batch_size=2048),
        "CNN (1D Conv)": CNNRegressor(epochs=30, batch_size=2048),
        "RNN (LSTM)": RNNRegressor(epochs=30, batch_size=2048),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test):
    classifiers = get_classifiers()
    results = []

    for name, clf in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training Classifier: {name}")
        print(f"{'='*60}")

        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred.astype(float)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")
        print(f"  Time:      {train_time:.1f}s")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Denied','Approved'])}")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "AUC-ROC": auc,
            "Training Time (s)": round(train_time, 1),
            "TN": cm[0][0], "FP": cm[0][1], "FN": cm[1][0], "TP": cm[1][1],
            "y_prob": y_prob,
            "y_pred": y_pred,
            "model": clf,
        })

    return results


def train_and_evaluate_regressors(X_train, X_test, y_train, y_test):
    regressors = get_regressors()
    results = []

    for name, reg in regressors.items():
        print(f"\n{'='*60}")
        print(f"Training Regressor: {name}")
        print(f"{'='*60}")

        start = time.time()
        reg.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"  MAE:   {mae:.2f} days")
        print(f"  RMSE:  {rmse:.2f} days")
        print(f"  R2:    {r2:.4f}")
        print(f"  Time:  {train_time:.1f}s")

        results.append({
            "Model": name,
            "MAE (days)": round(mae, 2),
            "RMSE (days)": round(rmse, 2),
            "R2 Score": round(r2, 4),
            "Training Time (s)": round(train_time, 1),
            "y_pred": y_pred,
            "model": reg,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MATPLOTLIB VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
COLORS = ["#10a37f", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]


def plot_classifier_comparison(results):
    """Bar chart comparing all classifier metrics."""
    models = [r["Model"] for r in results]
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    n_models = len(models)

    fig, ax = plt.subplots(figsize=(max(12, n_models * 2), 6))
    x = np.arange(len(models))
    width = min(0.15, 0.8 / len(metrics))

    for i, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        bars = ax.bar(x + i * width, values, width, label=metric, color=plt.cm.Set2(i))
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Classification Model Comparison (Claim Approval/Denial)", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 1.12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "classifier_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'classifier_comparison.png'}")


def plot_confusion_matrices(results, y_test):
    """Dynamic grid of confusion matrices."""
    n_models = len(results)
    ncols = 4
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    labels = ["Denied", "Approved"]

    for idx, r in enumerate(results):
        ax = axes.flat[idx]
        cm = confusion_matrix(y_test, r["y_pred"])
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(f'{r["Model"]}  (Acc: {r["Accuracy"]:.4f})', fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        fontsize=14, fontweight="bold", color=color)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        axes.flat[idx].set_visible(False)

    plt.suptitle("Confusion Matrices - All Models", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'confusion_matrices.png'}")


def plot_roc_curves(results, y_test):
    """ROC curves for all classifiers on one plot."""
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, r in enumerate(results):
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, color=COLORS[idx % len(COLORS)], linewidth=2.5,
                label=f'{r["Model"]} (AUC = {r["AUC-ROC"]:.4f})')

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - Claim Approval Prediction", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "roc_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'roc_curves.png'}")


def plot_regressor_comparison(results):
    """Bar chart comparing regressor metrics."""
    models = [r["Model"] for r in results]
    model_colors = [COLORS[i % len(COLORS)] for i in range(len(models))]

    fig, axes = plt.subplots(1, 3, figsize=(max(15, len(models) * 2), 6))

    # MAE
    vals = [r["MAE (days)"] for r in results]
    bars = axes[0].bar(models, vals, color=model_colors)
    axes[0].set_title("Mean Absolute Error (days)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("MAE (lower is better)")
    for bar, val in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", fontweight="bold", fontsize=8)

    # RMSE
    vals = [r["RMSE (days)"] for r in results]
    bars = axes[1].bar(models, vals, color=model_colors)
    axes[1].set_title("Root Mean Squared Error (days)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("RMSE (lower is better)")
    for bar, val in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{val:.2f}", ha="center", fontweight="bold", fontsize=8)

    # R2
    vals = [r["R2 Score"] for r in results]
    bars = axes[2].bar(models, vals, color=model_colors)
    axes[2].set_title("R² Score", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("R² (higher is better)")
    for bar, val in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", fontweight="bold", fontsize=8)

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.suptitle("Regression Model Comparison (Processing Days)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "regressor_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'regressor_comparison.png'}")


def plot_training_time(clf_results, reg_results):
    """Training time comparison."""
    n_models = max(len(clf_results), len(reg_results))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, n_models * 0.8)))

    # Classification
    models = [r["Model"] for r in clf_results]
    times = [r["Training Time (s)"] for r in clf_results]
    clf_colors = [COLORS[i % len(COLORS)] for i in range(len(models))]
    bars = axes[0].barh(models, times, color=clf_colors)
    axes[0].set_title("Classifier Training Time", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Seconds")
    for bar, t in zip(bars, times):
        axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{t:.1f}s", va="center", fontweight="bold")

    # Regression
    models = [r["Model"] for r in reg_results]
    times = [r["Training Time (s)"] for r in reg_results]
    reg_colors = [COLORS[i % len(COLORS)] for i in range(len(models))]
    bars = axes[1].barh(models, times, color=reg_colors)
    axes[1].set_title("Regressor Training Time", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Seconds")
    for bar, t in zip(bars, times):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                     f"{t:.1f}s", va="center", fontweight="bold")

    for ax in axes:
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Training Time Comparison (500K samples)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "training_time.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'training_time.png'}")


def plot_feature_importance(clf_results, feature_names):
    """Feature importance for tree-based models."""
    tree_models = [r for r in clf_results if hasattr(r["model"], "feature_importances_")]

    fig, axes = plt.subplots(1, len(tree_models), figsize=(6 * len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, r in zip(axes, tree_models):
        imp = r["model"].feature_importances_
        indices = np.argsort(imp)[-10:]  # top 10
        ax.barh([feature_names[i] for i in indices], imp[indices], color="#10a37f")
        ax.set_title(f'{r["Model"]} Feature Importance', fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_DIR / "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'feature_importance.png'}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  MULTI-MODEL COMPARISON - Claim Outcome Prediction")
    print("  7 Classifiers + 7 Regressors on 500K Claims")
    print("  (4 Traditional ML + 3 Neural Networks)")
    print("=" * 70)

    # Load and prepare data
    df = load_data()
    X, y_class, y_reg, feature_names = engineer_features(df)

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = (
        train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)
    )
    print(f"\nTrain: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    # ── Classification ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CLASSIFICATION: Approved vs Denied")
    print("=" * 70)
    clf_results = train_and_evaluate_classifiers(X_train, X_test, y_cls_train, y_cls_test)

    # ── Regression ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REGRESSION: Processing Days Prediction")
    print("=" * 70)
    reg_results = train_and_evaluate_regressors(X_train, X_test, y_reg_train, y_reg_test)

    # ── Find Best Models ────────────────────────────────────────────────
    best_clf = max(clf_results, key=lambda r: r["F1 Score"])
    best_reg = min(reg_results, key=lambda r: r["MAE (days)"])

    print("\n" + "=" * 70)
    print("  WINNER")
    print("=" * 70)
    print(f"  Best Classifier:  {best_clf['Model']}  (F1={best_clf['F1 Score']:.4f}, AUC={best_clf['AUC-ROC']:.4f})")
    print(f"  Best Regressor:   {best_reg['Model']}  (MAE={best_reg['MAE (days)']:.2f} days)")

    # ── Generate charts ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  GENERATING CHARTS")
    print("=" * 70)
    plot_classifier_comparison(clf_results)
    plot_confusion_matrices(clf_results, y_cls_test)
    plot_roc_curves(clf_results, y_cls_test)
    plot_regressor_comparison(reg_results)
    plot_training_time(clf_results, reg_results)
    plot_feature_importance(clf_results, feature_names)

    # ── Export CSVs for BI tools ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EXPORTING DATA FOR BI TOOLS")
    print("=" * 70)

    # Classification metrics CSV
    clf_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("y_prob", "y_pred", "model")}
        for r in clf_results
    ])
    clf_df.to_csv(str(OUTPUT_DIR / "classification_metrics.csv"), index=False)
    print(f"Saved: {OUTPUT_DIR / 'classification_metrics.csv'}")

    # Regression metrics CSV
    reg_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("y_pred", "model")}
        for r in reg_results
    ])
    reg_df.to_csv(str(OUTPUT_DIR / "regression_metrics.csv"), index=False)
    print(f"Saved: {OUTPUT_DIR / 'regression_metrics.csv'}")

    # Combined summary CSV
    summary = []
    for r in clf_results:
        summary.append({
            "Model": r["Model"], "Task": "Classification",
            "Primary Metric": "F1 Score", "Value": r["F1 Score"],
            "Secondary Metric": "AUC-ROC", "Secondary Value": r["AUC-ROC"],
            "Training Time (s)": r["Training Time (s)"],
        })
    for r in reg_results:
        summary.append({
            "Model": r["Model"], "Task": "Regression",
            "Primary Metric": "MAE (days)", "Value": r["MAE (days)"],
            "Secondary Metric": "R2 Score", "Secondary Value": r["R2 Score"],
            "Training Time (s)": r["Training Time (s)"],
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(str(OUTPUT_DIR / "model_summary.csv"), index=False)
    print(f"Saved: {OUTPUT_DIR / 'model_summary.csv'}")

    # Feature importance CSV for BI
    for r in clf_results:
        if hasattr(r["model"], "feature_importances_"):
            imp_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": r["model"].feature_importances_,
                "Model": r["Model"],
            }).sort_values("Importance", ascending=False)
            fname = r["Model"].lower().replace(" ", "_")
            imp_df.to_csv(str(OUTPUT_DIR / f"feature_importance_{fname}.csv"), index=False)
            print(f"Saved: feature_importance_{fname}.csv")

    # Save best model (handle both sklearn and Keras models)
    if isinstance(best_clf["model"], KerasClassifierWrapper):
        best_clf["model"].model.save(str(OUTPUT_DIR / "best_classifier_keras"))
        print("Saved: best_classifier_keras/ (Keras model)")
    else:
        joblib.dump(best_clf["model"], str(OUTPUT_DIR / "best_classifier.joblib"))
        print("Saved: best_classifier.joblib")

    if isinstance(best_reg["model"], KerasRegressorWrapper):
        best_reg["model"].model.save(str(OUTPUT_DIR / "best_regressor_keras"))
        print("Saved: best_regressor_keras/ (Keras model)")
    else:
        joblib.dump(best_reg["model"], str(OUTPUT_DIR / "best_regressor.joblib"))
        print("Saved: best_regressor.joblib")

    print(f"\n{'='*70}")
    print(f"  ALL DONE - Results saved to {OUTPUT_DIR}")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  Charts (PNG):  classifier_comparison.png, confusion_matrices.png,")
    print(f"                 roc_curves.png, regressor_comparison.png,")
    print(f"                 training_time.png, feature_importance.png")
    print(f"  BI Data (CSV): classification_metrics.csv, regression_metrics.csv,")
    print(f"                 model_summary.csv, feature_importance_*.csv")
    print(f"  Models:        best_classifier.joblib, best_regressor.joblib")


if __name__ == "__main__":
    main()
