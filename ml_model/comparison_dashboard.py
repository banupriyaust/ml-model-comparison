"""
Model Comparison BI Dashboard - ClaimBot AI
Interactive dashboard comparing 7 ML models with Plotly charts.
Open-source alternative to Power BI.

Run: streamlit run ml_model/comparison_dashboard.py
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Model Comparison", page_icon="\U0001f4ca", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #ffffff; }
header[data-testid="stHeader"] { background:#fff; border-bottom:1px solid #e5e5e5; }
[data-testid="stMetric"] { background:#e6f7f1; border:1px solid #10a37f; border-radius:12px; padding:16px; }
[data-testid="stMetric"] label { color:#1a1a1a !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color:#0d8c6d !important; }
h1,h2,h3 { color:#1a1a1a !important; }
</style>""", unsafe_allow_html=True)

RESULTS_DIR = Path(__file__).resolve().parent / "comparison_results"

st.title("ML Model Comparison Dashboard")
st.caption("7 Models Compared on 500K Health Insurance Claims (4 Traditional + 3 Neural Networks)")

# Load data
try:
    clf_df = pd.read_csv(str(RESULTS_DIR / "classification_metrics.csv"))
    reg_df = pd.read_csv(str(RESULTS_DIR / "regression_metrics.csv"))
    summary_df = pd.read_csv(str(RESULTS_DIR / "model_summary.csv"))
except FileNotFoundError:
    st.error("Run `python ml_model/compare_models.py` first to generate comparison data.")
    st.stop()

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PLOTLY = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    font=dict(color="#4a4a4a", size=13),
    xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
)
COLORS = ["#10a37f", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]

# ── Winner Banner ────────────────────────────────────────────
best_clf = clf_df.loc[clf_df["F1 Score"].idxmax()]
best_reg = reg_df.loc[reg_df["MAE (days)"].idxmin()]
fastest_clf = clf_df.loc[clf_df["Training Time (s)"].idxmin()]

st.markdown(f"""
<div style="background:#f0fdf4; border:1px solid #86efac; border-radius:12px; padding:16px 24px; margin-bottom:20px;">
    <h3 style="color:#166534 !important; margin:0 0 8px;">Best Model: {best_clf['Model']}</h3>
    <p style="color:#166534; margin:0;">
        Classification F1: <b>{best_clf['F1 Score']:.4f}</b> &nbsp; | &nbsp;
        AUC-ROC: <b>{best_clf['AUC-ROC']:.4f}</b> &nbsp; | &nbsp;
        Training: <b>{best_clf['Training Time (s)']:.1f}s</b> (fastest: {fastest_clf['Model']} at {fastest_clf['Training Time (s)']:.1f}s)
    </p>
</div>""", unsafe_allow_html=True)

# ── Key Metrics Row ──────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Models Compared", "7")
c2.metric("Training Samples", "500,000")
c3.metric("Test Samples", "100,000")
c4.metric("Features", "11")

st.divider()

# ── Tab Layout ───────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Classification", "Regression", "Training Time", "Feature Importance"])

with tab1:
    st.subheader("Classification: Approved vs Denied")

    # Metrics comparison bar chart
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
    fig = go.Figure()
    for i, model in enumerate(clf_df["Model"]):
        fig.add_trace(go.Bar(
            name=model,
            x=metrics_to_plot,
            y=[clf_df.iloc[i][m] for m in metrics_to_plot],
            marker_color=COLORS[i % len(COLORS)],
            text=[f"{clf_df.iloc[i][m]:.4f}" for m in metrics_to_plot],
            textposition="outside",
        ))
    min_val = clf_df[metrics_to_plot].min().min()
    y_floor = max(0, min_val - 0.02)
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#4a4a4a", size=13),
        xaxis=dict(gridcolor="#f0f0f0"),
        yaxis=dict(gridcolor="#f0f0f0", range=[y_floor, 1.005], title="Score"),
        barmode="group", height=500,
        title="Classification Metrics Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix Summary")
        cm_data = clf_df[["Model", "TN", "FP", "FN", "TP"]].copy()
        cm_data["Total Errors"] = cm_data["FP"] + cm_data["FN"]
        st.dataframe(cm_data, use_container_width=True, hide_index=True)

    with right:
        st.subheader("Raw Metrics Table")
        display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "Training Time (s)"]
        st.dataframe(clf_df[display_cols], use_container_width=True, hide_index=True)

    # Show saved chart images
    img_path = RESULTS_DIR / "confusion_matrices.png"
    if img_path.exists():
        st.subheader("Confusion Matrices")
        st.image(str(img_path), use_container_width=True)

    img_path = RESULTS_DIR / "roc_curves.png"
    if img_path.exists():
        st.subheader("ROC Curves")
        st.image(str(img_path), use_container_width=True)

with tab2:
    st.subheader("Regression: Processing Days Prediction")

    # MAE, RMSE, R2 comparison
    fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE (days)", "RMSE (days)", "R² Score"])

    for col_idx, metric in enumerate(["MAE (days)", "RMSE (days)", "R2 Score"], 1):
        fig.add_trace(go.Bar(
            x=reg_df["Model"], y=reg_df[metric],
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(reg_df))],
            text=[f"{v:.4f}" for v in reg_df[metric]],
            textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#4a4a4a", size=13),
        height=400, title="Regression Metrics Comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Regression Metrics Table")
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    img_path = RESULTS_DIR / "regressor_comparison.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)

with tab3:
    st.subheader("Training Time Comparison")

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Classification", "Regression"])

    fig.add_trace(go.Bar(
        y=clf_df["Model"], x=clf_df["Training Time (s)"],
        orientation="h", marker_color=[COLORS[i % len(COLORS)] for i in range(len(clf_df))],
        text=[f"{t:.1f}s" for t in clf_df["Training Time (s)"]],
        textposition="outside", showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        y=reg_df["Model"], x=reg_df["Training Time (s)"],
        orientation="h", marker_color=[COLORS[i % len(COLORS)] for i in range(len(reg_df))],
        text=[f"{t:.1f}s" for t in reg_df["Training Time (s)"]],
        textposition="outside", showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#4a4a4a", size=13),
        height=400, title="Training Time (500K samples)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Speed ranking
    st.markdown("### Speed Ranking")
    speed = pd.concat([
        clf_df[["Model", "Training Time (s)"]].assign(Task="Classification"),
        reg_df[["Model", "Training Time (s)"]].assign(Task="Regression"),
    ]).sort_values("Training Time (s)")
    st.dataframe(speed, use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Feature Importance (Tree-Based Models)")

    # Load feature importance CSVs
    fi_files = sorted(RESULTS_DIR.glob("feature_importance_*.csv"))
    if fi_files:
        model_choice = st.selectbox("Select Model",
            [f.stem.replace("feature_importance_", "").replace("_", " ").title() for f in fi_files])

        fname = model_choice.lower().replace(" ", "_")
        fi_path = RESULTS_DIR / f"feature_importance_{fname}.csv"
        if fi_path.exists():
            fi_df = pd.read_csv(str(fi_path))
            fig = px.bar(fi_df.head(11), x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Greens",
                        title=f"{model_choice} - Top Features")
            fig.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                font=dict(color="#4a4a4a", size=13),
                xaxis=dict(gridcolor="#f0f0f0"),
                yaxis=dict(gridcolor="#f0f0f0", categoryorder="total ascending"),
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(fi_df, use_container_width=True, hide_index=True)

    img_path = RESULTS_DIR / "feature_importance.png"
    if img_path.exists():
        st.subheader("All Models Feature Importance")
        st.image(str(img_path), use_container_width=True)

# ── Overall Summary Table ────────────────────────────────────
st.divider()
st.subheader("Complete Model Summary")
st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ── Download buttons ─────────────────────────────────────────
st.divider()
st.subheader("Export Data for Power BI / Tableau")
st.markdown("Download CSVs to import into any BI tool:")

col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("Classification Metrics CSV",
                       clf_df.to_csv(index=False), "classification_metrics.csv", "text/csv")
with col2:
    st.download_button("Regression Metrics CSV",
                       reg_df.to_csv(index=False), "regression_metrics.csv", "text/csv")
with col3:
    st.download_button("Model Summary CSV",
                       summary_df.to_csv(index=False), "model_summary.csv", "text/csv")
