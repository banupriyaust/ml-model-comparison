"""
Anomaly Detection Dashboard - Healthcare Claims
Interactive Streamlit dashboard for visualizing anomaly detection results.

Run: streamlit run anomaly_detection/anomaly_dashboard.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ─── Config ───
RESULTS_DIR = Path(__file__).resolve().parent / "results"
COLORS = ["#10a37f", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#06b6d4"]
PLOTLY_THEME = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    font=dict(color="#4a4a4a", size=13),
    xaxis=dict(gridcolor="#f0f0f0"),
    yaxis=dict(gridcolor="#f0f0f0"),
)

st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───
st.markdown("""
<style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetric"] {
        background: #e6f7f1; border: 1px solid #10a37f;
        border-radius: 12px; padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #1a1a1a !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #0d8c6d !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Data ───
@st.cache_data
def load_results():
    try:
        metrics = pd.read_csv(str(RESULTS_DIR / "anomaly_model_metrics.csv"))
        top_anomalies = pd.read_csv(str(RESULTS_DIR / "top_anomalous_claims.csv"))
        errors_data = np.load(str(RESULTS_DIR / "reconstruction_errors.npz"))
        errors = {key: errors_data[key] for key in errors_data.files}
        feature_names = json.loads((RESULTS_DIR / "feature_names.json").read_text())

        # Load per-model evaluation details
        eval_details = {}
        for f in RESULTS_DIR.glob("evaluation_*.csv"):
            model_name = f.stem.replace("evaluation_", "").replace("_", " ").title()
            eval_details[model_name] = pd.read_csv(str(f))

        # Load anomaly type breakdown
        type_breakdown = None
        tb_path = RESULTS_DIR / "anomaly_type_breakdown.csv"
        if tb_path.exists():
            type_breakdown = pd.read_csv(str(tb_path))

        return metrics, top_anomalies, errors, feature_names, eval_details, type_breakdown
    except FileNotFoundError as e:
        st.error(f"Results not found. Run training first:\n```\npython -m anomaly_detection.train_anomaly_models\n```\n\nMissing: {e}")
        st.stop()


metrics_df, top_anomalies_df, errors_dict, feature_names, eval_details, type_breakdown_df = load_results()

# ─── Header ───
st.title("Healthcare Claims Anomaly Detection")
st.caption("Deep Learning vs Traditional Methods on 1M Health Insurance Claims")

# Best model banner
best = metrics_df.iloc[0]
threshold_col = f"F1@97.5%"
st.markdown(f"""
<div style="background:#f0fdf4; border:1px solid #86efac; border-radius:12px; padding:16px 24px; margin-bottom:20px;">
    <h3 style="color:#166534; margin:0 0 8px;">Best Model: {best['Model']}</h3>
    <p style="color:#166534; margin:0;">
        AUC-PR: <b>{best['AUC-PR']:.4f}</b> &nbsp;|&nbsp;
        F1@97.5%: <b>{best.get(threshold_col, 0):.4f}</b> &nbsp;|&nbsp;
        Training: <b>{best.get('Training Time (s)', 0):.1f}s</b>
    </p>
</div>
""", unsafe_allow_html=True)

# Key metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Models Compared", "5")
c2.metric("Training Samples", "720,000")
c3.metric("Test Samples", "200,000")
c4.metric("Features", str(len(feature_names)))

st.divider()

# ─── Tabs ───
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview", "Reconstruction Errors", "Anomaly Explorer",
    "Model Comparison", "Feature Analysis", "Anomaly Patterns",
])

# ═══════════════════ TAB 1: OVERVIEW ═══════════════════
with tab1:
    st.subheader("Model Performance Summary")

    # Bar chart comparison
    fig = go.Figure()
    metric_cols = [c for c in metrics_df.columns if c.startswith("Precision") or c.startswith("Recall") or c.startswith("F1")]
    for i, col in enumerate(metric_cols):
        fig.add_trace(go.Bar(
            name=col.split("@")[0],
            x=metrics_df["Model"],
            y=metrics_df[col],
            marker_color=COLORS[i % len(COLORS)],
            text=metrics_df[col].apply(lambda x: f"{x:.3f}"),
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group", height=500,
        title="Detection Metrics at 97.5th Percentile Threshold",
        yaxis_range=[0, 1.15],
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Training time chart
    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        y=metrics_df["Model"],
        x=metrics_df["Training Time (s)"],
        orientation="h",
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(metrics_df))],
        text=metrics_df["Training Time (s)"].apply(lambda x: f"{x:.1f}s"),
        textposition="outside",
    ))
    fig_time.update_layout(
        title="Training Time Comparison",
        xaxis_title="Seconds", height=350,
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Full metrics table
    st.subheader("Complete Metrics Table")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ═══════════════════ TAB 2: RECONSTRUCTION ERRORS ═══════════════════
with tab2:
    st.subheader("Reconstruction Error Distributions")

    dl_models = [k for k in errors_dict.keys() if k in ["Autoencoder", "VAE", "CNN Autoencoder"]]
    if not dl_models:
        dl_models = list(errors_dict.keys())[:3]

    selected_model = st.selectbox("Select Model", dl_models, key="recon_model")
    threshold_pct = st.slider("Threshold Percentile", 85.0, 99.9, 97.5, 0.5, key="recon_threshold")

    if selected_model in errors_dict:
        errors = errors_dict[selected_model]
        threshold = np.percentile(errors, threshold_pct)
        n_flagged = int((errors > threshold).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Threshold Value", f"{threshold:.6f}")
        col2.metric("Claims Flagged", f"{n_flagged:,}")
        col3.metric("Flagged %", f"{100 * n_flagged / len(errors):.2f}%")

        # Histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=errors, nbinsx=200, name="All Claims",
            marker_color=COLORS[0], opacity=0.7,
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold ({threshold_pct}%)")
        fig.update_layout(
            title=f"{selected_model} — Reconstruction Error Distribution",
            xaxis_title="Reconstruction Error (MSE)",
            yaxis_title="Count", height=450,
            **PLOTLY_THEME,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Side-by-side all models
    st.subheader("All Models Comparison")
    cols = st.columns(len(dl_models))
    for i, model_name in enumerate(dl_models):
        if model_name in errors_dict:
            with cols[i]:
                err = errors_dict[model_name]
                fig_small = go.Figure()
                fig_small.add_trace(go.Histogram(x=err, nbinsx=100, marker_color=COLORS[i]))
                thr = np.percentile(err, 97.5)
                fig_small.add_vline(x=thr, line_dash="dash", line_color="red")
                fig_small.update_layout(title=model_name, height=300, showlegend=False, **PLOTLY_THEME)
                st.plotly_chart(fig_small, use_container_width=True)

# ═══════════════════ TAB 3: ANOMALY EXPLORER ═══════════════════
with tab3:
    st.subheader("Top Anomalous Claims Explorer")

    # PCA scatter if image exists
    scatter_img = RESULTS_DIR / "anomaly_scatter.png"
    if scatter_img.exists():
        st.image(str(scatter_img), caption="PCA Projection Colored by Anomaly Score", use_container_width=True)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        min_consensus = st.slider("Minimum Consensus (models agreeing)", 0, 5, 2, key="consensus_filter")
    with col2:
        if "claim_category" in top_anomalies_df.columns:
            categories = ["All"] + sorted(top_anomalies_df["claim_category"].dropna().unique().tolist())
            cat_filter = st.selectbox("Claim Category", categories, key="cat_filter")

    # Filter
    display_df = top_anomalies_df.copy()
    if "consensus_count" in display_df.columns:
        display_df = display_df[display_df["consensus_count"] >= min_consensus]
    if "claim_category" in display_df.columns and cat_filter != "All":
        display_df = display_df[display_df["claim_category"] == cat_filter]

    st.metric("Claims Matching Filters", f"{len(display_df):,}")
    st.dataframe(display_df.head(100), use_container_width=True, hide_index=True)

    # Download
    csv = display_df.to_csv(index=False).encode()
    st.download_button("Download Filtered Anomalies CSV", csv, "filtered_anomalies.csv", "text/csv")

# ═══════════════════ TAB 4: MODEL COMPARISON ═══════════════════
with tab4:
    st.subheader("Precision-Recall Curves")

    pr_img = RESULTS_DIR / "precision_recall_curves.png"
    if pr_img.exists():
        st.image(str(pr_img), caption="Precision-Recall Curves (Synthetic Anomalies)", use_container_width=True)

    # Detailed evaluation per model
    st.subheader("Detection Metrics at Multiple Thresholds")
    if eval_details:
        model_select = st.selectbox("Select Model", list(eval_details.keys()), key="eval_model")
        if model_select in eval_details:
            st.dataframe(eval_details[model_select], use_container_width=True, hide_index=True)
    else:
        st.info("Per-model evaluation files not found.")

    # AUC-PR comparison
    st.subheader("AUC-PR Ranking")
    fig_auc = go.Figure()
    fig_auc.add_trace(go.Bar(
        x=metrics_df["Model"],
        y=metrics_df["AUC-PR"],
        marker_color=[COLORS[i % len(COLORS)] for i in range(len(metrics_df))],
        text=metrics_df["AUC-PR"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
    ))
    fig_auc.update_layout(
        title="Area Under Precision-Recall Curve",
        yaxis_range=[0, 1.1], height=400,
        **PLOTLY_THEME,
    )
    st.plotly_chart(fig_auc, use_container_width=True)

# ═══════════════════ TAB 5: FEATURE ANALYSIS ═══════════════════
with tab5:
    st.subheader("Feature Importance for Anomaly Detection")

    feat_img = RESULTS_DIR / "feature_importance_anomaly.png"
    if feat_img.exists():
        st.image(str(feat_img), caption="Top Features Distinguishing Anomalous Claims", use_container_width=True)

    # Feature distribution comparison: normal vs anomalous
    st.subheader("Feature Distribution: Normal vs Anomalous")
    if "anomaly_score_ae" in top_anomalies_df.columns:
        score_cols = [c for c in top_anomalies_df.columns if c.startswith("score_")]
        num_cols = [c for c in top_anomalies_df.select_dtypes(include=[np.number]).columns
                    if c not in score_cols and not c.startswith("flag_") and c != "consensus_count"]

        if num_cols:
            feat_select = st.selectbox("Select Feature", num_cols[:20], key="feat_dist")
            if feat_select in top_anomalies_df.columns and "is_anomaly_ae" in top_anomalies_df.columns:
                fig_dist = go.Figure()
                normal = top_anomalies_df[top_anomalies_df["is_anomaly_ae"] == 0][feat_select].dropna()
                anomalous = top_anomalies_df[top_anomalies_df["is_anomaly_ae"] == 1][feat_select].dropna()
                if len(normal) > 0:
                    fig_dist.add_trace(go.Histogram(x=normal, name="Normal", opacity=0.6, marker_color=COLORS[0]))
                if len(anomalous) > 0:
                    fig_dist.add_trace(go.Histogram(x=anomalous, name="Anomalous", opacity=0.6, marker_color=COLORS[3]))
                fig_dist.update_layout(
                    barmode="overlay", title=f"Distribution: {feat_select}",
                    height=400, **PLOTLY_THEME,
                )
                st.plotly_chart(fig_dist, use_container_width=True)

    # Feature names list
    st.subheader("Engineered Features")
    feat_groups = {
        "Payment (5)": feature_names[:5],
        "Temporal (8)": feature_names[5:13],
        "Diagnosis (4)": feature_names[13:17],
        "Provider (5)": feature_names[17:22],
        "Member (3)": feature_names[22:25],
        "Categorical (3+)": feature_names[25:],
    }
    for group_name, feats in feat_groups.items():
        st.markdown(f"**{group_name}:** {', '.join(feats)}")

# ═══════════════════ TAB 6: ANOMALY PATTERNS ═══════════════════
with tab6:
    st.subheader("Anomaly Pattern Analysis")

    # Breakdown by claim category
    if "claim_category" in top_anomalies_df.columns:
        cat_counts = top_anomalies_df["claim_category"].value_counts()
        fig_cat = go.Figure(data=[go.Pie(
            labels=cat_counts.index, values=cat_counts.values,
            marker=dict(colors=COLORS[:len(cat_counts)]),
            textinfo="label+percent",
        )])
        fig_cat.update_layout(title="Anomalies by Claim Category", height=400, **PLOTLY_THEME)
        st.plotly_chart(fig_cat, use_container_width=True)

    # Synthetic anomaly type detection rates
    if type_breakdown_df is not None:
        st.subheader("Detection Rate by Anomaly Type")
        anomaly_rows = type_breakdown_df[type_breakdown_df["y_true"] == 1].copy()
        if len(anomaly_rows) > 0:
            detect_cols = [c for c in anomaly_rows.columns if c.startswith("detected_")]
            if detect_cols:
                type_rates = anomaly_rows.groupby("anomaly_type")[detect_cols].mean()
                type_rates.columns = [c.replace("detected_", "").replace("_", " ").title() for c in type_rates.columns]

                fig_heat = go.Figure(data=go.Heatmap(
                    z=type_rates.values,
                    x=type_rates.columns.tolist(),
                    y=type_rates.index.tolist(),
                    colorscale="Greens",
                    text=np.round(type_rates.values, 3),
                    texttemplate="%{text:.3f}",
                    showscale=True,
                ))
                fig_heat.update_layout(
                    title="Detection Rate: Anomaly Type x Model",
                    height=400, **PLOTLY_THEME,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

    # Payment distribution
    if "clm_pmt_amt" in top_anomalies_df.columns:
        st.subheader("Payment Distribution of Top Anomalies")
        fig_pay = go.Figure()
        fig_pay.add_trace(go.Histogram(
            x=top_anomalies_df["clm_pmt_amt"].dropna(),
            nbinsx=50, marker_color=COLORS[3],
        ))
        fig_pay.update_layout(
            title="Claim Payment Amount — Top Anomalous Claims",
            xaxis_title="Payment ($)", yaxis_title="Count",
            height=350, **PLOTLY_THEME,
        )
        st.plotly_chart(fig_pay, use_container_width=True)

    # Top anomalous providers
    if "prvdr_num" in top_anomalies_df.columns:
        st.subheader("Top 20 Providers by Anomaly Count")
        provider_counts = top_anomalies_df["prvdr_num"].value_counts().head(20)
        fig_prov = go.Figure()
        fig_prov.add_trace(go.Bar(
            x=provider_counts.values,
            y=provider_counts.index.astype(str),
            orientation="h",
            marker_color=COLORS[0],
        ))
        fig_prov.update_layout(
            title="Top 20 Providers with Most Anomalous Claims",
            xaxis_title="Anomaly Count", height=500,
            **PLOTLY_THEME,
        )
        fig_prov.update_yaxes(autorange="reversed")
        st.plotly_chart(fig_prov, use_container_width=True)

# ─── Sidebar: Export ───
with st.sidebar:
    st.header("Export Data")
    st.download_button(
        "Model Metrics CSV",
        metrics_df.to_csv(index=False).encode(),
        "anomaly_model_metrics.csv", "text/csv",
    )
    st.download_button(
        "Top Anomalies CSV",
        top_anomalies_df.to_csv(index=False).encode(),
        "top_anomalous_claims.csv", "text/csv",
    )
    st.markdown("---")
    st.markdown("**Research Question:**")
    st.markdown("*Can deep learning-based anomaly detection identify potentially fraudulent billing patterns in health insurance claims?*")
