"""
Fraud Detection — Streamlit Dashboard
=======================================
Pages:
  1. Model Performance   — PR curve, model metrics comparison, threshold slider
  2. Cost Simulator      — FP/FN cost calculator with live confusion matrix
  3. Drift Monitor       — per-feature PSI / KS against training distribution
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib
from sklearn.metrics import precision_recall_curve, roc_curve

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark, premium look
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark sidebar */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Main bg */
  .stApp { background: #0f172a; color: #e2e8f0; }

  /* Metric cards */
  [data-testid="metric-container"] {
      background: linear-gradient(135deg, #1e293b, #0f172a);
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 1rem;
  }
  [data-testid="metric-container"] label { color: #94a3b8 !important; font-size: 0.8rem; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
      color: #f8fafc !important; font-weight: 700;
  }

  /* Section header */
  h2, h3 { color: #f1f5f9; letter-spacing: -0.02em; }

  /* Divider */
  hr { border-color: #334155; }

  /* Plotly chart bg */
  .js-plotly-plot { border-radius: 12px; }

  /* Badges */
  .badge-green { background:#064e3b; color:#6ee7b7; padding:2px 10px;
                 border-radius:999px; font-size:0.75rem; font-weight:600; }
  .badge-red   { background:#7f1d1d; color:#fca5a5; padding:2px 10px;
                 border-radius:999px; font-size:0.75rem; font-weight:600; }

  /* Drift table */
  .drift-ok   { color: #34d399; }
  .drift-warn { color: #fbbf24; }
  .drift-bad  { color: #f87171; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)
MODEL_DIR  = os.path.join(ROOT_DIR, "models")
DATA_DIR   = os.path.join(ROOT_DIR, "data")
CSV_PATH   = os.path.join(DATA_DIR, "creditcard.csv")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artefacts …")
def load_artifacts():
    model    = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler   = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    train_dists = joblib.load(os.path.join(MODEL_DIR, "train_distributions.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, scaler, feat_cols, train_dists, metrics


@st.cache_data(show_spinner="Computing test-set scores …")
def compute_test_scores():
    model, scaler, feat_cols, _, _ = load_artifacts()
    df = pd.read_csv(CSV_PATH).sort_values("Time").reset_index(drop=True)
    split_idx = int(len(df) * 0.70)
    test = df.iloc[split_idx:].reset_index(drop=True)
    X = test[FEATURE_COLS].copy()
    X[["Time", "Amount"]] = scaler.transform(X[["Time", "Amount"]])
    scores = model.predict_proba(X)[:, 1]
    y_true = test["Class"].values
    return y_true, scores


# ─────────────────────────────────────────────────────────────────────────────
# Try loading; graceful fallback if model not yet trained
# ─────────────────────────────────────────────────────────────────────────────
try:
    model, scaler, feat_cols, train_dists, metrics = load_artifacts()
    model_ready = True
except Exception as e:
    model_ready = False
    _load_error = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Fraud Detection")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📈 Model Performance", "💰 Cost Simulator", "🔍 Drift Monitor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    if model_ready:
        st.markdown(
            f'<span class="badge-green">Model loaded ✓</span>',
            unsafe_allow_html=True,
        )
        st.caption(f"XGBoost · v1.0.0")
        st.caption(f"PR-AUC: **{metrics['xgb_pr_auc']:.4f}**")
    else:
        st.markdown(
            '<span class="badge-red">Model not found ✗</span>',
            unsafe_allow_html=True,
        )
        st.caption("Run `python train.py` to train the model.")

# ─────────────────────────────────────────────────────────────────────────────
# Guard
# ─────────────────────────────────────────────────────────────────────────────
if not model_ready:
    st.error("⚠️  Model artefacts not found.")
    st.info(
        "Train the model first:\n```bash\ncd fraud-detection-api\npython train.py\n```"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
if page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")

    # Top KPI metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("XGBoost PR-AUC",   f"{metrics['xgb_pr_auc']:.4f}")
    c2.metric("XGBoost ROC-AUC",  f"{metrics['xgb_roc_auc']:.4f}")
    c3.metric("LR Baseline PR-AUC", f"{metrics['lr_pr_auc']:.4f}")
    c4.metric("Precision@Recall=80%", f"{metrics['xgb_precision_at_r80']:.4f}")
    c5.metric("Recall@FPR=1%",    f"{metrics['xgb_recall_at_fpr1']:.4f}")

    st.markdown("---")

    # PR Curve
    y_true, y_scores = compute_test_scores()
    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)

    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(
        x=rec, y=prec,
        mode="lines",
        name="XGBoost",
        line=dict(color="#6366f1", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.12)",
    ))
    fig_pr.add_hline(
        y=y_true.mean(), line_dash="dot",
        line_color="#94a3b8",
        annotation_text=f"Random baseline ({y_true.mean():.4f})",
        annotation_position="bottom right",
    )
    fig_pr.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall", yaxis_title="Precision",
        height=380,
        paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
    )
    st.plotly_chart(fig_pr, use_container_width=True)

    st.markdown("---")

    # ROC Curve
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_scores)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr_vals, y=tpr_vals,
        mode="lines",
        name=f"XGBoost (AUC={metrics['xgb_roc_auc']:.4f})",
        line=dict(color="#22d3ee", width=2.5),
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Random",
        line=dict(color="#94a3b8", dash="dot"),
    ))
    fig_roc.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=380,
        paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    st.caption(
        f"Test set: **{metrics['test_size']:,}** transactions · "
        f"**{metrics['test_fraud_count']}** fraud · "
        f"imbalance ratio ~{metrics['test_size']//max(1,metrics['test_fraud_count'])}:1"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Cost Simulator
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💰 Cost Simulator":
    st.markdown("## 💰 Cost-of-FP vs. FN Calculator")
    st.caption("Adjust the threshold and costs to simulate real-world financial impact.")

    y_true, y_scores = compute_test_scores()
    prec, rec, thresholds = precision_recall_curve(y_true, y_scores)

    default_thr = metrics.get("threshold_at_1pct_fpr", 0.5)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        threshold = st.slider(
            "Fraud probability threshold",
            min_value=0.01, max_value=0.99,
            value=float(default_thr),
            step=0.001,
            format="%.3f",
        )
        st.markdown("---")
        cost_fp = st.number_input(
            "Cost per False Positive ($)\n(blocked legit customer, support cost)",
            min_value=0.0, value=5.0, step=1.0,
        )
        cost_fn = st.number_input(
            "Cost per False Negative ($)\n(missed fraud, average loss)",
            min_value=0.0, value=150.0, step=10.0,
        )

    preds = (y_scores >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_val    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    total_cost    = fp * cost_fp + fn * cost_fn

    with col_right:
        # Confusion matrix heatmap
        cm = [[tn, fp], [fn, tp]]
        labels = [["TN", "FP"], ["FN", "TP"]]
        z_text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"], [f"FN\n{fn:,}", f"TP\n{tp:,}"]]

        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            text=z_text,
            texttemplate="%{text}",
            colorscale=[[0, "#0f172a"], [0.5, "#1e3a5f"], [1, "#6366f1"]],
            showscale=False,
            xgap=4, ygap=4,
        ))
        fig_cm.update_layout(
            title=f"Confusion Matrix @ threshold={threshold:.3f}",
            xaxis=dict(tickvals=[0, 1], ticktext=["Predicted Legit", "Predicted Fraud"],
                       color="#e2e8f0"),
            yaxis=dict(tickvals=[0, 1], ticktext=["Actually Legit", "Actually Fraud"],
                       color="#e2e8f0"),
            height=320,
            paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
            font=dict(color="#e2e8f0"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("True Positives",  f"{tp:,}")
    k2.metric("False Positives", f"{fp:,}")
    k3.metric("False Negatives", f"{fn:,}")
    k4.metric("Precision / Recall", f"{precision_val:.3f} / {recall_val:.3f}")
    k5.metric("💸 Estimated Cost", f"${total_cost:,.0f}",
              delta=f"FP: ${fp*cost_fp:,.0f}  FN: ${fn*cost_fn:,.0f}",
              delta_color="inverse")

    st.markdown("---")
    # Threshold cost sweep chart
    sweep_thresholds = np.linspace(0.01, 0.99, 200)
    costs = []
    for t in sweep_thresholds:
        p = (y_scores >= t).astype(int)
        fp_t = int(((p == 1) & (y_true == 0)).sum())
        fn_t = int(((p == 0) & (y_true == 1)).sum())
        costs.append(fp_t * cost_fp + fn_t * cost_fn)

    fig_cost = go.Figure()
    fig_cost.add_trace(go.Scatter(
        x=sweep_thresholds, y=costs,
        mode="lines",
        line=dict(color="#f59e0b", width=2),
        name="Total cost",
    ))
    fig_cost.add_vline(
        x=threshold, line_dash="dot", line_color="#f87171",
        annotation_text=f"Current: ${total_cost:,.0f}",
        annotation_position="top right",
    )
    fig_cost.update_layout(
        title="Total Cost vs. Threshold",
        xaxis_title="Threshold", yaxis_title="Estimated Cost ($)",
        height=300,
        paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
    )
    st.plotly_chart(fig_cost, use_container_width=True)

    optimal_idx = int(np.argmin(costs))
    st.info(
        f"💡 Minimum cost ${min(costs):,.0f} is achieved at threshold "
        f"**{sweep_thresholds[optimal_idx]:.3f}** "
        f"(using your current FP/FN cost inputs)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Drift Monitor
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔍 Drift Monitor":
    st.markdown("## 🔍 Feature Drift Monitor")
    st.caption(
        "Compares a sample of recent transactions against the training distribution "
        "using Population Stability Index (PSI) and Kolmogorov-Smirnov test."
    )

    # Let the user pick features to check and upload/simulate data
    st.markdown("### Configure Drift Check")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_simulate = st.number_input(
            "Simulate N incoming transactions",
            min_value=50, max_value=5000, value=500, step=50,
        )
        drift_severity = st.select_slider(
            "Simulated drift severity (for demo)",
            options=["None", "Low", "Medium", "High"],
            value="Low",
        )
        features_to_check = st.multiselect(
            "Features to check",
            options=FEATURE_COLS,
            default=["V1", "V2", "V3", "V4", "V14", "V17", "Amount"],
        )
        run_btn = st.button("▶  Run Drift Check", type="primary")

    with col_b:
        st.markdown("""
**PSI guidelines**
| PSI | Interpretation |
|-----|---------------|
| < 0.1 | No significant shift |
| 0.1 – 0.25 | Moderate shift — monitor |
| ≥ 0.25 | Significant shift — investigate |

**KS p-value < 0.05** → distributions differ significantly.
        """)

    if run_btn and features_to_check:
        shift_map = {"None": 0.0, "Low": 0.3, "Medium": 1.0, "High": 2.5}
        shift = shift_map[drift_severity]

        results_rows = []
        for feat in features_to_check:
            if feat not in train_dists:
                continue
            ref_vals = np.array(train_dists[feat]["values"], dtype=float)
            mu  = train_dists[feat]["mean"]
            std = max(train_dists[feat]["std"], 1e-6)
            # Simulate incoming data with controlled drift
            incoming = np.random.normal(loc=mu + shift * std, scale=std, size=int(n_simulate))

            from api.drift import calculate_psi, calculate_ks
            psi_val = calculate_psi(ref_vals, incoming)
            ks_res  = calculate_ks(ref_vals, incoming)

            drifted = (psi_val >= 0.25) or (ks_res["p_value"] < 0.05)
            level   = "🔴 High" if psi_val >= 0.25 else ("🟡 Moderate" if psi_val >= 0.1 else "🟢 Stable")

            results_rows.append({
                "Feature":     feat,
                "PSI":         round(psi_val, 4),
                "KS Stat":     round(ks_res["statistic"], 4),
                "KS p-value":  round(ks_res["p_value"], 4),
                "Drift":       level,
                "_drifted":    drifted,
            })

        if results_rows:
            df_res = pd.DataFrame(results_rows)
            n_drifted = df_res["_drifted"].sum()

            st.markdown(f"### Results — {len(df_res)} features checked")

            badge = (
                f'<span class="badge-red">⚠ {n_drifted} features drifted</span>'
                if n_drifted else
                f'<span class="badge-green">✓ No drift detected</span>'
            )
            st.markdown(badge, unsafe_allow_html=True)
            st.markdown("---")

            # Display table (without internal _drifted column)
            display_df = df_res.drop(columns=["_drifted"])
            st.dataframe(
                display_df.style.background_gradient(subset=["PSI"], cmap="RdYlGn_r"),
                use_container_width=True,
            )

            # Bar chart of PSI values
            fig_psi = px.bar(
                df_res.sort_values("PSI", ascending=False),
                x="Feature", y="PSI",
                color="PSI",
                color_continuous_scale=["#22d3ee", "#fbbf24", "#f87171"],
                range_color=[0, 0.4],
                title="PSI per Feature",
                height=350,
            )
            fig_psi.add_hline(y=0.1, line_dash="dot", line_color="#fbbf24",
                              annotation_text="Moderate (0.1)")
            fig_psi.add_hline(y=0.25, line_dash="dot", line_color="#f87171",
                              annotation_text="Significant (0.25)")
            fig_psi.update_layout(
                paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                font=dict(color="#e2e8f0"),
                coloraxis_showscale=False,
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155"),
            )
            st.plotly_chart(fig_psi, use_container_width=True)

            # Distribution overlay for first drifted feature
            drifted_feats = df_res[df_res["_drifted"]]["Feature"].tolist()
            if drifted_feats:
                feat_show = drifted_feats[0]
                ref_vals = np.array(train_dists[feat_show]["values"], dtype=float)
                mu  = train_dists[feat_show]["mean"]
                std = max(train_dists[feat_show]["std"], 1e-6)
                incoming = np.random.normal(loc=mu + shift * std, scale=std, size=int(n_simulate))

                fig_dist = go.Figure()
                fig_dist.add_trace(go.Histogram(
                    x=ref_vals, nbinsx=50, name="Training",
                    marker_color="#6366f1", opacity=0.6,
                    histnorm="probability density",
                ))
                fig_dist.add_trace(go.Histogram(
                    x=incoming, nbinsx=50, name="Incoming",
                    marker_color="#f87171", opacity=0.6,
                    histnorm="probability density",
                ))
                fig_dist.update_layout(
                    barmode="overlay",
                    title=f"Distribution Shift — {feat_show}",
                    height=320,
                    paper_bgcolor="#1e293b", plot_bgcolor="#0f172a",
                    font=dict(color="#e2e8f0"),
                    xaxis=dict(gridcolor="#334155"),
                    yaxis=dict(gridcolor="#334155"),
                    legend=dict(bgcolor="#1e293b"),
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.warning("No matching features found in training distributions.")
    elif run_btn:
        st.warning("Please select at least one feature to check.")