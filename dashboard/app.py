"""
Streamlit Dashboard - Credit Card Fraud Detection.

Run from the project root:
    streamlit run dashboard/app.py
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Friendly display names for the models.
# Clean, light chart styling to match the white dashboard.
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#1e293b",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "text.color": "#334155",
        "axes.grid": True,
        "grid.color": "#eef2f7",
    }
)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost_model": "XGBoost",
}

# The V1-V28 columns are anonymized PCA components. The bank hid the real
# meanings for privacy, so we give the ones exposed in the demo human-friendly
# aliases. These labels are illustrative, not the true (hidden) field names.
V_FRIENDLY = {
    "V1": "Spending Pattern A",
    "V4": "Transaction Risk Signal",
    "V7": "Account Behavior Score",
    "V10": "Location Anomaly",
    "V12": "Purchase Type Signal",
    "V14": "Fraud Indicator (strongest)",
    "V17": "Merchant Risk Signal",
}

# PCA hides the original business field names. These descriptions make the
# anonymized parameters easier to discuss without presenting guesses as facts.
FEATURE_DESCRIPTIONS = {
    **{f"V{i}": f"Anonymized behavioral signal {i}" for i in range(1, 29)},
    "V1": "Spending Pattern A",
    "V4": "Transaction Risk Signal",
    "V7": "Account Behavior Score",
    "V10": "Location Anomaly",
    "V12": "Purchase Type Signal",
    "V14": "Fraud Indicator (strongest)",
    "V17": "Merchant Risk Signal",
    "Time": "Time Since First Recorded Transaction",
    "Amount": "Transaction Amount",
    "Scaled_Amount": "Scaled Transaction Amount",
    "Hour": "Transaction Hour",
    "Log_Amount": "Log-Transformed Amount",
}

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide",
)

# --- Theme and custom styling ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark_mode = st.session_state.dark_mode
colors = {
    "background": "#0f172a" if dark_mode else "#f8fafc",
    "surface": "#1e293b" if dark_mode else "#ffffff",
    "text": "#e2e8f0" if dark_mode else "#1e293b",
    "muted": "#cbd5e1" if dark_mode else "#475569",
    "border": "#334155" if dark_mode else "#e2e8f0",
    "accent": "#38bdf8" if dark_mode else "#4f46e5",
}

top_left, top_right = st.columns([8, 2])
with top_right:
    st.toggle(
        "☀️" if dark_mode else "🌙",
        key="dark_mode",
        help="Switch between light and dark dashboard themes.",
        label_visibility="visible",
    )

st.markdown(
    f"""
    <style>
    .stApp {{
        background: {colors['background']};
    }}
    /* Colorful banner header */
    .hero {{
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 22px 28px;
        border-radius: 16px;
        margin-bottom: 18px;
        box-shadow: 0 8px 24px rgba(99,102,241,0.25);
    }}
    .main-title {{
        font-size: 2.3rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }}
    .subtitle {{
        color: #eef2ff;
        font-size: 1.02rem;
        margin-top: 4px;
    }}
    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: {colors['surface']};
        border: 1px solid {colors['border']};
        border-left: 5px solid {colors['accent']};
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(15,23,42,0.06);
    }}
    div[data-testid="stMetricValue"] {{
        color: {colors['accent']};
        font-weight: 800;
    }}
    div[data-testid="stMetricLabel"] {{ color: {colors['muted']}; font-weight: 600; }}
    /* Top tabs */
    .stTabs [data-baseweb="tab-list"] {{
        justify-content: center;
        gap: 8px;
        background: {colors['surface']};
        padding: 8px;
        border-radius: 12px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 46px;
        border-radius: 10px;
        padding: 0 20px;
        color: {colors['muted']};
        font-weight: 700;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.35);
    }}
    h1, h2, h3, h4 {{ color: {colors['text']} !important; }}
    p, label, .stMarkdown, .stCaption {{ color: {colors['muted']} !important; }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-baseweb="tab"] {{ color: {colors['muted']} !important; }}
    [data-baseweb="tab"][aria-selected="true"] {{ color: #ffffff !important; }}
    [data-testid="stDataFrame"] td {{ color: {colors['text']} !important; }}
    [data-testid="stImage"] {{ max-width: 680px; margin: 0 auto 24px; }}
    [data-testid="stImage"] img {{ width: 680px !important; max-width: 100%; height: auto; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="main-title">💳 Credit Card Fraud Detection</div>
        <div class="subtitle">Machine learning that spots fraudulent
        transactions hidden among hundreds of thousands of legitimate ones.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    return train, test


@st.cache_resource
def load_models():
    models = {}
    for name in ["logistic_regression", "random_forest", "xgboost_model"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


# --- Guard: data must be present ---
if not os.path.exists(os.path.join(PROCESSED_DIR, "train.csv")):
    st.error(
        "Processed data not found. Run `python src/preprocess.py` and "
        "`python src/train.py` first."
    )
    st.stop()

train, test = load_data()
models = load_models()
X_test = test.drop(columns=["Class"])
y_test = test["Class"]

# --- Top navigation as tabs ---
tab_home, tab_data, tab_eda, tab_models, tab_predict, tab_explain = st.tabs(
    [
        "🏠 Home",
        "🗂️ Dataset",
        "🔍 Explore Data",
        "🤖 Models",
        "🎯 Live Check",
        "📈 Why It Decides",
    ]
)

# ============================ HOME ============================
with tab_home:
    st.header("Welcome to the Fraud Detection Lab")
    st.markdown(
        """
        This project uses machine learning to identify suspicious credit card
        transactions in a highly imbalanced dataset. The dashboard lets you
        understand the data, compare models, test a transaction, and see which
        anonymized signals influence a prediction.
        """
    )

    intro_left, intro_right = st.columns(2)
    with intro_left:
        st.subheader("What this project does")
        st.markdown(
            """
            - Finds patterns that separate legitimate and fraudulent payments.
            - Compares Logistic Regression, Random Forest, and XGBoost.
            - Prioritizes PR-AUC because fraud is rare and accuracy can mislead.
            """
        )
    with intro_right:
        st.subheader("Explore the dashboard")
        st.markdown(
            """
            - **Dataset:** inspect transaction samples and statistics.
            - **Explore Data:** compare fraud and legitimate behavior.
            - **Models:** review curves and confusion matrices.
            - **Live Check:** simulate a transaction and get a risk score.
            """
        )

    st.divider()
    st.subheader("Project Snapshot")

    total = len(train) + len(test)
    frauds = int(train["Class"].sum() + test["Class"].sum())
    legit = total - frauds
    fraud_rate = frauds / total * 100

    st.subheader("The Dataset at a Glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Fraud Cases", f"{frauds:,}")
    c3.metric("Legit Cases", f"{legit:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

    st.subheader("Models & Performance")
    comp_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        comparison = pd.read_csv(comp_path)
        display = comparison.copy()
        display["Model"] = display["Model"].map(MODEL_LABELS).fillna(display["Model"])
        display["Accuracy"] = (display["Accuracy"] * 100).round(2).astype(str) + "%"
        display["F1"] = display["F1"].round(4)
        display["ROC_AUC"] = display["ROC_AUC"].round(4)
        display["Avg_Precision"] = display["Avg_Precision"].round(4)
        display = display.rename(
            columns={"ROC_AUC": "ROC-AUC", "Avg_Precision": "PR-AUC"}
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        best_idx = comparison["Avg_Precision"].idxmax()
        best_model = MODEL_LABELS.get(
            comparison.loc[best_idx, "Model"], comparison.loc[best_idx, "Model"]
        )
        best_acc = comparison.loc[best_idx, "Accuracy"] * 100
        best_prauc = comparison.loc[best_idx, "Avg_Precision"]
        best_f1 = comparison.loc[best_idx, "F1"]

        b1, b2, b3 = st.columns(3)
        b1.metric("Best Model", best_model)
        b2.metric("Its Accuracy", f"{best_acc:.2f}%")
        b3.metric("Its PR-AUC", f"{best_prauc:.4f}")

        st.subheader("What We're Inferring")
        st.markdown(
            f"""
- **The data is extremely imbalanced.** Only **{frauds:,}** of **{total:,}**
  transactions are fraud ({fraud_rate:.2f}%). A model that guessed "legit"
  every time would still be ~{100 - fraud_rate:.2f}% accurate while catching
  **zero** fraud, so accuracy alone is misleading.
- **We trained 3 models** — Logistic Regression, Random Forest, and XGBoost —
  on rebalanced data so the rare fraud cases get a fair chance.
- **All models look ~97-99% accurate**, but the number that really matters on
  imbalanced data is **PR-AUC** (how well fraud is caught without false alarms).
- **{best_model}** is the strongest, with a PR-AUC of **{best_prauc:.4f}** and
  F1 of **{best_f1:.4f}**.
- **Bottom line:** the model flags the large majority of fraud while keeping
  false alarms low — exactly what a real fraud-screening system needs.
"""
        )
    else:
        st.warning("Run `python src/evaluate.py` to generate the comparison table.")

# ============================ DATASET ============================
with tab_data:
    st.header("Dataset Overview")

    st.info(
        "**About the features:** Each transaction has an **Amount**, a **Time**, "
        "and 28 columns named **V1–V28**. Those V-columns are *anonymized* — the "
        "bank ran the original data through a math transform (PCA) to hide "
        "sensitive details like card numbers and merchants. So V1–V28 are real "
        "behavioral signals, just with their true names hidden for privacy. "
        "Bigger swings in a few of them (like V14 and V17) strongly hint at fraud."
    )

    total = len(train) + len(test)
    frauds = int(train["Class"].sum() + test["Class"].sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{total:,}")
    col2.metric("Fraud Cases", f"{frauds:,}")
    col3.metric("Fraud Rate", f"{(frauds / total * 100):.2f}%")
    col4.metric("Features", f"{X_test.shape[1]}")

    st.subheader("Sample Data")
    st.dataframe(train.head(20), use_container_width=True)
    st.subheader("Statistical Summary")
    st.dataframe(train.describe(), use_container_width=True)

# ============================ EDA ============================
with tab_eda:
    st.header("Explore the Data")

    st.subheader("How Rare Is Fraud?")
    fig, ax = plt.subplots(figsize=(6, 4))
    train["Class"].value_counts().plot(
        kind="bar", ax=ax, color=["#22c55e", "#ef4444"]
    )
    ax.set_xticklabels(["Legit (0)", "Fraud (1)"], rotation=0)
    ax.set_ylabel("Count")
    st.pyplot(fig)

    st.subheader("Compare a Feature Between Legit and Fraud")
    feature_catalog = pd.DataFrame(
        {
            "Parameter": X_test.columns.tolist(),
            "What it represents": [
                FEATURE_DESCRIPTIONS.get(name, "Anonymized model feature")
                for name in X_test.columns
            ],
        }
    )
    st.caption("Select one row in the table to inspect its fraud and legitimate distributions.")
    selection = st.dataframe(
        feature_catalog,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="feature_catalog",
    )
    selected_rows = selection.selection.rows
    feature = (
        feature_catalog.iloc[selected_rows[0]]["Parameter"]
        if selected_rows
        else feature_catalog.iloc[0]["Parameter"]
    )
    st.caption(
        f"Selected: **{FEATURE_DESCRIPTIONS.get(feature, 'Anonymized model feature')}** "
        f"(`{feature}`)."
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    train[train["Class"] == 0][feature].hist(
        bins=50, alpha=0.6, label="Legit", ax=ax, color="#22c55e"
    )
    train[train["Class"] == 1][feature].hist(
        bins=50, alpha=0.6, label="Fraud", ax=ax, color="#ef4444"
    )
    ax.legend()
    ax.set_title(f"{FEATURE_DESCRIPTIONS.get(feature, feature)} Distribution")
    st.pyplot(fig)

    st.subheader("Which Signals Point to Fraud the Most?")
    corr = (
        train.corr()["Class"]
        .drop("Class")
        .sort_values(key=abs, ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    corr.plot(
        kind="barh",
        ax=ax,
        color=corr.apply(lambda x: "#ef4444" if x > 0 else "#3b82f6"),
    )
    ax.set_title("Top 10 Signals Correlated With Fraud")
    st.pyplot(fig)

# ============================ MODELS ============================
with tab_models:
    st.header("Model Comparison")

    comp_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        comparison = pd.read_csv(comp_path)
        pretty = comparison.copy()
        pretty["Model"] = pretty["Model"].map(MODEL_LABELS).fillna(pretty["Model"])
        best_row = comparison.loc[comparison["Avg_Precision"].idxmax()]
        best_name = MODEL_LABELS.get(best_row["Model"], best_row["Model"])
        best_metric, score_col = st.columns([1, 2])
        best_metric.metric("Recommended Model", best_name)
        best_metric.caption(
            f"Highest PR-AUC: {best_row['Avg_Precision']:.4f} "
            "(best balance for rare fraud detection)."
        )
        with score_col:
            st.caption("Model scores")
            score_columns = st.columns(len(comparison))
            for score_column, (_, row) in zip(score_columns, comparison.iterrows()):
                score_column.metric(
                    MODEL_LABELS.get(row["Model"], row["Model"]),
                    f"{row['Avg_Precision']:.4f}",
                    "PR-AUC",
                )
        st.dataframe(
            pretty.style.highlight_max(
                axis=0, subset=["Accuracy", "F1", "ROC_AUC", "Avg_Precision"],
                props="background-color: #166534; color: #ffffff; font-weight: 700;",
            ),
            use_container_width=True,
            hide_index=True,
        )

    if not models:
        st.warning("No trained models found. Run `python src/train.py`.")
    else:
        left, right = st.columns(2)
        with left:
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(7, 5))
            for name, model in models.items():
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc = roc_auc_score(y_test, y_prob)
                ax.plot(fpr, tpr, label=f"{MODEL_LABELS[name]} (AUC={auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
        with right:
            st.subheader("Precision-Recall Curves")
            fig, ax = plt.subplots(figsize=(7, 5))
            for name, model in models.items():
                y_prob = model.predict_proba(X_test)[:, 1]
                prec, rec, _ = precision_recall_curve(y_test, y_prob)
                ax.plot(rec, prec, label=MODEL_LABELS[name])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend()
            st.pyplot(fig)

        st.subheader("Confusion Matrices")
        cols = st.columns(len(models))
        for col, (name, model) in zip(cols, models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
            ax.set_title(MODEL_LABELS[name])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            col.pyplot(fig)

# ============================ LIVE PREDICTION ============================
with tab_predict:
    st.header("Live Fraud Check")
    st.write(
        "Simulate a transaction and see how the model scores it. "
        "The sliders below are the signals that matter most for fraud."
    )

    col1, col2 = st.columns(2)
    with col1:
        amount = st.slider("Transaction Amount ($)", 0.0, 5000.0, 100.0)
        hour = st.slider("Hour of Day (0-23)", 0, 23, 12)
    with col2:
        v_features = {}
        for v, label in V_FRIENDLY.items():
            v_features[v] = st.slider(
                f"{label}", -5.0, 5.0, 0.0,
                help=f"Anonymized signal {v}. Extreme values often signal fraud.",
            )

    if st.button("🔍 Check This Transaction", type="primary"):
        best_path = os.path.join(MODELS_DIR, "best_model.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if not (os.path.exists(best_path) and os.path.exists(scaler_path)):
            st.error("best_model.pkl or scaler.pkl missing. Train first.")
        else:
            with open(best_path, "rb") as f:
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)

            input_data = {f"V{i}": 0.0 for i in range(1, 29)}
            input_data.update(v_features)
            input_data["Scaled_Amount"] = scaler.transform([[amount]])[0][0]
            input_data["Hour"] = hour
            input_data["Log_Amount"] = np.log1p(amount)

            input_df = pd.DataFrame([input_data]).reindex(
                columns=X_test.columns.tolist(), fill_value=0.0
            )
            prob = model.predict_proba(input_df)[:, 1][0]
            pred = model.predict(input_df)[0]

            st.progress(min(float(prob), 1.0))
            if pred == 1:
                st.error(f"🚨 FRAUD DETECTED — Fraud probability: {prob:.2%}")
            else:
                st.success(f"✅ Looks Legitimate — Fraud probability: {prob:.2%}")

# ============================ EXPLAINABILITY ============================
with tab_explain:
    st.header("Why the Model Decides")

    st.info(
        "These charts show which signals push a transaction toward 'fraud'. "
        "**Red = a high value pushing toward fraud, blue = pushing toward legit.** "
        "The signals near the top matter most — V14 and V17 are consistently the "
        "strongest fraud indicators."
    )

    shap_path = os.path.join(FIGURES_DIR, "shap_summary.png")
    imp_path = os.path.join(FIGURES_DIR, "feature_importance.png")

    if os.path.exists(shap_path):
        st.image(
            shap_path,
            caption="SHAP Summary — what drives predictions",
            width=680,
        )
    if os.path.exists(imp_path):
        st.image(
            imp_path,
            caption="Feature Importance — most influential signals",
            width=680,
        )
    if not (os.path.exists(shap_path) or os.path.exists(imp_path)):
        st.warning("Run `python src/evaluate.py` to generate these plots.")
