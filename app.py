import urllib.request
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_score, recall_score, f1_score)

st.set_page_config(page_title="Sentinel | Fraud Detection", page_icon="🛡️", layout="wide")
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

@st.cache_data(show_spinner=False)
def load_dataset():
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return pd.read_csv(response)

@st.cache_resource(show_spinner=False)
def train_model(df, test_size=0.25, trees=200):
    X, y = df.drop(columns=["Class"]), df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=trees, class_weight="balanced_subsample",
            min_samples_leaf=2, random_state=42, n_jobs=-1))
    ])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    return model, X_test, y_test, probabilities

st.title("🛡️ Sentinel")
st.subheader("Credit Card Fraud Detection")
st.caption("Real anonymized data · Random Forest · Streamlit")
with st.sidebar:
    st.header("Model settings")
    test_size = st.slider("Test set size", 0.15, 0.35, 0.25, 0.05)
    trees = st.select_slider("Random Forest trees", options=[100, 150, 200, 250, 300], value=200)
    threshold = st.slider("Fraud alert threshold", 0.05, 0.95, 0.35, 0.05)
    st.info("Lower thresholds usually catch more fraud but can increase false alerts.")

try:
    with st.spinner("Downloading the public dataset..."):
        data = load_dataset()
except Exception as exc:
    st.error("Dataset download failed. Check the app's internet access and try again.")
    st.exception(exc)
    st.stop()

if "Class" not in data.columns:
    st.error("Expected target column 'Class' was not found.")
    st.stop()
data = data.dropna(subset=["Class"]).copy()
data["Class"] = data["Class"].astype(int)
if not set(data["Class"].unique()).issubset({0, 1}):
    st.error("Class must contain 0 (legitimate) and 1 (fraud).")
    st.stop()

fraud_n = int((data["Class"] == 1).sum())
legit_n = int((data["Class"] == 0).sum())
a, b, c, d = st.columns(4)
a.metric("Transactions", f"{len(data):,}")
b.metric("Legitimate", f"{legit_n:,}")
c.metric("Fraud cases", f"{fraud_n:,}")
d.metric("Fraud rate", f"{100*fraud_n/max(len(data),1):.3f}%")

overview, evaluation, scoring, dataset_tab = st.tabs(
    ["📊 Overview", "🧠 Model evaluation", "🔍 Check transaction", "📁 Dataset"])

with overview:
    left, right = st.columns(2)
    counts = data["Class"].map({0: "Legitimate", 1: "Fraud"}).value_counts().rename_axis("Type").reset_index(name="Count")
    with left:
        st.plotly_chart(px.bar(counts, x="Type", y="Count", title="Class imbalance"), use_container_width=True)
    with right:
        sample = data.sample(min(20000, len(data)), random_state=42).assign(
            Type=lambda x: x["Class"].map({0: "Legitimate", 1: "Fraud"}))
        st.plotly_chart(px.histogram(sample, x="Amount", color="Type", nbins=50,
            barmode="overlay", opacity=0.65, title="Transaction amount distribution"), use_container_width=True)
    st.warning("Fraud is rare. Accuracy alone can be misleading; inspect precision, recall, PR-AUC, and the confusion matrix.")
    st.markdown("""
    **Dataset facts**
    - Historical European card transactions from two days in September 2013.
    - `Class = 1` means fraud; `Class = 0` means legitimate.
    - `V1`–`V28` are anonymized PCA components; `Time` and `Amount` are retained.
    - Intended for research and education, not real-time banking decisions.
    """)

with evaluation:
    st.markdown("### Train and evaluate Random Forest")
    st.write("Uses a stratified holdout split and class-weighted Random Forest.")
    if st.button("Train / retrain model", type="primary"):
        train_model.clear()
    try:
        with st.spinner("Training and evaluating..."):
            model, X_test, y_test, probs = train_model(data, test_size, trees)
        preds = (probs >= threshold).astype(int)
        metrics = {
            "ROC-AUC": roc_auc_score(y_test, probs),
            "Average precision (PR-AUC)": average_precision_score(y_test, probs),
            "Precision": precision_score(y_test, preds, zero_division=0),
            "Recall": recall_score(y_test, preds, zero_division=0),
            "F1 score": f1_score(y_test, preds, zero_division=0),
        }
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.4f}")
        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        st.markdown("#### Confusion matrix")
        st.dataframe(pd.DataFrame(cm, index=["Actual legitimate", "Actual fraud"],
            columns=["Predicted legitimate", "Predicted fraud"]), use_container_width=True)
        report = classification_report(y_test, preds, target_names=["Legitimate", "Fraud"],
            output_dict=True, zero_division=0)
        st.markdown("#### Classification report")
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)
        rows = []
        for t in np.linspace(0.05, 0.95, 19):
            p = (probs >= t).astype(int)
            rows.append({"Threshold": t, "Precision": precision_score(y_test, p, zero_division=0),
                         "Recall": recall_score(y_test, p, zero_division=0),
                         "F1": f1_score(y_test, p, zero_division=0)})
        curve = pd.DataFrame(rows).melt(id_vars="Threshold", var_name="Metric", value_name="Score")
        st.plotly_chart(px.line(curve, x="Threshold", y="Score", color="Metric", markers=True),
                        use_container_width=True)
        st.download_button("Download test predictions",
            pd.DataFrame({"actual_class": y_test.to_numpy(), "fraud_probability": probs,
                          "predicted_class": preds}).to_csv(index=False).encode(),
            "fraud_test_predictions.csv", "text/csv")
        st.caption("Random holdout is a baseline evaluation. For stronger evidence, use chronological validation and select thresholds on a separate validation set.")
    except Exception as exc:
        st.error("Training or evaluation failed.")
        st.exception(exc)

with scoring:
    st.markdown("### Score a transaction")
    st.write("The model requires a feature vector. This demo initializes anonymized PCA features from an example test row.")
    try:
        model, X_test, y_test, probs = train_model(data, test_size, trees)
        base = X_test.iloc[0].copy()
        with st.form("score_form"):
            amount = st.number_input("Transaction amount", min_value=0.0, max_value=1000000.0,
                                     value=float(max(0.0, base.get("Amount", 100.0))), step=1.0)
            time_value = st.number_input("Time feature", min_value=0.0, max_value=200000.0,
                                         value=float(base.get("Time", 50000.0)), step=100.0)
            st.caption("V1–V28 are anonymized PCA features. The defaults are sample values; replace them only with a valid feature vector.")
            values = {}
            cols = st.columns(4)
            for i, feature in enumerate([x for x in X_test.columns if x.startswith("V")]):
                with cols[i % 4]:
                    values[feature] = st.number_input(feature, value=float(base[feature]),
                        step=0.1, format="%.4f", key=f"feat_{feature}")
            submitted = st.form_submit_button("Analyze transaction")
        if submitted:
            row = base.copy()
            row["Amount"], row["Time"] = amount, time_value
            for feature, value in values.items():
                row[feature] = value
            probability = float(model.predict_proba(pd.DataFrame([row], columns=X_test.columns))[:, 1][0])
            st.metric("Estimated fraud probability", f"{probability:.2%}")
            if probability >= threshold:
                st.error(f"High-risk alert: score is at or above the {threshold:.0%} threshold.")
            else:
                st.success(f"Below the selected {threshold:.0%} alert threshold.")
            st.caption("A model score is not proof of fraud and should not be the sole basis for declining a transaction.")
    except Exception as exc:
        st.error("The model is not ready for scoring.")
        st.exception(exc)

with dataset_tab:
    st.markdown("### Dataset preview")
    st.dataframe(data.head(30), use_container_width=True)
    st.write("Dataset shape:", data.shape)
    st.write("Missing values:", int(data.isna().sum().sum()))
    st.download_button("Download first 5,000 rows",
        data.head(5000).to_csv(index=False).encode(), "creditcard_sample.csv", "text/csv")
    st.markdown("Source: [Public CSV mirror](https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv) · Original dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)")

st.markdown("---")
st.caption("Educational prototype only. The data is historical and anonymized. Protect payment data and perform security, calibration, monitoring, and human-review checks before real-world use.")
