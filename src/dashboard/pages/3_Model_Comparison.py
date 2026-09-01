import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Model Comparison", layout="wide")
st.title("Model Comparison")
st.markdown(
    "Week 2 evaluation results across candidate models. "
    "**Headline metric: PR-AUC** (better than accuracy on imbalanced purchase data). "
    "Final chosen model: **CatBoost** (saved for the API / dashboard)."
)

# Metrics from notebooks/03_model_evaluation.ipynb (test set)
metrics = pd.DataFrame(
    [
        {
            "Model": "XGBoost",
            "Accuracy": 0.9346,
            "Precision": 0.8097,
            "Recall": 0.7600,
            "F1": 0.7840,
            "ROC-AUC": 0.9690,
            "PR-AUC": 0.8644,
        },
        {
            "Model": "CatBoost",
            "Accuracy": 0.9367,
            "Precision": 0.8195,
            "Recall": 0.7627,
            "F1": 0.7901,
            "ROC-AUC": 0.9695,
            "PR-AUC": 0.8612,
        },
        {
            "Model": "RandomForest",
            "Accuracy": 0.9179,
            "Precision": 0.8321,
            "Recall": 0.5947,
            "F1": 0.6936,
            "ROC-AUC": 0.9624,
            "PR-AUC": 0.8374,
        },
        {
            "Model": "LogisticRegression",
            "Accuracy": 0.9158,
            "Precision": 0.6784,
            "Recall": 0.8773,
            "F1": 0.7651,
            "ROC-AUC": 0.9598,
            "PR-AUC": 0.8145,
        },
        {
            "Model": "DecisionTree",
            "Accuracy": 0.8988,
            "Precision": 0.6854,
            "Recall": 0.6507,
            "F1": 0.6676,
            "ROC-AUC": 0.7977,
            "PR-AUC": 0.5005,
        },
    ]
).sort_values("PR-AUC", ascending=False)

st.subheader("Metrics table")
st.dataframe(
    metrics.style.format(
        {
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
            "ROC-AUC": "{:.4f}",
            "PR-AUC": "{:.4f}",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

fig = px.bar(
    metrics,
    x="Model",
    y="PR-AUC",
    color="Model",
    title="PR-AUC by Model (higher is better)",
    text="PR-AUC",
)
fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig.update_layout(yaxis_range=[0, 1], showlegend=False)
st.plotly_chart(fig, use_container_width=True)

fig2 = px.bar(
    metrics.melt(
        id_vars="Model",
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    ),
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    title="Precision / Recall / F1 by Model",
)
fig2.update_layout(yaxis_range=[0, 1])
st.plotly_chart(fig2, use_container_width=True)

st.info(
    "XGBoost edged CatBoost slightly on PR-AUC, but **CatBoost** was selected as the final "
    "deployed model (strong PR-AUC ~0.86, solid recall/precision balance, and used for the "
    "saved `final_model.pkl` + SHAP explanations)."
)
