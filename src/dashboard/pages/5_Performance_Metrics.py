import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

from src.config.settings import load_config
from src.dashboard.loaders import load_data, load_model, load_pipeline, load_threshold
from src.features.engineering import add_engineered_features

st.set_page_config(page_title="Performance Metrics", layout="wide")
st.title("Performance Metrics")
st.markdown(
    "Closer look at the **final CatBoost** model on the held-out test split "
    "(same seed/settings as training). Artifacts are loaded only — nothing is retrained here."
)

config = load_config()
df = load_data()
model = load_model()
pipeline = load_pipeline()
threshold = load_threshold()

df_feat = add_engineered_features(df.copy())
X = df_feat.drop(columns=[config["target_column"]])
y = df_feat[config["target_column"]]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=config["test_size"],
    random_state=config["random_seed"],
)

X_test_t = pipeline.transform(X_test)
y_prob = model.predict_proba(X_test_t)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

pr_auc = average_precision_score(y_test, y_prob)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

m1, m2, m3, m4 = st.columns(4)
m1.metric("PR-AUC", f"{pr_auc:.3f}")
m2.metric("Precision @ threshold", f"{prec:.3f}")
m3.metric("Recall @ threshold", f"{rec:.3f}")
m4.metric("F1 @ threshold", f"{f1:.3f}")

st.caption(f"Operating threshold from config: **{threshold:.2f}**")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_text = [[str(v) for v in row] for row in cm]
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=["Pred No Purchase", "Pred Purchase"],
    y=["Actual No Purchase", "Actual Purchase"],
    annotation_text=cm_text,
    colorscale="Blues",
    showscale=False,
)
fig_cm.update_layout(title="Confusion Matrix (final CatBoost)")
st.plotly_chart(fig_cm, use_container_width=True)

# Precision-recall curve
precision_vals, recall_vals, thr_vals = precision_recall_curve(y_test, y_prob)
pr_df = pd.DataFrame({"Recall": recall_vals, "Precision": precision_vals})
fig_pr = px.line(
    pr_df,
    x="Recall",
    y="Precision",
    title="Precision–Recall Curve",
)
fig_pr.add_hline(
    y=prec,
    line_dash="dot",
    annotation_text=f"Precision @ {threshold}",
)
fig_pr.add_vline(
    x=rec,
    line_dash="dot",
    annotation_text=f"Recall @ {threshold}",
)
st.plotly_chart(fig_pr, use_container_width=True)

# Threshold trade-off table
rows = []
for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
    pred_t = (y_prob >= t).astype(int)
    rows.append(
        {
            "Threshold": t,
            "Precision": round(precision_score(y_test, pred_t), 4),
            "Recall": round(recall_score(y_test, pred_t), 4),
            "F1": round(f1_score(y_test, pred_t), 4),
            "Selected": t == threshold,
        }
    )
thr_df = pd.DataFrame(rows)
st.subheader("Threshold trade-offs")
st.dataframe(thr_df, use_container_width=True, hide_index=True)

st.markdown(
    f"""
### Threshold justification

We operate at **{threshold:.2f}** (from `config.yaml`).

- Lower thresholds (e.g. 0.3–0.4) catch more buyers (higher recall) but create more false alarms.  
- Higher thresholds (e.g. 0.6) are more precise but miss more true buyers.  
- **0.5** is a balanced default for this project: solid precision (~0.82) while still recovering
  a large share of buyers (~0.76 recall) on the Week 2 evaluation.

If the business cost of a missed buyer is much higher than a wasted outreach, move the
threshold down; if outreach is expensive, move it up.
"""
)
