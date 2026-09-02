import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import plotly.express as px
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
from src.dashboard.loaders import (
    load_calibrated_model,
    load_data,
    load_model,
    load_pipeline,
    load_threshold,
)
from src.evaluation.calibration import brier_score, reliability_bins
from src.evaluation.profit import expected_value, profit_curve
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
fig_cm = px.imshow(
    cm,
    x=["Pred No Purchase", "Pred Purchase"],
    y=["Actual No Purchase", "Actual Purchase"],
    text_auto=True,
    color_continuous_scale="Blues",
    title="Confusion Matrix (final CatBoost)",
)
fig_cm.update_layout(coloraxis_showscale=False)
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

# --- Stretch goal #1: profit / expected-value curve ---
st.subheader("Profit curve (expected value)")
st.markdown(
    "Stretch goal: turn the threshold trade-off into money. "
    "Net value = `(true positives × conversion value) − (false positives × intervention cost)`."
)

default_cost = float(config.get("intervention_cost", 5))
default_value = float(config.get("conversion_value", 50))

c1, c2 = st.columns(2)
intervention_cost = c1.number_input(
    "Intervention cost (false positive)",
    min_value=0.0,
    value=default_cost,
    step=1.0,
)
conversion_value = c2.number_input(
    "Conversion value (true positive)",
    min_value=0.0,
    value=default_value,
    step=1.0,
)

curve = profit_curve(
    y_test,
    y_prob,
    intervention_cost=intervention_cost,
    conversion_value=conversion_value,
)
best_row = curve.loc[curve["is_optimal"]].iloc[0]
optimal_threshold = float(best_row["threshold"])
optimal_ev = float(best_row["expected_value"])
week2_ev = expected_value(
    y_test,
    y_prob,
    threshold=threshold,
    intervention_cost=intervention_cost,
    conversion_value=conversion_value,
)

p1, p2, p3 = st.columns(3)
p1.metric("Profit-optimal threshold", f"{optimal_threshold:.2f}")
p2.metric("Expected value @ optimal", f"{optimal_ev:,.0f}")
p3.metric(f"Expected value @ Week 2 ({threshold:.2f})", f"{week2_ev:,.0f}")

fig_profit = px.line(
    curve,
    x="threshold",
    y="expected_value",
    title="Net expected value vs decision threshold",
    labels={
        "threshold": "Decision threshold",
        "expected_value": "Net expected value",
    },
)
fig_profit.add_vline(
    x=optimal_threshold,
    line_dash="dash",
    annotation_text=f"Profit-optimal ({optimal_threshold:.2f})",
)
fig_profit.add_vline(
    x=threshold,
    line_dash="dot",
    annotation_text=f"Week 2 ({threshold:.2f})",
)
st.plotly_chart(fig_profit, use_container_width=True)

st.markdown(
    f"""
With intervention cost **{intervention_cost:g}** and conversion value **{conversion_value:g}**,
the profit-optimal threshold on this test split is **{optimal_threshold:.2f}**
(expected value **{optimal_ev:,.0f}**), vs Week 2’s **{threshold:.2f}**
(**{week2_ev:,.0f}**).

Week 2’s 0.5 was chosen for balanced precision/recall, not ROI. Under cheap interventions
relative to purchase value, the curve peaks lower because catching buyers is worth more than
avoiding a few false alarms. I’m keeping **{threshold:.2f}** as the default in config —
the profit-optimal value depends on the cost assumptions you plug in above.
"""
)

# --- Stretch goal #2: reliability / calibration ---
st.subheader("Reliability curve (probability calibration)")
st.markdown(
    "Stretch goal: check whether a predicted probability of 0.7 really means "
    "~70% of those sessions convert. Perfect calibration follows the diagonal."
)

raw_bins = reliability_bins(y_test, y_prob, n_bins=10)
raw_brier = brier_score(y_test, y_prob)

calibrated = load_calibrated_model()
cal_probs = None
cal_bins = None
cal_brier = None
if calibrated is not None:
    cal_probs = calibrated.predict_proba(X_test_t)[:, 1]
    cal_bins = reliability_bins(y_test, cal_probs, n_bins=10)
    cal_brier = brier_score(y_test, cal_probs)

b1, b2 = st.columns(2)
b1.metric("Raw Brier score", f"{raw_brier:.4f}")
if cal_brier is not None:
    b2.metric("Calibrated Brier score", f"{cal_brier:.4f}", delta=f"{cal_brier - raw_brier:+.4f}")
else:
    b2.metric("Calibrated Brier score", "n/a")
    st.caption("Run `notebooks/06_calibration.ipynb` to create `calibrated_model.pkl`.")

rel_rows = raw_bins.assign(Model="Raw CatBoost")
if cal_bins is not None:
    rel_rows = pd.concat(
        [rel_rows, cal_bins.assign(Model="Isotonic calibrated")],
        ignore_index=True,
    )

fig_rel = px.line(
    rel_rows,
    x="mean_predicted_probability",
    y="fraction_of_positives",
    color="Model",
    markers=True,
    title="Reliability curve (before vs after isotonic calibration)",
    labels={
        "mean_predicted_probability": "Mean predicted probability",
        "fraction_of_positives": "Fraction of actual purchases",
    },
)
fig_rel.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=1,
    y1=1,
    line=dict(dash="dash", color="gray"),
)
fig_rel.update_xaxes(range=[0, 1])
fig_rel.update_yaxes(range=[0, 1])
st.plotly_chart(fig_rel, use_container_width=True)

if cal_brier is not None:
    st.markdown(
        f"""
Raw CatBoost is already fairly well calibrated on this test set
(Brier **{raw_brier:.4f}**). Isotonic calibration ends at **{cal_brier:.4f}**
({cal_brier - raw_brier:+.4f}) — so it does **not** meaningfully improve things here.
The API/dashboard keep using the raw probabilities; the calibrated model is just for comparison.
"""
    )
else:
    st.markdown(
        f"""
Raw CatBoost Brier score on this test split: **{raw_brier:.4f}**.
Compare against the diagonal above — points close to it mean probabilities are trustworthy.
"""
    )
