import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.dashboard.loaders import (
    load_data,
    load_explainer,
    load_model,
    load_pipeline,
    load_threshold,
)
from src.features.engineering import add_engineered_features

st.set_page_config(page_title="Live Session Scoring", layout="wide")
st.title("Live Session Scoring")
st.markdown(
    "Enter session details (or load a real row), then get a **live prediction** "
    "and a **SHAP explanation** of why the model said that."
)

df = load_data()
model = load_model()
pipeline = load_pipeline()
explainer = load_explainer()
threshold = load_threshold()
feature_names = list(pipeline.get_feature_names_out())

RAW_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

MONTHS = sorted(df["Month"].dropna().unique().tolist())
VISITOR_TYPES = sorted(df["VisitorType"].dropna().unique().tolist())
OS_OPTS = sorted(df["OperatingSystems"].dropna().unique().tolist())
BROWSER_OPTS = sorted(df["Browser"].dropna().unique().tolist())
REGION_OPTS = sorted(df["Region"].dropna().unique().tolist())
TRAFFIC_OPTS = sorted(df["TrafficType"].dropna().unique().tolist())

DEFAULTS = {
    "Administrative": 1,
    "Administrative_Duration": 14.65,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 19,
    "ProductRelated_Duration": 283.88,
    "BounceRates": 0.008,
    "ExitRates": 0.042,
    "PageValues": 68.58,
    "SpecialDay": 0.0,
    "Month": "June" if "June" in MONTHS else MONTHS[0],
    "OperatingSystems": int(OS_OPTS[0]),
    "Browser": int(BROWSER_OPTS[0]),
    "Region": int(REGION_OPTS[0]),
    "TrafficType": int(TRAFFIC_OPTS[0]),
    "VisitorType": "Returning_Visitor"
    if "Returning_Visitor" in VISITOR_TYPES
    else VISITOR_TYPES[0],
    "Weekend": False,
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(f"score_{key}", value)


def _set_form_from_row(row: pd.Series) -> None:
    for col in RAW_COLS:
        val = row[col]
        if col == "Weekend":
            st.session_state[f"score_{col}"] = bool(val)
        elif col in {
            "Administrative",
            "Informational",
            "ProductRelated",
            "OperatingSystems",
            "Browser",
            "Region",
            "TrafficType",
        }:
            st.session_state[f"score_{col}"] = int(val)
        elif col in {"Month", "VisitorType"}:
            st.session_state[f"score_{col}"] = str(val)
        else:
            st.session_state[f"score_{col}"] = float(val)


col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Load a random real session", use_container_width=True):
        sample = df.sample(1, random_state=None).iloc[0]
        _set_form_from_row(sample)
        st.session_state["score_actual"] = int(sample["Converted"])
        st.rerun()
with col_b:
    if st.button("Reset form defaults", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[f"score_{key}"] = value
        st.session_state.pop("score_actual", None)
        st.rerun()

with st.form("session_form"):
    st.subheader("Session inputs")
    c1, c2, c3 = st.columns(3)

    with c1:
        administrative = st.number_input(
            "Administrative",
            min_value=0,
            max_value=50,
            step=1,
            key="score_Administrative",
        )
        administrative_duration = st.number_input(
            "Administrative_Duration",
            min_value=0.0,
            step=0.1,
            key="score_Administrative_Duration",
        )
        informational = st.number_input(
            "Informational",
            min_value=0,
            max_value=50,
            step=1,
            key="score_Informational",
        )
        informational_duration = st.number_input(
            "Informational_Duration",
            min_value=0.0,
            step=0.1,
            key="score_Informational_Duration",
        )
        product_related = st.number_input(
            "ProductRelated",
            min_value=0,
            max_value=300,
            step=1,
            key="score_ProductRelated",
        )
        product_related_duration = st.number_input(
            "ProductRelated_Duration",
            min_value=0.0,
            step=0.1,
            key="score_ProductRelated_Duration",
        )

    with c2:
        bounce_rates = st.number_input(
            "BounceRates",
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.4f",
            key="score_BounceRates",
        )
        exit_rates = st.number_input(
            "ExitRates",
            min_value=0.0,
            max_value=1.0,
            step=0.001,
            format="%.4f",
            key="score_ExitRates",
        )
        page_values = st.number_input(
            "PageValues",
            min_value=0.0,
            step=0.1,
            key="score_PageValues",
        )
        special_day = st.slider(
            "SpecialDay",
            min_value=0.0,
            max_value=1.0,
            step=0.2,
            key="score_SpecialDay",
        )
        month = st.selectbox("Month", options=MONTHS, key="score_Month")
        visitor_type = st.selectbox(
            "VisitorType", options=VISITOR_TYPES, key="score_VisitorType"
        )

    with c3:
        operating_systems = st.selectbox(
            "OperatingSystems", options=OS_OPTS, key="score_OperatingSystems"
        )
        browser = st.selectbox("Browser", options=BROWSER_OPTS, key="score_Browser")
        region = st.selectbox("Region", options=REGION_OPTS, key="score_Region")
        traffic_type = st.selectbox(
            "TrafficType", options=TRAFFIC_OPTS, key="score_TrafficType"
        )
        weekend = st.checkbox("Weekend", key="score_Weekend")

    submitted = st.form_submit_button(
        "Score this session", type="primary", use_container_width=True
    )

if submitted:
    session = {
        "Administrative": int(administrative),
        "Administrative_Duration": float(administrative_duration),
        "Informational": int(informational),
        "Informational_Duration": float(informational_duration),
        "ProductRelated": int(product_related),
        "ProductRelated_Duration": float(product_related_duration),
        "BounceRates": float(bounce_rates),
        "ExitRates": float(exit_rates),
        "PageValues": float(page_values),
        "SpecialDay": float(special_day),
        "Month": month,
        "OperatingSystems": int(operating_systems),
        "Browser": int(browser),
        "Region": int(region),
        "TrafficType": int(traffic_type),
        "VisitorType": visitor_type,
        "Weekend": bool(weekend),
    }

    row = pd.DataFrame([session])
    row = add_engineered_features(row)
    transformed = pipeline.transform(row)
    if hasattr(transformed, "toarray"):
        dense = transformed.toarray()
    else:
        dense = np.asarray(transformed)

    probability = float(model.predict_proba(transformed)[0][1])
    prediction = "Purchase" if probability >= threshold else "No Purchase"

    shap_values = explainer.shap_values(dense)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_row = np.asarray(shap_values)
    if shap_row.ndim == 2:
        shap_row = shap_row[0]

    base_value = explainer.expected_value
    if not np.isscalar(base_value):
        base_value = base_value[1]

    st.divider()
    st.subheader("Prediction result")

    m1, m2, m3 = st.columns(3)
    m1.metric("Prediction", prediction)
    m2.metric("Purchase probability", f"{probability:.1%}")
    m3.metric("Threshold", f"{threshold:.2f}")

    st.progress(min(max(probability, 0.0), 1.0), text=f"P(Purchase) = {probability:.3f}")

    if "score_actual" in st.session_state:
        actual = "Purchase" if st.session_state["score_actual"] == 1 else "No Purchase"
        st.caption(f"Loaded row actual label (for reference): **{actual}**")

    # Top SHAP contributions table
    order = np.argsort(np.abs(shap_row))[::-1][:8]
    top_df = pd.DataFrame(
        {
            "feature": [feature_names[i] for i in order],
            "shap_value": [round(float(shap_row[i]), 4) for i in order],
        }
    )

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("#### Why? (SHAP waterfall)")
        explanation = shap.Explanation(
            values=shap_row,
            base_values=base_value,
            data=dense[0],
            feature_names=feature_names,
        )
        fig = plt.figure()
        shap.plots.waterfall(explanation, max_display=10, show=False)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

    with right:
        st.markdown("#### Top feature contributions")
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.info(
            "Positive SHAP values push toward **Purchase**; "
            "negative values push toward **No Purchase**."
        )
