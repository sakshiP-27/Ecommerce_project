import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from src.dashboard.loaders import load_data, load_model, load_pipeline, load_threshold

st.set_page_config(
    page_title="Purchase-Intent Dashboard",
    page_icon="🛒",
    layout="wide",
)

st.title("Purchase-Intent Prediction Dashboard")
st.caption("Overview — headline stats from the saved dataset (cached loaders).")

# Cached loads — reused across pages / reruns (no retrain here)
df = load_data()
model = load_model()
pipeline = load_pipeline()
threshold = load_threshold()

n_sessions = len(df)
conversion_rate = df["Converted"].mean()
n_buyers = int(df["Converted"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total sessions", f"{n_sessions:,}")
col2.metric("Conversion rate", f"{conversion_rate:.1%}")
col3.metric("Purchases", f"{n_buyers:,}")
col4.metric("Decision threshold", f"{threshold:.2f}")

st.markdown(
    f"""
**Model in use:** `{type(model).__name__}` (loaded from saved artifact)
**Pipeline:** `{type(pipeline).__name__}` with
{len(pipeline.get_feature_names_out())} output features

Use the sidebar for Live Scoring, Funnel Analytics, Model Comparison,
Explainability, and Performance Metrics.
"""
)

st.success(
    "Step 3: data uses `@st.cache_data`; model/pipeline use `@st.cache_resource`. "
    "Clicking around should stay fast because artifacts are not reloaded from disk every time."
)
