import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from sklearn.model_selection import train_test_split

from src.config.settings import load_config
from src.dashboard.loaders import load_data, load_model, load_pipeline, load_threshold
from src.evaluation.drift import check_dataframe_drift
from src.features.engineering import add_engineered_features

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

# --- Stretch goal #7: simple drift status ---
st.subheader("Data drift status")
config = load_config()
df_feat = add_engineered_features(df.copy())
train_df, holdout_df = train_test_split(
    df_feat,
    test_size=config["test_size"],
    random_state=config["random_seed"],
)
control = check_dataframe_drift(train_df, holdout_df)

shifted = holdout_df.copy()
for col in ["PageValues", "BounceRates", "ExitRates", "ProductRelated"]:
    if col in shifted.columns:
        shifted[col] = shifted[col] * 5 + 10
shifted_result = check_dataframe_drift(train_df, shifted)

c_left, c_right = st.columns(2)
with c_left:
    if control["batch_drifted"]:
        st.error(f"Holdout split: {control['message']} ({control['n_drifted']} features)")
    else:
        st.success(f"Holdout split: {control['message']} ({control['n_drifted']} features flagged)")
with c_right:
    if shifted_result["batch_drifted"]:
        st.error(
            f"Simulated shift: {shifted_result['message']} "
            f"({shifted_result['n_drifted']} features) — checker works"
        )
    else:
        st.warning("Simulated shift did not flag drift (unexpected).")

st.caption(
    "KS test on numeric features vs training split. "
    "Batch drift = at least 3 features with p < 0.05. "
    "See `docs/data_drift.md`."
)
