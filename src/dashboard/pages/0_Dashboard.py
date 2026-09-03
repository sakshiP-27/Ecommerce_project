"""Dashboard overview (metrics + data health). Landing page is Home.py."""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from sklearn.model_selection import train_test_split

from src.config.settings import load_config
from src.dashboard.loaders import load_data, load_model, load_pipeline, load_threshold
from src.dashboard.theme import page_setup
from src.evaluation.drift import check_dataframe_drift
from src.features.engineering import add_engineered_features

page_setup("Dashboard")

df = load_data()
model = load_model()
pipeline = load_pipeline()
threshold = load_threshold()

n_sessions = len(df)
conversion_rate = float(df["Converted"].mean())
n_buyers = int(df["Converted"].sum())

st.markdown('<div class="cartiq-kicker">Dashboard</div>', unsafe_allow_html=True)
st.title("Operations overview")
st.caption("Live snapshot from the saved dataset and production model.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sessions", f"{n_sessions:,}")
col2.metric("Conversion rate", f"{conversion_rate:.1%}")
col3.metric("Purchases", f"{n_buyers:,}")
col4.metric("Default threshold", f"{threshold:.2f}")

st.markdown(
    f"""
<div class="cartiq-panel">
  <div class="cartiq-kicker">In production</div>
  <p style="margin:0;color:#C9C1B1;font-family:'Source Sans 3',sans-serif;line-height:1.55;">
    <strong style="color:#EEE9DF;">{type(model).__name__}</strong> with a fitted preprocessing pipeline
    ({len(pipeline.get_feature_names_out())} features).
    Open <em>Live Scoring</em> for a session prediction, or use the sidebar for funnels,
    model comparison, explanations, and performance.
  </p>
</div>
""",
    unsafe_allow_html=True,
)

cta1, cta2, _ = st.columns([1.15, 1.15, 2])
with cta1:
    st.page_link("pages/1_Live_Scoring.py", label="Score a session")
with cta2:
    st.page_link("pages/5_Performance_Metrics.py", label="View performance")

st.markdown('<div class="cartiq-section">', unsafe_allow_html=True)
st.markdown('<div class="cartiq-kicker">Data health</div>', unsafe_allow_html=True)
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
        st.error(f"Holdout check: {control['message']} ({control['n_drifted']} features)")
    else:
        st.success(
            f"Holdout check: {control['message']} ({control['n_drifted']} features flagged)"
        )
with c_right:
    if shifted_result["batch_drifted"]:
        st.error(
            f"Simulated shift: {shifted_result['message']} "
            f"({shifted_result['n_drifted']} features). Monitor is working"
        )
    else:
        st.warning("Simulated shift did not flag drift (unexpected).")

st.caption(
    "Drift monitor uses a KS test on numeric features vs the training split. "
    "Batch drift = at least 3 features with p < 0.05."
)
st.markdown("</div>", unsafe_allow_html=True)
