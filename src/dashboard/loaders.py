"""
Cached loaders for the Streamlit dashboard.

- @st.cache_data  → dataset (DataFrame)
- @st.cache_resource → model / pipeline / SHAP explainer (heavy objects)

Dashboard only READS Week 1–3 artifacts — never retrains or refits.
"""

import sys
from pathlib import Path

# Project root must be on path before any `src.*` imports
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = _PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "src" / "models"
DATA_PATH = PROJECT_ROOT / "dataset" / "ecommerce_sessions.csv"


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the ecommerce sessions CSV once and reuse it."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    """Load the final CatBoost model once and reuse it."""
    return joblib.load(MODELS_DIR / "final_model.pkl")


@st.cache_resource
def load_pipeline():
    """Load the fitted preprocessing pipeline once and reuse it."""
    return joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")


@st.cache_resource
def load_explainer():
    """Build the SHAP TreeExplainer once from the cached model."""
    import shap

    model = load_model()
    return shap.TreeExplainer(model)


@st.cache_resource
def load_calibrated_model():
    """Load isotonic-calibrated wrapper if the stretch-goal artifact exists."""
    path = MODELS_DIR / "calibrated_model.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_threshold() -> float:
    from src.config.settings import load_config

    return float(load_config().get("threshold", 0.5))
