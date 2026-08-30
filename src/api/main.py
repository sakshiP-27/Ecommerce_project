"""
FastAPI service for purchase-intent prediction + SHAP explanations.
Run from project root:
    uvicorn src.api.main:app --reload
Then open http://localhost:8000/docs
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config.settings import load_config
from src.features.engineering import add_engineered_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "src" / "models"

config = load_config()
THRESHOLD = float(config.get("threshold", 0.5))

# Load once at startup (not per request)
model = joblib.load(MODELS_DIR / "final_model.pkl")
pipeline = joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")
explainer = shap.TreeExplainer(model)
FEATURE_NAMES = list(pipeline.get_feature_names_out())

app = FastAPI(
    title="Ecommerce Purchase-Intent API",
    description="Predict purchase likelihood for a session and return top SHAP feature contributions.",
    version="1.0.0",
)


class SessionInput(BaseModel):
    """Raw session fields (before engineered features)."""

    Administrative: int
    Administrative_Duration: float
    Informational: int
    Informational_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float = Field(ge=0.0, le=1.0)
    Month: str
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: str
    Weekend: bool


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    threshold: float
    top_features: list[FeatureContribution]


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _purchase_shap_row(transformed) -> np.ndarray:
    """SHAP values for the purchase class, shape (n_features,)."""
    dense = _to_dense(transformed)
    shap_values = explainer.shap_values(dense)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    row = np.asarray(shap_values)
    if row.ndim == 2:
        row = row[0]
    return row


def _top_features(shap_row: np.ndarray, k: int = 5) -> list[FeatureContribution]:
    order = np.argsort(np.abs(shap_row))[::-1][:k]
    return [
        FeatureContribution(
            feature=FEATURE_NAMES[i],
            shap_value=round(float(shap_row[i]), 4),
        )
        for i in order
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(session: SessionInput):
    df = pd.DataFrame([session.model_dump()])
    df = add_engineered_features(df)
    transformed = pipeline.transform(df)

    probability = float(model.predict_proba(transformed)[0][1])
    prediction = "Purchase" if probability >= THRESHOLD else "No Purchase"

    shap_row = _purchase_shap_row(transformed)
    top_features = _top_features(shap_row, k=5)

    return PredictResponse(
        prediction=prediction,
        probability=round(probability, 3),
        threshold=THRESHOLD,
        top_features=top_features,
    )
