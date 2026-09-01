# Architecture

Simple view of how the pieces connect. Nothing fancy — just the flow I actually built.

```text
┌──────────────────────┐
│ ecommerce_sessions   │
│        .csv          │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Feature engineering  │  TotalPages, AvgTimePerPage,
│ (Week 1)             │  ProductEngagementRatio, etc.
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Preprocessing        │  OneHotEncoder + RobustScaler
│ pipeline.pkl         │  (handle once on train)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ CatBoost model       │  final_model.pkl
│ (Week 2)             │  threshold = 0.5
└──────────┬───────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
┌─────────┐  ┌────────────────┐
│ FastAPI │  │ Streamlit      │
│ /predict│  │ dashboard      │
│ /health │  │ (6 pages)      │
│ + SHAP  │  │ live scoring   │
└─────────┘  │ uses same      │
             │ model/pipeline │
             └────────────────┘
```

## How it runs in Docker

```text
docker compose up --build
        │
        ├── api service        → port 8000  (uvicorn)
        └── frontend service   → port 8501  (streamlit)
                 │
                 └── depends_on api (starts after API container)
```

Both images install from `requirements.txt` and copy `src/` (plus `dataset/` + `reports/` for the dashboard).

## Explainability

SHAP `TreeExplainer` is built from the CatBoost model.

- API: returns top contributing features with `/predict`
- Dashboard Live Scoring: shows waterfall + top features for the session you score
- Explainability page: shows the global SHAP plots saved under `reports/`

## Design choice I stuck to

The dashboard **does not retrain** anything. It only reads the Week 2/3 artifacts. Training stays in the notebooks/scripts.
