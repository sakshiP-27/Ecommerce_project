# Week 4 Step 1 : Dashboard Section Plan

One-line plan for each section (before writing Streamlit code).

| # | Section | What it will show |
|---|---------|-------------------|
| 1 | **Overview** (`src/dashboard/Home.py`) | At-a-glance headline stats: total sessions, conversion rate, class balance, and a short project summary. |
| 2 | **Live Session Scoring** | Interactive form (manual inputs or load a random real row) that calls the model/API and shows Purchase/No Purchase, probability, and a SHAP explanation. |
| 3 | **Funnel / Cohort Analytics** | Interactive Plotly charts from Week 1 EDA: conversion by Month, VisitorType, TrafficType, Weekend, etc. |
| 4 | **Model Comparison** | Week 2 six-model metrics (esp. PR-AUC / recall) as a table + bar chart highlighting CatBoost as the chosen model. |
| 5 | **Explainability** | Week 3 SHAP summary/bar plots, example waterfalls (buyer / non-buyer / uncertain), plus the SHAP vs LIME comparison notes. |
| 6 | **Performance Metrics** | Final CatBoost detail: confusion matrix, precision–recall curve, threshold = 0.5, and the plain-English justification. |

## Priority / polish order
1. Live Session Scoring 
2. Overview
3. Explainability
4. Model Comparison + Performance Metrics
5. Funnel Analytics

## Content already available to reuse
- Dataset: `dataset/ecommerce_sessions.csv`
- Model + pipeline: `src/models/final_model.pkl`, `src/models/preprocessing_pipeline.pkl`
- API: `src/api/main.py` (`/predict`, `/health`)
- Metrics: Week 2 notebook (`03_model_evaluation.ipynb`) — CatBoost PR-AUC ~0.86, recall ~76% @ 0.5
- SHAP/LIME: `notebooks/04_explainabliity.ipynb`, `reports/shap_*.png`
- Model card: `reports/model_card.md`

