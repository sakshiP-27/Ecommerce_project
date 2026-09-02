# Probability calibration (stretch goal #2)

## What I checked

Whether CatBoost’s predicted probabilities match real conversion rates (reliability curve), then whether isotonic `CalibratedClassifierCV` improves that.

## Result (same train/test split as training)

| Version | Brier score (lower is better) |
|---|---|
| Raw CatBoost | ~0.0482 |
| Isotonic calibrated | ~0.0484 |

Calibration does **not** meaningfully help on this dataset — the raw model is already close enough to the diagonal that wrapping it barely moves Brier (and can be slightly worse).

API / Live Scoring still use the **raw** probabilities. The calibrated artifact is for comparison only.

## Where to look

- Notebook: `notebooks/06_calibration.ipynb`
- Dashboard: **Performance Metrics** → Reliability curve section
- Helper: `src/evaluation/calibration.py`
- Optional artifact: `src/models/calibrated_model.pkl`
