# Data-drift check (Goal #7)

## What it does

For each numeric feature, compare the **training** distribution vs a **new batch** with a Kolmogorov–Smirnov test (`scipy.stats.ks_2samp`).

- Per feature: drift if `p_value < 0.05`
- Whole batch: drift if **≥ 3** features are flagged
- Logs a warning when batch drift is detected

## Code

- `src/evaluation/drift.py` — `check_drift`, `check_dataframe_drift`
- Self-test: `python -m src.evaluation.run_drift_self_test`
- Notebook: `notebooks/09_data_drift.ipynb`

## Deliberate-shift test

| Batch | Expected | Result |
|---|---|---|
| Normal holdout (same dataset split) | No batch drift | Passes |
| Holdout with `PageValues` / bounce / exit / product pages shifted (`×5 + 10`) | Batch drift flagged | Passes |

## Dashboard

Home page shows a green/red indicator for:
- consistency check on the held-out split
- a simulated shifted batch (to prove the red path works)
