# PageValues ablation — deployment leakage (stretch goal #3)

## Experiment

Same CatBoost config as Week 2 (`n_estimators=100`, `max_depth=4`, `learning_rate=0.05`), same train/test split (`test_size=0.2`, `seed=42`), same preprocessing style.

Trained twice:

1. Full feature set **with** `PageValues`
2. Same features **without** `PageValues`

## Metrics (held-out test)

| Variant | PR-AUC | Precision @ 0.5 | Recall @ 0.5 | F1 |
|---|---:|---:|---:|---:|
| With `PageValues` | 0.8628 | 0.8177 | 0.7653 | 0.7906 |
| Without `PageValues` | 0.5545 | 0.6316 | 0.3840 | 0.4776 |

CSV copy: `reports/pagevalues_ablation.csv`

## Deployment-leakage framing

The drop is large — especially recall (0.77 → 0.38) and PR-AUC (0.86 → 0.55). That isn’t just “a good feature got removed.” `PageValues` reflects value from pages near checkout, so for **early-session live scoring** you often wouldn’t have it yet.

In that deployment setting, the **without PageValues** metrics are the more honest real-world estimate. The main project model still keeps `PageValues` because Week 1 framed the use case as **end-of-session / offline** scoring. If the product had to score visits in the first 30 seconds, I’d use (or retrain) the no-`PageValues` version instead of quoting the 0.86 PR-AUC.

## Where to look

- Notebook: `notebooks/07_pagevalues_ablation.ipynb`
- Helper: `src/evaluation/ablation.py`
- Results table: `reports/pagevalues_ablation.csv`
