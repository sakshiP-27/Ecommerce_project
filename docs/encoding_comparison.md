# Encoding comparison — one-hot vs CatBoostEncoder (stretch goal #4)

## Setup

Same CatBoost config and train/test split as Week 2.

- **one_hot:** all categoricals one-hot (current project approach)
- **catboost_encoder_ids:** `CatBoostEncoder` on `OperatingSystems`, `Browser`, `Region`, `TrafficType` only; other categoricals still one-hot; numerics (including encoded IDs) scaled

Uses `category_encoders.CatBoostEncoder` when available; otherwise the equivalent local `CatBoostStyleEncoder` in `src/evaluation/encoding_compare.py` (needed on some Anaconda setups where `category_encoders` conflicts with sklearn/scipy).

## Results

| Encoding | PR-AUC | Precision | Recall | F1 | Train time (s) | Features out |
|---|---:|---:|---:|---:|---:|---:|
| one_hot | 0.8628 | 0.8177 | 0.7653 | 0.7906 | 0.783 | 79 |
| catboost_encoder_ids | 0.8646 | 0.8106 | 0.7760 | 0.7929 | 0.712 | 33 |

CSV: `reports/encoding_comparison.csv`

## Conclusion

**Keep CatBoost encoding for the four ID columns** when rebuilding preprocessing — slightly better PR-AUC, a bit faster, much smaller feature matrix. The shipped pipeline stays one-hot for stability (gap is only ~0.002 PR-AUC).

## Where to look

- Notebook: `notebooks/08_encoding_comparison.ipynb`
- Helper: `src/evaluation/encoding_compare.py`
