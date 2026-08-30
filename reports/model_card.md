# Model Card: Ecommerce Purchase-Intent Predictor

**Model:** CatBoost classifier (`final_model.pkl`)  
**Owner / project:** Purchase-intent prediction (ecommerce sessions)  
**Last updated:** August 2026

---

## Intended use

This model estimates how likely a website session is to end in a **purchase** (`Converted = 1`). It is meant to help flag high-intent sessions so a business can prioritize actions such as offers, live chat, or follow-up messaging.

**It is not:**
- a guarantee that a flagged visitor will buy
- a substitute for human judgment on individual customers
- a real-time credit, fraud, or pricing decision system

Treat scores as decision support, not as automatic truth.

---

## Performance summary (Week 2 test set)

Among the models compared, **CatBoost** was selected as the final model.

| Metric | Result (approx.) | Plain English |
|--------|------------------|---------------|
| **PR-AUC** | **0.86** | Strong ranking of true buyers vs non-buyers on imbalanced data (headline metric) |
| **Recall** | **~76%** (at threshold 0.5) | Catches roughly 3 in 4 actual buyers |
| **Precision** | **~82%** (at threshold 0.5) | About 4 in 5 sessions we flag as “buy” are true buyers |
| **F1** | **~0.79** | Balanced precision/recall trade-off |

Default operating point in config is **threshold = 0.5**. Lowering the threshold (e.g. 0.4) catches more buyers but increases false positives; raising it does the opposite. Choose the threshold based on the cost of a missed buyer vs the cost of acting on a false alert.

---

## Key features (from SHAP)

Across the test set, the strongest drivers of predictions are:

1. **PageValues** — high values push strongly toward purchase; zero/low values push against it  
2. **ProductRelated / ProductRelated_Duration** — more product browsing supports purchase intent  
3. **TotalPages** — broader session engagement tends to increase purchase likelihood  
4. **ExitRates** — higher exit rates push toward “no purchase”  
5. **BounceRates** — higher bounce-like behavior also reduces purchase likelihood  

Local SHAP and LIME checks on clear buyer, clear non-buyer, and borderline sessions largely agreed that **PageValues** is the dominant driver, which increases confidence in these explanations.

---

## Limitations

- **Single dataset / store context.** The model was trained on one ecommerce sessions dataset. Behavior may differ for another brand, traffic mix, season, or device mix.
- **`PageValues` leakage risk.** `PageValues` is a very strong predictor and may not be fully known early in a live visit (it reflects value near checkout). We **kept** it for offline modelling after Week 1 EDA, but a real-time deployment may need a version without it or with delayed features only.
- **Class imbalance.** Purchases are the minority class. Accuracy alone is misleading; PR-AUC/recall are the right lenses, and threshold choice still matters in production.
- **Explanations are approximate.** SHAP/LIME help interpret predictions but do not prove causation. Feature names after encoding (e.g. one-hot categories) can look technical to non-specialists.
- **Not calibrated for every business cost.** The current threshold is a modelling choice, not a guarantee of optimal ROI for every campaign.

---

## Ethical / practical note

Use this model to **support** marketing or UX decisions, not to unfairly treat individual visitors. Review false positives/negatives periodically and retrain if site content, traffic sources, or conversion behavior change.
