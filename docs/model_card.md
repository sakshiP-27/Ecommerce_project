# Model Card: Ecommerce Purchase-Intent Predictor

**Model:** CatBoost classifier (`final_model.pkl`)  
**Project:** Purchase-intent prediction from ecommerce sessions  

---

## Intended use

This model estimates how likely a website session is to end in a **purchase** (`Converted = 1`). I’d use it to flag high-intent sessions so a shop could prioritise things like offers, chat, or follow-ups.

**What it’s not for**
- not a guarantee someone will buy
- not a replacement for human judgment on one customer
- not a fraud / credit / pricing system

Treat the score as decision support.

---

## Performance summary (Week 2 test set)

I compared a few models and shipped **CatBoost**.

| Metric | Approx. result | Plain English |
|--------|----------------|---------------|
| **PR-AUC** | **0.86** | Main metric I cared about (imbalanced data) |
| **Recall** | **~76%** @ 0.5 | Catches about 3 in 4 actual buyers |
| **Precision** | **~82%** @ 0.5 | About 4 in 5 “Purchase” flags are real buyers |
| **F1** | **~0.79** | Decent balance of the two |

Threshold in config is **0.5**. Lower it to catch more buyers (more false alarms). Raise it if false alarms are expensive.

---

## Key features (from SHAP)

What usually moves the prediction:

1. **PageValues** — high → purchase; zero/low → against purchase  
2. **ProductRelated / duration** — more product browsing helps  
3. **TotalPages** — more engagement helps  
4. **ExitRates** — higher exit → less likely to buy  
5. **BounceRates** — similar story  

On a few example sessions, SHAP and LIME both pointed at **PageValues** a lot, which made me more confident the explanation wasn’t just one method being weird.

---

## Limitations (honest ones)

- Trained on **one** dataset — another store/season might behave differently.
- **`PageValues` leakage concern:** it’s a strong feature and may not be fully known early in a live visit. I kept it for this offline project after Week 1 EDA, but a real-time system might need a version without it.
- Class imbalance: don’t judge the model on accuracy alone.
- SHAP/LIME explain the model; they don’t prove real-world causation.
- Threshold 0.5 is a project choice, not guaranteed “best ROI” for every business.

---
