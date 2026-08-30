# Week 2 Guide: Purchase-Intent Prediction Project
### A beginner-friendly walkthrough — Modelling & Evaluation

---

## Before You Start

**Goal for this week:** Turn last week's "thin, rough" pipeline into a real one — engineer a few smart features, train and compare six different models properly, and pick metrics and a threshold that actually make business sense (not just "highest accuracy").

**Why this matters:** This is the week where the actual "machine learning" happens. But the biggest mistake beginners make here is chasing accuracy. In this dataset, only ~16% of sessions convert — so a model that predicts "no purchase" every single time would already be ~84% accurate, while being completely useless. This week is as much about *measuring correctly* as it is about *training well*.

**What you need before starting:**
- Last week's fitted preprocessing pipeline (saved with joblib)
- Last week's EDA notebook and its findings, especially your decision on `PageValues`
- MLflow installed (`pip install mlflow`) — this is a free tool that logs and compares your experiments so you don't have to track results in a spreadsheet

---

## Step 1: Engineer a Few New Features

**What this means:** Beyond the raw columns in the CSV, you can create a few new columns that capture patterns a model might not "see" on its own — combinations or ratios of existing columns.

**Why it matters:** Good features often help a model more than switching to a fancier algorithm does. A few thoughtful features usually beat ten random ones.

**A few good candidates to try (don't overdo it — 3 to 5 is plenty):**
1. **Total pages viewed** — sum of `Administrative`, `Informational`, and `ProductRelated` page counts. Captures overall session engagement in one number.
2. **Average time per page** — total duration ÷ total pages viewed. Captures whether someone is skimming or reading carefully.
3. **Product-engagement ratio** — `ProductRelated` pages ÷ total pages. Captures whether the visit was mostly product-browsing versus admin/info pages.
4. **Returning-visitor flag** — you may already have `VisitorType`, but a simple 0/1 version can sometimes help simpler models pick up the signal more easily.

**How to do it:** Add this as a small function in `src/features/`, so it's reusable — not just typed once into a notebook cell. It should take the raw DataFrame in and return it with the new columns added.

```python
def add_engineered_features(df):
    df["TotalPages"] = df["Administrative"] + df["Informational"] + df["ProductRelated"]
    df["AvgTimePerPage"] = (
        (df["Administrative_Duration"] + df["Informational_Duration"] + df["ProductRelated_Duration"])
        / df["TotalPages"].replace(0, 1)  # avoid dividing by zero
    )
    df["ProductEngagementRatio"] = df["ProductRelated"] / df["TotalPages"].replace(0, 1)
    return df
```

**Important:** Call this function *before* your preprocessing pipeline from last week, so the new columns get scaled/encoded along with everything else.

✅ **Done when:** You have 3–5 new, clearly-named feature columns, created by a reusable function, not copy-pasted notebook code.

---

## Step 2: Confirm Which Features Actually Matter

**What this means:** Once your features exist, check whether they're actually useful before locking them in — using two different methods, so you're not trusting just one opinion.

**Two simple ways to check:**
1. **Model-based importance** — train a quick Random Forest and look at `.feature_importances_`. Fast, but can be biased toward high-cardinality columns.
2. **Permutation importance** — scikit-learn's `permutation_importance` shuffles one column at a time and sees how much performance drops. Slower, but more trustworthy.

**How to do it:** Run both on your training data, list the top 10–15 features from each, and write one paragraph in your notebook: do the two methods roughly agree? Are your new engineered features showing up as useful, or not really?

✅ **Done when:** You have a short, written justification for your final feature set — not just "I used everything."

---

## Step 3: Build One Common "Trainer" So You're Not Repeating Code Six Times

**What this means:** You're about to train six different models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost). Instead of writing near-identical training code six separate times, write **one function or class** that takes "which model" as an input, and handles training + evaluation the same way for all of them.

**Why it matters:** If your training and evaluation code is even slightly different between models, your comparison between them becomes unfair and unreliable. One shared trainer guarantees a fair, apples-to-apples comparison.

**Rough shape of what you're building:**

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

def train_and_evaluate(model, X_train, y_train, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring=["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    )
    return scores
```

**A note on `StratifiedKFold`:** "Stratified" means each fold keeps roughly the same 16%-conversion ratio as the full dataset. Without this, some folds could randomly end up with almost no buyers at all, giving you misleading results — this matters a lot on imbalanced data like this.

✅ **Done when:** You can train any of the six models by changing one line of code, and every model goes through the exact same evaluation process.

---

## Step 4: Train All Six Models

**What this means:** Run your trainer from Step 3 on each of the six required models: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost.

**Why six, and why these?** They represent a spread — a simple linear model (Logistic Regression), a simple tree (Decision Tree), an ensemble of trees (Random Forest), and three modern "gradient boosting" models (XGBoost, LightGBM, CatBoost) that usually perform best on tabular data like this one. Comparing across this spread shows you actually understand the trade-offs, not just that you can call `.fit()`.

**Practical tips:**
- Install what you need: `pip install xgboost lightgbm catboost`.
- Start with each model's **default settings** first, just to get a baseline for all six. Only tune hyperparameters (Step 5) on your top 2–3 performers — tuning all six in depth is a waste of a week.
- Keep every model's results (not just the best one) — you'll need to show the comparison later, not just declare a winner.

✅ **Done when:** You have baseline cross-validated results for all six models, saved somewhere (a small CSV or table is fine) so you can compare them side by side.

---

## Step 5: Tune Your Best 2–3 Models

**What this means:** Once you can see which models look strongest from Step 4, do a focused search over a few key hyperparameters for just those top performers.

**How to do it:** Use `RandomizedSearchCV` rather than a full `GridSearchCV` — it checks a random sample of combinations instead of every single one, which is far faster and usually finds a similarly good result.

A few parameters worth tuning per model type (don't tune everything — pick 2–4 per model):
- **Tree-based models** (Random Forest, XGBoost, LightGBM, CatBoost): `max_depth`, `n_estimators`, `learning_rate` (for boosting models)
- **Logistic Regression:** `C` (regularization strength)

✅ **Done when:** Your top 2–3 models have been tuned, and you can show a "before tuning vs. after tuning" comparison for at least one of them.

---

## Step 6: Report the Right Metrics (This Is the Most Important Step)

**What this means:** For every model, report a full set of metrics — not just accuracy. Then explicitly explain *why* accuracy is misleading here.

**The metrics to report for every model:**
| Metric | What it tells you |
|---|---|
| Accuracy | % of all predictions correct — **misleading here, report but don't lead with it** |
| Precision (buying class) | Of sessions predicted "will buy," how many actually did |
| Recall (buying class) | Of sessions that actually bought, how many did the model catch |
| F1 Score | Balance between precision and recall |
| ROC-AUC | How well the model ranks buyers above non-buyers overall |
| **PR-AUC** | Like ROC-AUC, but far more informative on imbalanced data — **this is your headline metric** |
| Confusion Matrix | The actual counts of correct/incorrect predictions, broken down |

**Why PR-AUC over accuracy:** Imagine a model that just predicts "no purchase" for every single session. It would score ~84.5% accuracy — and be worthless. PR-AUC doesn't get fooled by this, because it specifically measures how well you're finding the rare "buy" cases, which is the entire point of this project.

**How to present this well:** Make one summary table with all six models as rows and all these metrics as columns. Bold or highlight the best PR-AUC. In your write-up, lead with a sentence like: *"XGBoost achieved the best PR-AUC of 0.61, correctly identifying 78% of actual buyers (recall) while keeping false positives manageable."* — not *"Random Forest had 88% accuracy."*

✅ **Done when:** You have one clean comparison table for all six models, and your written summary leads with PR-AUC/recall, not accuracy.

---

## Step 7: Choose a Real Decision Threshold

**What this means:** By default, models predict "buy" if the probability is above 0.5. But 0.5 is an arbitrary number — it's rarely the *right* number for a real business decision. This step is about picking a threshold on purpose, and explaining why.

**Why it matters:** Imagine your model will trigger something real — say, an email discount offer to sessions predicted to buy. If you set the threshold too low, you'll waste discounts on people who were never going to leave anyway. Too high, and you'll miss real potential buyers. The "right" threshold depends on what it costs to act on a false positive versus what you lose from missing a true buyer.

**How to do it:**
1. Plot the **precision-recall curve** for your best model (scikit-learn has `precision_recall_curve` built in).
2. Pick a few candidate thresholds along that curve (e.g. 0.3, 0.4, 0.5, 0.6).
3. For each, write out in plain English what it would mean in practice — e.g. *"At threshold 0.35, we catch 82% of buyers but only 1 in 3 people we flag actually buys."*
4. Pick one, and write one paragraph justifying your choice as a business trade-off, not just a technical one.

✅ **Done when:** You've picked a threshold that isn't 0.5-by-default, and you can explain the trade-off behind it in a sentence a non-technical person would understand.

---

## Step 8: Log Everything to MLflow

**What this means:** MLflow automatically records the parameters, metrics, and model file for every training run you do, so you can compare experiments later without keeping your own spreadsheet.

**How to do it — minimal version:**

```python
import mlflow

mlflow.set_experiment("purchase-intent-prediction")

with mlflow.start_run(run_name="xgboost_baseline"):
    mlflow.log_params({"max_depth": 5, "n_estimators": 200})
    mlflow.log_metrics({"pr_auc": 0.61, "recall": 0.78, "f1": 0.55})
    mlflow.sklearn.log_model(model, "model")
```

Do this for every model you train in Step 4 and Step 5 — one run per model/configuration. Afterward, run `mlflow ui` in your terminal to see a visual dashboard comparing every run side by side.

**Then register your best model:** MLflow lets you formally mark one specific run's model as "the" chosen model (called the Model Registry). This is the exact model file your API will load in Week 3, so this step isn't just record-keeping — it's what connects this week's work to next week's.

✅ **Done when:** Every model + configuration you trained shows up in `mlflow ui`, and your final chosen model is registered.

---

## Step 9: Save Your Final Artifacts

**What this means:** Save the two things Week 3 will need: your final trained model, and your final fitted preprocessing pipeline (from Week 1, possibly re-fit if you added new features).

**How to do it:**
```python
import joblib

joblib.dump(final_model, "models/final_model.pkl")
joblib.dump(preprocessing_pipeline, "models/preprocessing_pipeline.pkl")
```

Keep the version number or date in the filename if you retrain later, so you don't accidentally overwrite a working model with a worse one.

✅ **Done when:** You have one final model file and one final pipeline file saved to disk, and you could hand just those two files to someone else and they'd be able to make a prediction.

---

## Week 2 Checklist — Quick Reference

- [ ] 3–5 engineered features created, via a reusable function
- [ ] Feature importance checked two ways, final feature set justified in writing
- [ ] One shared trainer function used for all models (fair comparison)
- [ ] All six models trained with baseline settings, results saved for comparison
- [ ] Top 2–3 models tuned with `RandomizedSearchCV`
- [ ] Full metrics table built for all six models — PR-AUC and recall as headline metrics, not accuracy
- [ ] Decision threshold chosen deliberately and justified in plain English
- [ ] All runs logged to MLflow, best model registered
- [ ] Final model + pipeline saved to disk

---

## A Few Words of Encouragement

It's tempting to spend all week chasing a slightly higher accuracy number. Resist that. A model with honestly-reported, well-understood metrics and a clearly justified threshold will score better than one with a marginally higher accuracy and no explanation behind it. The graders are checking whether she *understands* what the numbers mean — not just whether the numbers are high.

If a model like CatBoost or LightGBM throws a confusing installation error, that's common — it's almost always a missing system dependency, not a mistake in the code. Worth Googling the exact error message rather than assuming something's fundamentally wrong.

Good luck with Week 2! 🎯
