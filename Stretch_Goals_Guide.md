# Stretch Goals Guide: Purchase-Intent Prediction Project
### A beginner-friendly walkthrough — Optional, Post-Core Enhancements

---

## Before You Start

**Important context:** Everything in this doc is **optional**. The core project (Weeks 1–4) is already a complete, submittable project on its own. These are portfolio differentiators — things that make the project stand out, not things required to pass. Only start here once the core is fully finished, working, and documented.

**How to use this doc:** Unlike Weeks 1–4, these seven items don't need to be done in order, and don't need to all be done. Pick whichever ones fit the time left and sound genuinely interesting — doing 2 of these well is worth more than rushing through all 7.

**Suggested priority, if time is limited:**
1. **PageValues ablation** (#3) and **MLflow versioning** (#5) — cheapest, lowest-risk, build directly on work already done.
2. **Profit-curve optimizer** (#1) and **Calibration** (#2) — moderate effort, strong differentiators.
3. **Encoding comparison** (#4) and **Data-drift check** (#7) — solid additions if there's still time.
4. **Real AWS deployment** (#6) — most impressive, but highest risk of eating unexpected time. Only attempt if the rest is solid and there's real time left.

---

## 1. Profit-Curve / Expected-Value Optimizer

**What it is, in plain terms:** Right now, the model outputs a probability, and Week 2 picked a threshold based on precision/recall trade-offs. This stretch goal takes it one step further — converting that trade-off into actual money, by asking: *"if we act on this prediction, what does it cost us when we're wrong, and what do we gain when we're right?"*

**Why it's a strong differentiator:** Most student projects stop at "here's the ROC curve." This turns the whole exercise into a business decision, which is exactly what a real company would actually care about. It shows she can connect ML output to business impact, not just model output to a metric.

**How to do it, step by step:**
1. Define two numbers: **intervention cost** (what it costs to act on a session flagged as "will buy" — e.g. cost of showing a discount) and **conversion value** (what one actual purchase is worth to the business).
2. For every possible threshold, calculate:
   - True positives × conversion value (money gained from correctly caught buyers)
   - False positives × intervention cost (money spent on people who weren't going to buy anyway)
   - Net expected value = true positive gains − false positive costs
3. Plot **net expected value vs. threshold** — this is your "profit curve." It'll usually look like a hill: value rises, peaks somewhere, then falls again.
4. The threshold at the peak is your **profit-optimal threshold** — compare it to the threshold chosen in Week 2 and discuss the difference.

**A simple version in code:**

```python
def expected_value(y_true, y_probs, threshold, intervention_cost, conversion_value):
    preds = (y_probs >= threshold).astype(int)
    true_positives = ((preds == 1) & (y_true == 1)).sum()
    false_positives = ((preds == 1) & (y_true == 0)).sum()
    return (true_positives * conversion_value) - (false_positives * intervention_cost)

thresholds = [i / 100 for i in range(101)]
values = [expected_value(y_test, y_probs, t, intervention_cost=5, conversion_value=50) for t in thresholds]
```

✅ **Done when:** You have a profit-curve chart, a clearly stated "optimal" threshold from it, and a short paragraph comparing it to the Week 2 threshold.

---

## 2. Probability Calibration + Reliability Curve

**What it is, in plain terms:** When the model says "70% probability of purchase," does that actually mean that, out of all sessions it said that about, 70% really did purchase? Often, the honest answer is no — models can be systematically over- or under-confident. Calibration checks this, and fixes it if needed.

**Why it's a strong differentiator:** This is a subtle but important distinction most beginners never think to check. Showing awareness of it signals real statistical maturity, not just "I called `.fit()` and `.predict()`."

**How to do it, step by step:**
1. Use scikit-learn's `calibration_curve` to check: for sessions the model predicted "0.7," what fraction actually converted? Do this across several probability bins (0.0–0.1, 0.1–0.2, etc.).
2. Plot this as a **reliability curve** — predicted probability on the x-axis, actual observed rate on the y-axis. A perfectly calibrated model follows the diagonal line exactly.
3. If the curve deviates noticeably from the diagonal, try scikit-learn's `CalibratedClassifierCV`, which adjusts the model's raw output to be better calibrated (two common methods: Platt scaling or isotonic regression — the tool handles the details, you just pick one and compare before/after).
4. Add the reliability curve as a chart on the Week 4 dashboard's Performance Metrics page.

**A simple version in code:**

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

prob_true, prob_pred = calibration_curve(y_test, y_probs, n_bins=10)
# plot prob_pred (x) vs prob_true (y), plus a diagonal reference line

calibrated_model = CalibratedClassifierCV(final_model, method="isotonic", cv=5)
calibrated_model.fit(X_train, y_train)
```

✅ **Done when:** You have a before/after reliability curve, and a sentence stating whether calibration meaningfully changed anything for this dataset.

---

## 3. Rigorous PageValues Ablation

**What it is, in plain terms:** Back in Week 1, a judgment call was made about whether `PageValues` is fair to use, since it reflects late-session behavior close to checkout. This stretch goal turns that judgment call into a proper experiment: train the model twice — once with `PageValues`, once without — and honestly report how much performance depends on it.

**Why it's a strong differentiator:** This directly extends a decision already made and documented in Week 1, so it's low-effort relative to how good it looks — it shows the project isn't just using a feature because it happened to score well, but actually understanding *why* it scores well, and being honest about the trade-off.

**How to do it, step by step:**
1. Take the exact same final model configuration from Week 2.
2. Train it once on the full feature set (including `PageValues`), and once on the same features minus `PageValues`.
3. Compare PR-AUC, recall, and precision between the two versions, using the same train/test split for a fair comparison.
4. Write this up explicitly as a **deployment-leakage analysis**: if the real-world use case is scoring sessions *early*, before `PageValues` is meaningfully populated, then the "without PageValues" version might be the *actually honest* number for real-world performance — even if it looks worse on paper.

✅ **Done when:** You have a clear before/after metrics table, and a paragraph framing the gap as a leakage/deployment-realism issue, not just "removing a feature made it worse."

---

## 4. Target/CatBoost Encoding vs. One-Hot

**What it is, in plain terms:** Back in Week 1, the four high-cardinality ID columns (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) were one-hot encoded — turned into many 0/1 columns. This stretch goal tries a different encoding method and compares it honestly.

**What target/CatBoost encoding means:** Instead of creating a new column for every category, each category gets replaced with a single number — roughly, "the average conversion rate for sessions with this category." CatBoost encoding is a more careful version of this idea that avoids leaking information from the target in a naive way (plain "target encoding" can overfit if done carelessly; CatBoost's version handles this more safely).

**Why it's worth trying:** One-hot encoding on high-cardinality columns creates lots of new columns, which can dilute a model's ability to find patterns. Target-style encoding keeps things compact. Comparing the two, rather than just picking one, shows a deliberate, evidence-based choice.

**How to do it, step by step:**
1. Use `category_encoders` library (`pip install category_encoders`) — it has a ready-made `CatBoostEncoder`.
2. Build a second version of the preprocessing pipeline using this encoder instead of one-hot, just for the four ID columns.
3. Retrain your best model from Week 2 on both versions, using identical settings otherwise.
4. Compare PR-AUC and training time between the two. Report which one performed better, and by how much.

✅ **Done when:** You have a side-by-side comparison table (one-hot vs. target/CatBoost encoding) and a one-sentence conclusion on which to keep.

---

## 5. Model Versioning + Rollback via MLflow Registry

**What it is, in plain terms:** Week 2 already logs runs to MLflow and registers the best model. This stretch goal builds a proper *workflow* around that registry — formally moving models through stages (e.g. "Staging" → "Production"), and having a documented way to roll back to a previous version if a new one turns out to be worse.

**Why it's worth doing:** This is a real-world MLOps practice, not just an academic exercise — companies genuinely need a way to undo a bad model deployment quickly. It also builds directly on work already done in Week 2, so it's a relatively small addition.

**How to do it, step by step:**
1. In the MLflow UI or via code, formally transition your current best model to a **"Production"** stage in the Model Registry (MLflow supports named stages like `Staging`, `Production`, `Archived`).
2. Train one more model version (even a small tweak), register it as a new version of the same model name, and move it to `"Staging"`.
3. Write a short script or set of steps showing how you'd **promote** the staging model to production, or **roll back** to the previous production version if the new one underperforms.
4. Document this rollback process in a short markdown file — this doesn't require actually building an automated rollback system, just a clear, correct, written procedure.

```python
import mlflow
client = mlflow.tracking.MlflowClient()

client.transition_model_version_stage(
    name="purchase-intent-model",
    version=2,
    stage="Production"
)
```

✅ **Done when:** You have at least two versions of your model in the MLflow Registry, one marked Production, and a short written rollback procedure.

---

## 6. Actually Deploy to AWS

**What it is, in plain terms:** Week 4 asked for a documented deployment *plan*. This stretch goal is doing that plan for real — pushing the Docker images to AWS and getting a live, publicly reachable URL.

**Why it's the highest-impact but highest-risk item here:** A working live URL in the README is genuinely impressive and immediately verifiable by anyone reviewing the project. But AWS setup (billing, IAM permissions, networking) is also the most likely thing here to eat unexpected hours on a first attempt — budget real time for this, and don't start it if the deadline is close.

**A reasonably simple path (ECR + a single EC2 instance):**
1. Create an AWS account if she doesn't have one, and set up billing alerts immediately (easy to forget, easy to get an unexpected bill).
2. Push both Docker images (API and dashboard) to **ECR** (Elastic Container Registry) — this is basically Docker Hub, but AWS's own version.
3. Launch a small **EC2** instance, install Docker on it, and run `docker compose up` there, pulling the images from ECR.
4. Open the right ports (8000 for the API, 8501 for the dashboard) in the instance's security group settings.
5. Test the live URL from a different device or network, to confirm it's genuinely publicly reachable, not just working on her own network.
6. Add the live URL to the README, along with a note on how to stop/restart the instance (to control cost).

**A note on cost control:** Stop or terminate the EC2 instance when not actively demoing it, and set a billing alert at a low threshold (e.g. $5) so nothing unexpected happens.

✅ **Done when:** A live URL in the README, tested from a separate device/network, actually returns real predictions from the dashboard.

---

## 7. Simple Data-Drift Check

**What it is, in plain terms:** Over time, real-world sessions might start looking statistically different from the sessions the model was trained on — new browsers becoming common, seasonal shopping patterns shifting, etc. A drift check flags when incoming data starts looking meaningfully different from the training data, which is often an early warning sign that the model needs retraining.

**Why it's worth doing:** This is a genuinely practical, real-world ML concern that most student projects never touch — it demonstrates thinking beyond "train once and forget about it."

**How to do it, a simple approach:**
1. For each numeric feature, compare its distribution in the training data against its distribution in a batch of new/incoming data, using a statistical test like the **Kolmogorov-Smirnov (KS) test** (`scipy.stats.ks_2samp`) — it checks whether two samples plausibly come from the same distribution.
2. If the test flags a meaningful difference (a low p-value) for several features, that batch is "drifting" from training data.
3. Add a small function that runs this check and logs a warning if drift is detected — this can be as simple as printing a warning, no fancy alerting system required for a student project.
4. Optionally, show a simple drift-check status on the dashboard (e.g. a green/red indicator: "Data looks consistent with training data" vs. "Some features have drifted").

```python
from scipy.stats import ks_2samp

def check_drift(train_col, new_col, threshold=0.05):
    stat, p_value = ks_2samp(train_col, new_col)
    return p_value < threshold  # True means drift detected
```

✅ **Done when:** You have a working drift-check function, tested on at least one deliberately shifted sample of data to confirm it correctly flags drift when it should.

---

## Stretch Goals Checklist — Quick Reference

- [ ] Profit-curve optimizer built, with a business-justified optimal threshold
- [ ] Calibration checked with a reliability curve, before/after comparison done
- [ ] PageValues ablation done, framed as a deployment-leakage analysis
- [ ] Target/CatBoost encoding compared against one-hot for ID columns
- [ ] Model versioning set up in MLflow Registry, with a documented rollback procedure
- [ ] Real AWS deployment done, live URL tested and added to README
- [ ] Simple data-drift check built and tested

---

## A Few Words of Encouragement

These are genuinely the kind of additions that make a project stand out in an interview — not because they're complicated, but because most people don't get around to them. Picking even two or three of these, done well and clearly written up, will say more about her than rushing through all seven superficially.

There's no shame in stopping after the core 4 weeks, either — that's already a complete, strong project. Treat this list as "if there's time and curiosity left," not as a hidden fifth requirement.

Good luck with whichever ones you pick! 🎯
