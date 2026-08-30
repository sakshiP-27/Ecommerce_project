# Week 3 Guide: Purchase-Intent Prediction Project
### A beginner-friendly walkthrough — Explainability & API

---

## Before You Start

**Goal for this week:** Make your model explain *why* it predicts what it predicts, then wrap it in a working web API that anyone (or the dashboard, next week) can send a session to and get a prediction + explanation back.

**Why this matters:** A model that just outputs "0.73 probability of purchase" isn't very useful to a real business — nobody can act on a number they don't understand. Explainability turns "the model says so" into "the model says so *because this session had high page values and long product-browsing time*." This is also the "X" in XAI (eXplainable AI) — the core idea of this whole project, not an optional extra.

**What you need before starting:**
- Your final trained model from Week 2 (`models/final_model.pkl`)
- Your final fitted preprocessing pipeline (`models/preprocessing_pipeline.pkl`)
- `pip install shap lime fastapi uvicorn pydantic`

---

## Step 1: Understand SHAP Before Using It

**What it is, in plain terms:** SHAP (SHapley Additive exPlanations) answers the question *"how much did each feature push this specific prediction up or down?"* It comes from an idea in game theory about fairly splitting credit among team members — here, the "team" is your features, and the "result" is the prediction.

**Why it matters over just looking at feature importance:** Feature importance (from Week 2) tells you which features matter *on average, across the whole dataset*. SHAP can do that too, but it can *also* tell you, for one single session, exactly which features pushed that one prediction up or down, and by how much. That's the difference between "PageValues matters in general" and "PageValues added +0.15 to THIS session's purchase probability."

**Two things SHAP gives you:**
1. **Global explanation** — across all sessions, which features matter most overall (similar to Week 2's feature importance, but usually more reliable).
2. **Local explanation** — for one single session, exactly which features pushed the prediction up or down, and by how much.

✅ **Done when:** You can explain in one sentence, in your own words, the difference between global and local SHAP explanations.

---

## Step 2: Pick the Right SHAP Explainer for Your Model

**What this means:** SHAP has different "explainer" classes depending on what kind of model you're explaining. Using the right one matters a lot for speed.

**The one you'll almost certainly want:** `TreeExplainer`. It's built specifically for tree-based models (Random Forest, XGBoost, LightGBM, CatBoost — likely your best model from Week 2) and is dramatically faster than the generic option, because it's optimized to understand how trees work internally.

**Avoid `KernelExplainer` unless you have to.** It works on *any* model (even Logistic Regression), but it's much slower because it treats the model as a black box and has to run many extra calculations to approximate the explanation.

**How to do it:**

```python
import shap

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
```

**Important:** Build this explainer **once**, and reuse it. Don't recreate it every time you want to explain a new session — that's slow and unnecessary. You'll load it once at startup in your API later this week too.

✅ **Done when:** You have one working explainer object, and you can generate SHAP values for your test set without errors.

---

## Step 3: Create Global Explanation Visuals

**What this means:** A couple of standard SHAP charts that summarize, across your whole test set, which features matter and how.

**Two charts to create, both built into the SHAP library:**
1. **Summary plot** (`shap.summary_plot`) — shows every feature ranked by importance, and for each one, whether high or low values of that feature push predictions up or down. This is usually the single most useful SHAP chart.
2. **Bar plot** (`shap.summary_plot(..., plot_type="bar")`) — a simpler version showing just the ranked importance, without the up/down detail. Easier for a non-technical audience to read.

```python
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
```

**What to write alongside them:** One paragraph describing what the summary plot shows in plain English — e.g. *"High PageValues and longer ProductRelated_Duration consistently push predictions toward 'will purchase,' while high BounceRates pushes toward 'will not purchase.'"*

✅ **Done when:** You have both charts saved as images (for your report later), with a plain-English paragraph explaining what they show.

---

## Step 4: Create Local (Per-Session) Explanations

**What this means:** Pick a few individual sessions and explain exactly why the model predicted what it did for *that one session specifically*.

**How to do it:** Use a **waterfall plot** or **force plot** for one row at a time:

```python
shap.plots.waterfall(shap_values[0])   # explains just the first test session
```

**What to include in your notebook:** Pick 2–3 example sessions — ideally one clear "will buy," one clear "won't buy," and one the model was unsure about (probability near your threshold from Week 2). Explain each one in a sentence, e.g. *"For this session, high ProductRelated pages and low BounceRate pushed the prediction strongly toward 'will purchase.'"*

**Why this specific step matters for the project:** This is exactly the kind of output your API and dashboard will need to produce *live*, for any new session — so working through a few examples by hand now means you'll understand exactly what your API needs to return later this week.

✅ **Done when:** You can explain, in plain English, why the model made 2–3 specific individual predictions.

---

## Step 5: Add LIME as a Second, Independent Lens

**What it is, in plain terms:** LIME (Local Interpretable Model-agnostic Explanations) explains one prediction at a time too — but does it completely differently from SHAP. It works by slightly changing the input many times, watching how the prediction changes, and then fitting a very simple model to approximate what happened *locally, near that one prediction*.

**Why bother if you already have SHAP?** They're built on different ideas, so when they roughly agree, that's a good sign your explanation is trustworthy — not just an artifact of one particular method. Showing both, and briefly comparing them, is a stronger project than relying on just one.

**How to do it:**

```python
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Purchase", "Purchase"],
    mode="classification"
)

explanation = lime_explainer.explain_instance(X_test.iloc[0].values, final_model.predict_proba)
explanation.show_in_notebook()
```

**What to write:** For the same 2–3 sessions you explained with SHAP in Step 4, run LIME on them too, and write one or two sentences comparing the two: did they point to the same top features, or not?

✅ **Done when:** You've compared SHAP and LIME on the same few sessions, and written down whether they agree.

---

## Step 6: Write a Short Model Card

**What this means:** A short, plain-language summary document describing your model — not code, just a page of honest, clear writing.

**What to include (keep it to about one page):**
- **Intended use** — what this model is for (flagging likely-to-purchase sessions), and what it's *not* for (e.g. not a guarantee, not meant to replace human judgment on individual customers).
- **Performance summary** — your headline PR-AUC/recall numbers from Week 2, in plain English.
- **Key features** — the top 3–5 features driving predictions, from your SHAP summary plot.
- **Limitations** — be honest here. For example: the model was trained on one dataset and may not generalize to a different store or season; the `PageValues` decision you made in Week 1 should be restated here.

✅ **Done when:** You have a one-page model card that a non-technical manager could read and understand.

---

## Step 7: Build the FastAPI Service

**What this means:** Wrap your model, pipeline, and explainer in a small web application so that sending it session data over the internet (or `localhost`) returns a prediction and explanation.

**The two endpoints you need:**
1. `GET /health` — a simple "is this API alive?" check. Returns something like `{"status": "ok"}`. Used by monitoring tools (and Docker, later) to check the service hasn't crashed.
2. `POST /predict` — takes one session's data in, returns a prediction out.

**How to validate incoming data — use Pydantic:** FastAPI works together with a library called Pydantic to automatically check that incoming requests have the right fields and types, and reject bad requests with a clear error message *before* your code even runs. Define what a valid request looks like:

```python
from pydantic import BaseModel

class SessionInput(BaseModel):
    Administrative: int
    Administrative_Duration: float
    ProductRelated: int
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    Month: str
    VisitorType: str
    Weekend: bool
    # ... include the rest of your columns
```

**A minimal version of the API:**

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load everything ONCE at startup, not per-request
model = joblib.load("models/final_model.pkl")
pipeline = joblib.load("models/preprocessing_pipeline.pkl")
explainer = shap.TreeExplainer(model)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(session: SessionInput):
    df = pd.DataFrame([session.dict()])
    df_transformed = pipeline.transform(df)

    probability = model.predict_proba(df_transformed)[0][1]
    prediction = "Purchase" if probability >= 0.5 else "No Purchase"  # use your Week 2 threshold here

    shap_values = explainer.shap_values(df_transformed)
    # extract top contributing features from shap_values here

    return {
        "prediction": prediction,
        "probability": round(float(probability), 3),
        "top_features": []  # fill in from shap_values
    }
```

**Why "load once at startup" matters so much:** Loading a model from disk takes real time (sometimes a full second or more). If you did it inside the `/predict` function, every single request would reload the model from scratch — turning a fast API into a painfully slow one. Loading once when the app starts, and reusing it for every request, is the difference between a usable API and an unusable one.

✅ **Done when:** You can run the API locally (`uvicorn src.api.main:app --reload`), visit `http://localhost:8000/docs` (FastAPI gives you this automatically), and successfully send a test request that returns a prediction.

---

## Step 8: Handle Unknown Categories Gracefully

**What this means:** What happens if, in the future, a real session comes in with a `Month` value your training data never saw, or a new `Browser` code? By default, this would crash your API.

**How to fix it:** When you built your `OneHotEncoder` back in Week 1, you should have set `handle_unknown="ignore"`. If you did, unseen categories are safely treated as "none of the known categories" instead of throwing an error. If you didn't set this originally, go back and add it now, then re-save your pipeline.

**Why this matters for the project specifically:** This is exactly the kind of "will it survive contact with the real world?" detail that separates a notebook-only project from something that behaves like real production software.

✅ **Done when:** You've tested sending a deliberately unusual value (e.g. a `Month` your training data didn't include) to your API, and confirmed it doesn't crash.

---

## Step 9: Write Tests for the Data Loading and Prediction Logic

**What this means:** A few small automated checks (using `pytest`) that confirm your core code works correctly — and, importantly, will keep working correctly if she changes something later.

**A few tests worth writing:**
1. Test that your data loader raises an error on a missing file (from Week 1).
2. Test that your data loader raises an error if a required column is missing.
3. Test that your `/predict` endpoint returns a response with the expected fields (`prediction`, `probability`) for a valid, realistic input.
4. Test that your `/predict` endpoint doesn't crash on an edge-case input (e.g. the unusual category from Step 8).

**A minimal example:**

```python
def test_predict_returns_expected_fields(client, sample_session):
    response = client.post("/predict", json=sample_session)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
```

**Why bother, at this stage of the project?** Tests aren't about proving the code is perfect — they're a safety net. When she comes back next week to build the dashboard and inevitably tweaks something in the API, these tests will immediately tell her if she accidentally broke something, instead of finding out much later in a confusing way.

✅ **Done when:** Running `pytest` in the terminal shows all tests passing, with at least one test each for the data loader and the `/predict` endpoint.

---

## Week 3 Checklist — Quick Reference

- [ ] Understand the difference between global and local SHAP explanations
- [ ] Correct SHAP explainer chosen (`TreeExplainer` for tree-based models), built once and reused
- [ ] Global SHAP summary plots created, with a plain-English paragraph explaining them
- [ ] 2–3 individual sessions explained with SHAP waterfall/force plots
- [ ] Same sessions explained with LIME, and compared against SHAP
- [ ] One-page model card written (intended use, performance, key features, limitations)
- [ ] FastAPI service built with `/health` and `/predict` endpoints
- [ ] Pydantic model validates incoming request data
- [ ] Model, pipeline, and explainer all loaded once at startup, not per-request
- [ ] Encoder handles unknown categories without crashing
- [ ] Pytest tests written for data loader and `/predict`, all passing

---

## A Few Words of Encouragement

This is the week the project starts feeling like a "real" application instead of a notebook — that jump can feel intimidating, but it's mostly just wiring together things she already built in Weeks 1 and 2. If SHAP feels confusing at first, that's completely normal — even experienced ML engineers usually need to sit with a few waterfall plots before it clicks. The goal isn't to memorize the math behind it, just to be able to read a chart and explain what it's saying in plain words.

If the FastAPI docs page (`/docs`) feels like magic — it kind of is. It's auto-generated from the Pydantic model, and it's a genuinely great way to test the API by hand before building the dashboard around it next week.

Good luck with Week 3! 🎯
