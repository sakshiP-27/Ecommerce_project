# Week 4 Guide: Purchase-Intent Prediction Project
### A beginner-friendly walkthrough — Dashboard, Deployment & Docs

---

## Before You Start

**Goal for this week:** Turn everything built over the last three weeks into something a real person could open, click through, and understand — a live dashboard, a containerized app that runs anywhere, an automated check that catches mistakes, and documentation that lets a stranger set it all up. This is the "make it presentable and shippable" week.

**Why this matters:** A brilliant model that only she can run on her own laptop isn't a finished project. This week proves the whole thing actually works end-to-end, for someone who wasn't there while it was built — which is exactly what a grader (or a future employer) is checking for.

**A heads-up before you start:** This is usually the week solo projects run out of time, mostly because of Docker surprises late in the process. Don't leave Docker to the last day — start it by the middle of the week even if the dashboard isn't fully polished yet.

**What you need before starting:**
- Week 3's saved model, pipeline, explainer, and working FastAPI service
- `pip install streamlit plotly`
- Docker Desktop installed and running

---

## Step 1: Plan the Dashboard's Six Sections First

**What this means:** Before writing any Streamlit code, sketch out (on paper or in a notes file) what each of the six sections will show. Five minutes of planning here saves a lot of restructuring later.

**The six sections, and what belongs in each:**
1. **Overview** — headline numbers: total sessions, overall conversion rate, dataset size. The "at a glance" page.
2. **Live Session Scoring** — a form where someone can enter session details by hand (or pick a sample row) and get a live prediction + explanation back. This is the most interactive, most impressive section — the one to spend the most polish on.
3. **Funnel / Cohort Analytics** — charts from your Week 1 EDA, rebuilt as interactive Plotly charts: conversion by month, by visitor type, by traffic source, etc.
4. **Model Comparison** — the six-model metrics table from Week 2, shown visually (e.g. a bar chart comparing PR-AUC across models).
5. **Explainability** — the SHAP summary plot and a couple of example waterfall plots from Week 3, with the plain-English explanations you already wrote.
6. **Performance Metrics** — a closer look at your final chosen model: confusion matrix, precision-recall curve, and the threshold you picked, with your written justification for it.

**Why plan this first:** Most of the actual content for sections 3–6 already exists from Weeks 1–3 — this week is largely about *presenting* work you've already done, not creating new analysis from scratch.

✅ **Done when:** You have a one-line description of what goes in each of the six sections, written down before opening the code editor.

---

## Step 2: Set Up the Streamlit App Structure

**What this means:** Streamlit turns a Python script into a web app automatically — you write normal-looking Python, and it renders as a page in the browser. Multi-page apps use a specific folder structure Streamlit recognizes automatically.

**How to do it:**
```
dashboard/
├── Home.py                    ← the Overview page (Section 1)
└── pages/
    ├── 1_Live_Scoring.py
    ├── 2_Funnel_Analytics.py
    ├── 3_Model_Comparison.py
    ├── 4_Explainability.py
    └── 5_Performance_Metrics.py
```

Streamlit automatically turns every file in `pages/` into a separate page with its own entry in the sidebar — you don't need to write any extra routing code yourself. The number prefixes control the order they appear in the sidebar.

**Run it locally with:**
```
streamlit run dashboard/Home.py
```

✅ **Done when:** Running the command above opens a browser tab showing a sidebar with all six page names, even if the pages are still empty.

---

## Step 3: Load Data and Model Efficiently — Use Caching

**What this means:** Streamlit re-runs your *entire* script from top to bottom every single time someone clicks anything on the page. Without caching, that means reloading your CSV and model from disk on every single click — which would make the dashboard painfully slow.

**The two caching decorators to know:**
- `@st.cache_data` — for data that doesn't change, like a loaded DataFrame. Use this for your dataset.
- `@st.cache_resource` — for things that should be created once and reused, like a loaded model, pipeline, or SHAP explainer. Use this for anything from your `models/` folder.

**How to use them:**

```python
import streamlit as st
import joblib
import pandas as pd

@st.cache_data
def load_data():
    return pd.read_csv("dataset/ecommerce_sessions.csv")

@st.cache_resource
def load_model():
    return joblib.load("models/final_model.pkl")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/preprocessing_pipeline.pkl")
```

**One important rule for this whole dashboard:** It should only ever **read** the saved artifacts from Weeks 1–3 — never retrain a model or refit the pipeline inside the dashboard itself. Retraining belongs in your Week 2 training scripts, not here. The dashboard's job is to *present* results, not produce new ones.

✅ **Done when:** Data and model loading both use the correct caching decorator, and clicking around the dashboard feels fast, not sluggish.

---

## Step 4: Build the Live Session Scoring Page

**What this means:** This is the centerpiece page — a form where someone enters (or picks) session details, and instantly sees the model's prediction and explanation. It's the closest thing to a "demo" of the whole project in one page.

**How to do it:**
1. Use Streamlit input widgets (`st.number_input`, `st.selectbox`, `st.checkbox`) to build a form matching your model's expected input fields — the same fields as your Pydantic model from Week 3.
2. Consider adding a "load a random real session" button too, so the person testing it doesn't have to think up realistic numbers themselves.
3. On submit, either call your FastAPI `/predict` endpoint directly (if it's running), or reuse your prediction + SHAP code directly inside the dashboard.
4. Display: the prediction (Purchase / No Purchase), the probability as a clear visual (e.g. `st.progress` or a gauge chart), and a small SHAP waterfall plot showing why.

**Why this page matters most:** If a grader only clicks on one page before deciding whether the project "feels real," it's this one. It's worth the extra polish.

✅ **Done when:** You can fill in the form, hit submit, and see a prediction with an explanation appear on screen within a couple of seconds.

---

## Step 5: Build the Remaining Pages Using Interactive Charts

**What this means:** For the Funnel Analytics, Model Comparison, and Performance Metrics pages, reuse the analysis from Weeks 1–2, but rebuild the charts in **Plotly** instead of static matplotlib images — Plotly charts let the viewer hover, zoom, and filter, which makes the dashboard feel far more like a real product.

**A simple conversion example (matplotlib → Plotly):**

```python
import plotly.express as px

fig = px.bar(conversion_by_month, x="Month", y="ConversionRate",
             title="Conversion Rate by Month")
st.plotly_chart(fig, use_container_width=True)
```

**For the Explainability page:** SHAP plots are usually matplotlib-based by default, and that's fine to keep as-is (`st.pyplot(fig)`) — you don't need to convert everything to Plotly, just the charts where interactivity genuinely adds value (bar charts, trends, comparisons).

✅ **Done when:** All six pages show real content pulled from your saved artifacts and earlier notebooks, with at least the funnel/comparison charts rendered as interactive Plotly charts.

---

## Step 6: Write the Dockerfiles

**What this means:** A Dockerfile is a recipe that describes how to build a self-contained, portable version of your app — one that will run identically on any machine, not just hers. You'll need two: one for the API, one for the dashboard.

**A simple Dockerfile for the FastAPI service** (`docker/Dockerfile.api`):

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**A simple Dockerfile for the Streamlit dashboard** (`docker/Dockerfile.dashboard`):

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY dataset/ ./dataset/

CMD ["streamlit", "run", "dashboard/Home.py", "--server.address=0.0.0.0"]
```

**A known gotcha to watch for:** `LightGBM` sometimes needs a system library called `libgomp1` inside the Docker image, which isn't there by default on the slim Python image. If the build fails with an error mentioning `libgomp`, add this line before the `pip install` step:
```dockerfile
RUN apt-get update && apt-get install -y libgomp1
```
`CatBoost`'s installed package is also fairly large, so don't be alarmed if that specific build step takes a few extra minutes — that's normal, not a sign something's broken.

✅ **Done when:** Both `docker build` commands complete successfully, with no errors.

---

## Step 7: Bring It All Up Together with Docker Compose

**What this means:** Instead of manually starting the API container and the dashboard container separately every time, one `docker-compose.yml` file starts both together with one command.

```yaml
version: "3.8"
services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"

  dashboard:
    build:
      context: .
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
```

**Run everything with:**
```
docker compose up --build
```

Then check `http://localhost:8000/docs` (the API) and `http://localhost:8501` (the dashboard) both come up correctly.

✅ **Done when:** One single command brings up both the API and the dashboard, and both are reachable in the browser.

---

## Step 8: Set Up GitHub Actions (Automated Checks on Every Push)

**What this means:** A small config file that tells GitHub to automatically run checks — code style, tests, and a Docker build — every time code is pushed, without her having to remember to run them manually.

**How to do it:** Create `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pip install flake8 pytest
      - run: flake8 src/ --max-line-length=100
      - run: pytest tests/
```

**Why this matters, beyond looking impressive:** It's a genuine safety net. If a future change accidentally breaks something, this catches it automatically on GitHub, before it becomes a bigger problem — exactly the pytest tests from Week 3 finally paying off here.

✅ **Done when:** After pushing to GitHub, the "Actions" tab on the repository shows the workflow running, and passing.

---

## Step 9: Add AWS-Ready Deployment Notes (No Live Deployment Required)

**What this means:** The PRD asks for the project to be *ready* for AWS deployment — not for it to actually be deployed and running live 24/7. This step is about documenting the deployment plan clearly, which is what will actually get evaluated.

**What to write, in a `docs/deployment.md` file:**
1. Which AWS services would be used — typically **ECR** (Elastic Container Registry, to store the Docker images) and either **ECS** or a plain **EC2** instance (to run them).
2. The rough steps: build images → push to ECR → run on ECS/EC2 → configure environment variables for things like the model path.
3. Any environment variables or secrets the app would need in production (referencing your `.env.example` from Week 1).

✅ **Done when:** You have a short, clear deployment plan written down — even without ever actually deploying it live.

---

## Step 10: Write the Full Documentation Set

**What this means:** The written materials that let a complete stranger understand, install, and use the project without her explaining it in person.

**What to include:**
1. **README.md** — project overview, quick-start instructions (`docker compose up --build` should be enough for someone to get it running), and a short description of each of the four weeks' work.
2. **Architecture diagram** — a simple visual showing how the pieces connect: data → preprocessing → model → API → dashboard. A simple boxes-and-arrows diagram is enough; it doesn't need to be fancy.
3. **API docs** — FastAPI mostly generates this automatically at `/docs`, but add a short written note in the README pointing people to it.
4. **Install guide** — step-by-step setup instructions, written as if for someone who has never seen the project before.
5. **User guide** — how to use the dashboard, section by section.
6. **Model documentation** — this is your Week 3 model card, included here rather than rewritten.

✅ **Done when:** All six documentation pieces exist, and the README alone would let a stranger get the project running without asking a single question.

---

## Step 11: Record the Demo Video

**What this means:** A short screen recording (3–5 minutes) walking through the project, meant to be watchable by someone who has never seen it before and won't ask follow-up questions.

**A solid structure to follow:**
1. **0:00–0:30** — one-sentence problem statement: what the project predicts, and why it matters.
2. **0:30–1:30** — a live prediction on the dashboard's Live Scoring page, showing a real example.
3. **1:30–2:30** — the explainability section: why the model made that prediction (SHAP waterfall).
4. **2:30–3:30** — a quick tour of the analytics/model comparison pages.
5. **3:30–4:30** — mention the engineering behind it: Docker, GitHub Actions, the API — briefly, no need to show every line of code.

**Practical tip:** Do a rough practice run first without recording, so the actual recording doesn't have long pauses while figuring out what to click next.

✅ **Done when:** You have one clean video file, under 5 minutes, covering all five points above.

---

## Step 12: Do a Fresh-Clone Test Before Submitting

**What this means:** Clone the repository into a completely new, empty folder (as if you were a stranger downloading it for the first time), and follow only the README instructions — nothing from memory.

**Why this is the single most valuable last step:** "Works on my machine" is the most common way capstone projects lose easy points. Environment variables that were only ever set manually, a config file that only exists locally, a path that's hard-coded to her exact folder name — all of these hide perfectly until someone else (or a fresh clone) tries to run it.

**How to do it:**
1. `git clone` the repo into a brand-new folder, somewhere else on the machine.
2. Follow the README, step by step, exactly as written — don't skip ahead using knowledge from having built it.
3. Note down anywhere the instructions were unclear, incomplete, or assumed something that wasn't actually there.
4. Fix those gaps before submitting.

✅ **Done when:** A fresh clone, following only the README, gets the full app running with no undocumented extra steps.

---

## Week 4 Checklist — Quick Reference

- [ ] Six dashboard sections planned out before coding
- [ ] Streamlit multi-page structure set up (`Home.py` + `pages/`)
- [ ] Data and model loading both cached correctly (`@st.cache_data` / `@st.cache_resource`)
- [ ] Dashboard only reads saved artifacts — never retrains inside itself
- [ ] Live Session Scoring page built and working end-to-end
- [ ] Funnel/comparison charts rebuilt as interactive Plotly charts
- [ ] Dockerfiles written for both API and dashboard, both build successfully
- [ ] `docker-compose.yml` brings up both services together with one command
- [ ] GitHub Actions workflow running lint + tests + build on every push
- [ ] AWS deployment plan documented (no live deployment required)
- [ ] Full documentation set written: README, architecture diagram, install guide, user guide, model card
- [ ] Demo video recorded (3–5 minutes)
- [ ] Fresh-clone test done, and any gaps in the README fixed

---

## A Few Words of Encouragement

This is the finish line, and also historically the week where things feel like they're falling apart right when the deadline is closest — usually because of a Docker build error nobody saw coming, or a README that made perfect sense to her but confuses a fresh reader. That's completely normal, and it's exactly why Step 12 (the fresh-clone test) exists — it's meant to catch precisely that.

If time gets genuinely tight, it's better to have a working Docker Compose setup with a slightly rougher dashboard than a beautiful dashboard that only runs on her laptop — the engineering rigor is weighted heavily in this project on purpose.

Four weeks of solid, well-documented work is a genuinely strong project to walk into an interview with. Good luck wrapping it up! 🎯
