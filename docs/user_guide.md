# Dashboard user guide

Run the app (`docker compose up` or `streamlit run src/dashboard/Home.py`), then use the **sidebar** to move between pages.

---

## 1. Overview (Home)

First page you land on.

Shows quick numbers from the dataset:
- total sessions
- conversion rate
- number of purchases
- decision threshold (0.5)

Also confirms the saved CatBoost model / pipeline loaded correctly.

---

## 2. Live Session Scoring (main demo page)

This is the one to show in a demo.

**What you can do**
1. Click **Load a random real session** (fills the form from the CSV), or type values yourself.
2. Click **Score this session**.
3. Read the result:
   - prediction (`Purchase` / `No Purchase`)
   - probability bar
   - SHAP waterfall (why the model said that)
   - top feature contributions table

Positive SHAP values push toward purchase; negative ones push away from it. In most examples, `PageValues` dominates.

---

## 3. Funnel / Cohort Analytics

Interactive Plotly charts for conversion rate by:
- Month
- Visitor type
- Traffic type
- Weekend vs weekday

Hover on bars to see session/purchase counts. This is basically Week 1 EDA, but clickable.

---

## 4. Model Comparison

Week 2 results table + charts.

Look at **PR-AUC** first (better headline metric than accuracy here because purchases are rarer). CatBoost is the model I saved and shipped, even though XGBoost was very close on PR-AUC.

---

## 5. Explainability

Shows the global SHAP plots I saved in Week 3 (`reports/shap_summary_plot.png`, `reports/shap_bar_plot.png`), plus a short SHAP vs LIME write-up for the three example sessions (clear buyer, clear non-buyer, uncertain).

For a single live explanation, use **Live Session Scoring** instead.

---

## 6. Performance Metrics

Closer look at the final CatBoost model on the same train/test split settings:
- PR-AUC / precision / recall / F1 at threshold 0.5
- confusion matrix
- precision–recall curve
- threshold trade-off table (0.3 → 0.6)

There’s also a short justification for sticking with 0.5 for this project.

---

## Tips

- If a page errors the first time, refresh once — model load can take a second.
- You don’t need the API running for Live Scoring right now; the dashboard loads the pickle files itself.
- Unknown months won’t crash scoring/API thanks to `handle_unknown="ignore"` in the encoder.
