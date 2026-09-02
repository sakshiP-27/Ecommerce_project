# Demo video script (Step 11)

Rough talking script for a 3–5 minute screen recording.
Practice once while clicking before you record.

**Before recording**
- Dashboard open: http://localhost:8501
- Optional: http://localhost:8000/docs in another tab
- Know sidebar pages: Live Scoring, Explainability, Funnel, Model Comparison

---

## 0:00–0:30 — Problem

*[Show Overview page]*

Hi — this is my ecommerce purchase-intent project.

The goal is simple: given a website session — pages viewed, time on site, bounce rate, page values, and so on — predict whether that session is likely to convert into a purchase, and also explain why the model thinks that.

That matters because a shop can then focus offers or support on high-intent visitors instead of treating every session the same.

---

## 0:30–1:30 — Live prediction

*[Sidebar → Live Session Scoring]*

This is the main demo page — Live Session Scoring.

I’ll click Load a random real session so we’re using an actual row from the dataset…
…and now Score this session.

You can see the model’s output here: either Purchase or No Purchase, and the probability.
In this case it’s [say whatever appears], using a threshold of 0.5.

---

## 1:30–2:30 — Explainability (waterfall)

*[Stay on Live Scoring, point at SHAP waterfall / top features]*

Underneath the prediction is the SHAP explanation.

This waterfall starts from the model’s average baseline and then shows which features pushed the score up or down for this session.

Usually the biggest driver is PageValues — if it’s high, that strongly pushes toward purchase; if it’s zero, it pushes the other way. Things like product pages viewed, total pages, exit rate and bounce rate also show up a lot.

So it’s not just a black-box number — you can actually read the reason.

*[Optional: sidebar → Explainability]*

I’ve also got global SHAP plots and a short SHAP vs LIME comparison on the Explainability page from Week 3.

---

## 2:30–3:30 — Analytics / model comparison

*[Sidebar → Funnel Analytics]*

Quickly on analytics — these are interactive Plotly charts for conversion by month, visitor type, traffic type, and weekend vs weekday. That’s the EDA side, but as a dashboard.

*[Sidebar → Model Comparison]*

And here is the Week 2 model comparison. I cared mainly about PR-AUC, because purchases are the minority class and accuracy can look fine even if you’re missing buyers.

I compared models like logistic regression, trees, random forest, XGBoost and CatBoost. CatBoost is the one I saved and use in the API and dashboard — roughly 0.86 PR-AUC and about 76% recall at threshold 0.5.

*[Optional: Performance Metrics]*

Performance Metrics has the confusion matrix and precision–recall curve for that final model.

---

## 3:30–4:30 — Engineering wrap-up

*[API /docs tab, or speak over Overview]*

On the engineering side: the model is wrapped in a FastAPI service with /health and /predict — interactive docs are at localhost:8000/docs.

Everything also runs with Docker Compose — one command brings up the API and dashboard together.

And there’s GitHub Actions CI that runs lint, pytest, and a Docker build on every push.

So overall: predict purchase intent, explain it with SHAP, and ship it as an API plus dashboard that’s containerised and tested.

Thanks for watching.

---

## Tips

- If the random session is boring (very low PageValues), load again once before scoring.
- Don’t read word-for-word — glance at the section headings.
- Aim under 5 minutes; cut Funnel if you’re over time.
