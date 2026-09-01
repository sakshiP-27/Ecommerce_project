# Ecommerce Purchase-Intent Prediction

Capstone project: predict whether a website session is likely to convert into a purchase, explain *why*, and wrap it in an API + Streamlit dashboard that runs with Docker.

I built this over four weeks — data/EDA → modelling → explainability/API → dashboard/deployment docs.

---

## Quick start (easiest way)

You mainly need **Docker Desktop** running.

```bash
git clone <your-repo-url>
cd Ecommerce_project
docker compose up --build
```

Then open:

- Dashboard: http://localhost:8501  
- API docs (interactive): http://localhost:8000/docs  
- Health check: http://localhost:8000/health  

To stop everything: `Ctrl+C`, then `docker compose down`.

> If `docker compose up --build` fails with a registry/DNS error, that’s usually network/VPN related — restart Docker Desktop and try again. You can also run the API/dashboard locally (see [docs/install_guide.md](docs/install_guide.md)).

---

## What this project does

Given session features (pages viewed, durations, bounce/exit rates, `PageValues`, month, visitor type, etc.), the model outputs:

- **Purchase** or **No Purchase**
- purchase **probability**
- **top SHAP features** explaining the score

Final model: **CatBoost** (`src/models/final_model.pkl`)  
Preprocessing: saved pipeline (`src/models/preprocessing_pipeline.pkl`)  
Decision threshold: **0.5** (in `src/config/config.yaml`)

---

## Project layout (short version)

```text
dataset/ecommerce_sessions.csv   # main data
src/
  api/main.py                    # FastAPI
  dashboard/                     # Streamlit multi-page app
  models/                        # final_model.pkl + pipeline
  features/ / preprocessing/     # feature engineering + encoding
docs/                            # install, user guide, architecture, AWS notes
reports/                         # SHAP plots + model card
tests/                           # pytest
docker/ + docker-compose.yml     # containers
.github/workflows/ci.yml         # GitHub Actions
```

---

## Week-by-week summary

| Week | What I did |
|------|------------|
| **1** | Loaded the sessions data, EDA, decided to keep `PageValues` (with a leakage caveat), built preprocessing + engineered features |
| **2** | Compared several models, focused on **PR-AUC / recall**, tuned & saved CatBoost + pipeline |
| **3** | SHAP (+ LIME checks), model card, FastAPI `/health` + `/predict`, unknown-category handling, pytest |
| **4** | Streamlit dashboard, Docker Compose, GitHub Actions CI, AWS deployment notes, docs |

More detail:

- Install: [docs/install_guide.md](docs/install_guide.md)  
- How to use the dashboard: [docs/user_guide.md](docs/user_guide.md)  
- Architecture: [docs/architecture.md](docs/architecture.md)  
- API notes: [docs/api.md](docs/api.md)  
- AWS plan: [docs/deployment.md](docs/deployment.md)  
- Model card: [docs/model_card.md](docs/model_card.md) (same as `reports/model_card.md`)

---

## API (quick note)

FastAPI auto-generates Swagger docs at **http://localhost:8000/docs**.

Useful endpoints:

- `GET /health` → `{"status":"ok"}`
- `POST /predict` → prediction, probability, threshold, top SHAP features

There’s a sample JSON body in [docs/api.md](docs/api.md).

---

## Local run (without Docker)

From the project root, with your conda/venv active:

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
# new terminal
streamlit run src/dashboard/Home.py
```

---

## Tests / CI

```bash
pytest tests/ -q
```

On every push/PR, GitHub Actions runs lint + tests + Docker build (`.github/workflows/ci.yml`).

---

## Notes / known quirks

- Model artifacts must stay on **scikit-learn 1.5.2** (pinned in `requirements.txt`) or the pipeline pickle won’t load.
- Dashboard lives under `src/dashboard/` (not a top-level `dashboard/` folder).
- `PageValues` is a strong signal; I kept it for this project but called out the early-session leakage risk in the model card.
