# Install guide

This is for someone cloning the repo for the first time.

## Option A — Docker (recommended)

### What you need
- Docker Desktop installed and actually running (whale icon in the system tray)
- Git

### Steps
1. Clone the repo and go into the folder:
   ```bash
   git clone <your-repo-url>
   cd Ecommerce_project
   ```
2. Copy the env example if you want (optional for the current app):
   ```bash
   copy .env.example .env
   ```
   (On Mac/Linux: `cp .env.example .env`)
3. Build and start:
   ```bash
   docker compose up --build
   ```
4. Wait until you see Streamlit say it’s running, then open:
   - http://localhost:8501
   - http://localhost:8000/docs

### Common issues
- **`requirements.txt` / `dataset` not found during build**  
  Make sure you’re in the project root (where `docker-compose.yml` is).
- **sklearn pickle / `_RemainderColsList` error**  
  Needs `scikit-learn==1.5.2` (already pinned in `requirements.txt`). Rebuild with `--no-cache` if an old image stuck around.
- **Can’t reach `registry-1.docker.io`**  
  Network/DNS/VPN problem with Docker Hub. Restart Docker Desktop, or run Option B locally.

---

## Option B — Local Python (no Docker)

### What you need
- Python 3.11-ish (I used Anaconda)
- The repo contents including:
  - `dataset/ecommerce_sessions.csv`
  - `src/models/final_model.pkl`
  - `src/models/preprocessing_pipeline.pkl`

### Steps
1. Open a terminal in the project root.
2. Create/activate an environment (example with conda):
   ```bash
   conda create -n ecommerce python=3.11 -y
   conda activate ecommerce
   ```
3. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the API:
   ```bash
   uvicorn src.api.main:app --reload
   ```
5. In a **second** terminal (same env + same folder):
   ```bash
   streamlit run src/dashboard/Home.py
   ```

### Check it worked
- http://localhost:8000/health should return `{"status":"ok"}`
- http://localhost:8501 should show the Overview page with session counts

---

## Running tests

From project root:

```bash
pip install -r requirements.txt
pip install pytest httpx flake8
pytest tests/ -q
```

---

## If something still breaks

1. Confirm you’re in the repo root (you should see `src/`, `docker-compose.yml`, `requirements.txt`).
2. Confirm the two pickle files exist under `src/models/`.
3. Don’t retrain inside the dashboard — it only reads saved artifacts.
