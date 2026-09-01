# AWS Deployment Plan

This project is **ready for AWS**, but does **not** need to be deployed live for the assignment. Below is the plan we would follow to run the API and Streamlit dashboard in the cloud.

---

## Target services

| Service | Role |
|--------|------|
| **ECR** (Elastic Container Registry) | Store Docker images for the API and dashboard |
| **ECS Fargate** (preferred) *or* **EC2** | Run the containers |
| **Application Load Balancer** (optional) | Public HTTPS access to API (`:8000`) and dashboard (`:8501`) |
| **Secrets Manager / SSM Parameter Store** (optional) | Store production secrets instead of hard-coding them |

We already have local Docker recipes:
- `docker/Dockerfile.backend` → FastAPI (`src.api.main:app`)
- `docker/Dockerfile.frontend` → Streamlit (`src/dashboard/Home.py`)
- `docker-compose.yml` → runs both together locally

---

## High-level steps

1. **Build images locally (or in CI)**  
   ```bash
   docker compose build
   ```
   This produces the `api` and `frontend` images.

2. **Create two ECR repositories**  
   Example names: `ecommerce-api` and `ecommerce-dashboard`.

3. **Authenticate Docker to ECR and push**  
   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker tag ecommerce_project-api:latest <account>.dkr.ecr.<region>.amazonaws.com/ecommerce-api:latest
   docker tag ecommerce_project-frontend:latest <account>.dkr.ecr.<region>.amazonaws.com/ecommerce-dashboard:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/ecommerce-api:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/ecommerce-dashboard:latest
   ```

4. **Run the containers**
   - **Option A — ECS Fargate:** create a task definition with both containers (or two services), map ports `8000` and `8501`, set CPU/memory, attach to a VPC with public subnets (or private + load balancer).
   - **Option B — EC2:** launch an instance, install Docker, pull both images from ECR, run them with the same ports (similar to `docker compose up`).

5. **Configure networking**  
   Open/security-group allow:
   - `8000` → API (`/docs`, `/predict`, `/health`)
   - `8501` → Streamlit dashboard  
   Prefer HTTPS via a load balancer + certificate in production.

6. **Point the dashboard at the API (if needed later)**  
   Today the dashboard scores sessions using local model files inside the container. If we switch Live Scoring to call the API over HTTP, set an env var such as `API_URL=https://api.example.com`.

---

## Environment variables / config for production

Create a `.env` from `.env.example` (never commit real secrets).

| Variable | Purpose | Example |
|----------|---------|---------|
| `PYTHONPATH` | Ensure `src` imports resolve | `/app` |
| `AWS_DEFAULT_REGION` | Region for ECR/ECS | `eu-west-2` |
| `API_URL` | (Optional) public API base URL for dashboard | `https://api.example.com` |
| `DATA_PATH` | (Optional override) path to sessions CSV | `dataset/ecommerce_sessions.csv` |
| `MODEL_DIR` | (Optional override) folder with `.pkl` artifacts | `src/models` |
| `THRESHOLD` | Purchase decision threshold | `0.5` |

**Artifacts baked into the images today**
- Model: `src/models/final_model.pkl`
- Pipeline: `src/models/preprocessing_pipeline.pkl`
- Dataset (dashboard): `dataset/ecommerce_sessions.csv`
- Reports (explainability page): `reports/`

No database password is required for the current app. If MLflow or a DB were added later, those credentials would go in Secrets Manager, not in the image.

---

## Recommended production shape (simple)

```text
Internet
   │
   ▼
Load Balancer
   ├─ /api/*     → ECS service: ecommerce-api       (port 8000)
   └─ /dash/*    → ECS service: ecommerce-dashboard (port 8501)
         │
         ▼
       ECR (images)
```

---

## What we are *not* doing for the coursework

- No live 24/7 AWS deployment is required.
- This document proves the project is **AWS-ready**: containers exist, services are chosen, and env/config needs are listed.

To validate locally before any cloud push:

```bash
docker compose up --build
```

Then open:
- API docs: http://localhost:8000/docs  
- Dashboard: http://localhost:8501  
