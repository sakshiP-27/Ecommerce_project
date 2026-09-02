# Goal 5 MLflow Model Registry — promote & rollback

## What’s set up

Registered model name: **`purchase-intent-model`**

Tracking DB: `sqlite:///mlflow.db` (project root)

| Stage | Meaning |
|---|---|
| **Production** | Current “live” model version (Week-2-style CatBoost: 100 trees, depth 4, lr 0.05) |
| **Staging** | Candidate version (small tweak: 120 trees) waiting for review |
| **Archived** | Previous Production versions kept for rollback |

## One-time setup

From the project root:

```bash
pip install mlflow
python -m src.models.mlflow_registry
```

That trains/logs two CatBoost variants, registers them, and sets stages to **Production** + **Staging**.

Check:

```bash
python -m src.models.mlflow_rollback status
```

Optional UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://127.0.0.1:5000 → Models → `purchase-intent-model`.

## Promote Staging → Production

When Staging looks good:

```bash
python -m src.models.mlflow_rollback promote
```

What this does:
1. Moves current **Production** → **Archived**
2. Moves **Staging** → **Production**

## Rollback (if the new Production is worse)

```bash
python -m src.models.mlflow_rollback rollback
```

What this does:
1. Moves current **Production** → **Archived**
2. Restores the latest **Archived** version (previous Production) back to **Production**

## Manual equivalent (MLflow client)

```python
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

# Example: put version 2 into Production
client.transition_model_version_stage(
    name="purchase-intent-model",
    version=2,
    stage="Production",
    archive_existing_versions=True,
)
```

## Notes

- This stretch goal versions models in the **MLflow Registry**. The FastAPI/dashboard still load `src/models/final_model.pkl` swapping that pickle to match Registry Production would be a separate deployment step.
- Newer MLflow builds may warn that stages are legacy (aliases are the newer API). Stages are what the assignment asks for and still work for this project.
