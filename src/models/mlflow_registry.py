"""
Stretch goal #5 — set up MLflow Model Registry with Production + Staging.

Usage (from project root):
    python -m src.models.mlflow_registry
"""

from __future__ import annotations

import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.config.settings import load_config
from src.data.loader import load_data
from src.features.engineering import add_engineered_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "purchase-intent-model"
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "purchase-intent-registry"


def _prepare_xy():
    config = load_config()
    df = add_engineered_features(load_data())
    X = df.drop(columns=[config["target_column"]])
    y = df[config["target_column"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["test_size"],
        random_state=config["random_seed"],
    )
    categorical_columns = config["categorical_columns"]
    numeric_columns = [
        col
        for col in X_train.select_dtypes(include="number").columns
        if col not in categorical_columns
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", RobustScaler(), numeric_columns),
        ]
    )
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    return X_train_t, X_test_t, y_train, y_test


def _train_and_register(
    *,
    run_name: str,
    params: dict,
    stage: str,
    client: MlflowClient,
) -> str:
    X_train_t, X_test_t, y_train, y_test = _prepare_xy()
    model = CatBoostClassifier(**params, verbose=0)
    model.fit(X_train_t, y_train)
    y_prob = model.predict_proba(X_test_t)[:, 1]
    pr_auc = float(average_precision_score(y_test, y_prob))

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.catboost.log_model(model, name="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, MODEL_NAME)
        version = str(mv.version)

    # Guide uses named stages (Production / Staging / Archived)
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=(stage == "Production"),
    )
    logger.info(
        "Registered %s v%s -> stage=%s (PR-AUC=%.4f)",
        MODEL_NAME,
        version,
        stage,
        pr_auc,
    )
    return version


def setup_registry() -> None:
    """
    Create two registry versions:
      v1 (or next): Week-2-style CatBoost → Production
      next: small hyperparameter tweak → Staging
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Ensure the registered model name exists (created on first register_model)
    production_params = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.05,
    }
    staging_params = {
        "n_estimators": 120,  # small tweak for a new version
        "max_depth": 4,
        "learning_rate": 0.05,
    }

    prod_version = _train_and_register(
        run_name="catboost_production_candidate",
        params=production_params,
        stage="Production",
        client=client,
    )
    staging_version = _train_and_register(
        run_name="catboost_staging_candidate",
        params=staging_params,
        stage="Staging",
        client=client,
    )

    print(f"Registry ready: {MODEL_NAME}")
    print(f"  Production = v{prod_version}")
    print(f"  Staging    = v{staging_version}")
    print("Tracking URI:", TRACKING_URI)
    print("View UI: mlflow ui --backend-store-uri sqlite:///mlflow.db")


def print_status() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print(f"No versions found for '{MODEL_NAME}'. Run setup first.")
        return
    print(f"Model: {MODEL_NAME}")
    for mv in sorted(versions, key=lambda v: int(v.version)):
        print(f"  v{mv.version}: stage={mv.current_stage}  run_id={mv.run_id}")


if __name__ == "__main__":
    setup_registry()
    print_status()
