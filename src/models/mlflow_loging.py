import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import mlflow.xgboost
import mlflow.catboost
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score)

from src.data.loader import load_data
from src.features.engineering import add_engineered_features
from src.config.settings import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

def prepare_data():
    df = load_data()
    df = add_engineered_features(df)
    df = pd.get_dummies(df, columns=config["categorical_columns"])

    X = df.drop(config["target_column"], axis=1)
    y = df[config["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_seed"]
    )
    return X_train, X_test, y_train, y_test

def log_all_models():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    X_train, X_test, y_train, y_test = prepare_data()

    mlflow.set_experiment("purchase-intent-prediction")

    models = {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=5000, class_weight="balanced"),
            "params": {"max_iter": 5000, "class_weight": "balanced"}
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(class_weight="balanced"),
            "params": {"class_weight": "balanced"}
        },
        "RandomForest": {
            "model": RandomForestClassifier(class_weight="balanced", n_estimators=100),
            "params": {"n_estimators": 100, "class_weight": "balanced"}
        },
        "XGBoost": {
            "model": XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric="logloss"),
            "params": {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.05}
        },
        "CatBoost_tuned": {
            "model": CatBoostClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, verbose=0),
            "params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}
        },
    }

    for name, config_item in models.items():
        with mlflow.start_run(run_name=name):
            model = config_item["model"]
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            # Log parameters
            mlflow.log_params(config_item["params"])

            # Log metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_prob),
                "pr_auc": average_precision_score(y_test, y_prob),
            }
            mlflow.log_metrics(metrics)

            # Log model
            if "XGB" in name:
                mlflow.xgboost.log_model(model, "model")
            elif "CatBoost" in name:
                mlflow.catboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            logger.info(f"{name} logged — PR-AUC: {metrics['pr_auc']:.4f}")

    # Register best model (CatBoost)
    logger.info("All models logged to MLflow. Run 'mlflow ui' to view.")

if __name__ == "__main__":
    log_all_models()