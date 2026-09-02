"""PageValues ablation helpers (deployment-leakage analysis)."""

from __future__ import annotations

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def _build_preprocessor(X_train: pd.DataFrame, categorical_columns: list[str]):
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
    return preprocessor


def _train_catboost(X_train_processed, y_train):
    """Same final CatBoost config as Week 2 / save_model.py."""
    model = CatBoostClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        verbose=0,
    )
    model.fit(X_train_processed, y_train)
    return model


def evaluate_variant(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    categorical_columns: list[str],
    threshold: float = 0.5,
    label: str = "with_PageValues",
) -> dict:
    """Fit preprocessor + CatBoost on one feature set; return test metrics."""
    preprocessor = _build_preprocessor(X_train, categorical_columns)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model = _train_catboost(X_train_t, y_train)
    y_prob = model.predict_proba(X_test_t)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "variant": label,
        "n_features_raw": int(X_train.shape[1]),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "model": model,
        "preprocessor": preprocessor,
        "y_prob": y_prob,
    }


def run_pagevalues_ablation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    categorical_columns: list[str],
    threshold: float = 0.5,
    pagevalues_col: str = "PageValues",
) -> pd.DataFrame:
    """
    Train once with PageValues and once without; return a metrics table.

    Uses the same CatBoost hyperparameters and the same train/test rows.
    """
    with_pv = evaluate_variant(
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_columns,
        threshold=threshold,
        label="with_PageValues",
    )

    drop_cols = [pagevalues_col] if pagevalues_col in X_train.columns else []
    without_pv = evaluate_variant(
        X_train.drop(columns=drop_cols),
        X_test.drop(columns=drop_cols),
        y_train,
        y_test,
        categorical_columns,
        threshold=threshold,
        label="without_PageValues",
    )

    rows = []
    for result in (with_pv, without_pv):
        rows.append(
            {
                "variant": result["variant"],
                "n_features_raw": result["n_features_raw"],
                "PR-AUC": round(result["pr_auc"], 4),
                "Precision": round(result["precision"], 4),
                "Recall": round(result["recall"], 4),
                "F1": round(result["f1"], 4),
            }
        )

    table = pd.DataFrame(rows)
    table.attrs["with_PageValues"] = with_pv
    table.attrs["without_PageValues"] = without_pv
    return table
