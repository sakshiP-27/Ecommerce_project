"""One-hot vs CatBoost-style encoding for high-cardinality ID columns."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# The four "sneaky" ID columns from Week 1
ID_COLUMNS = ["OperatingSystems", "Browser", "Region", "TrafficType"]


def _numeric_columns(X: pd.DataFrame, categorical_columns: list[str]) -> list[str]:
    return [
        col
        for col in X.select_dtypes(include="number").columns
        if col not in categorical_columns
    ]


def _make_model() -> CatBoostClassifier:
    """Same final CatBoost config as Week 2."""
    return CatBoostClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        verbose=0,
    )


def _metrics(y_test, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "PR-AUC": round(float(average_precision_score(y_test, y_prob)), 4),
        "Precision": round(float(precision_score(y_test, y_pred)), 4),
        "Recall": round(float(recall_score(y_test, y_pred)), 4),
        "F1": round(float(f1_score(y_test, y_pred)), 4),
    }


class CatBoostStyleEncoder(BaseEstimator, TransformerMixin):
    """
    CatBoost-like target encoding for high-cardinality IDs.

    Same idea as category_encoders.CatBoostEncoder: replace each category with a
    smoothed conversion rate, using an online/leave-one-out style mean on the
    training rows (random order) so target leakage is reduced vs naive means.
    Transform on unseen rows uses the fitted category means (+ global prior).

    Implemented locally so notebooks still run when category_encoders is broken
    in the active kernel (common on mismatched Anaconda stacks).
    """

    def __init__(self, cols: list[str], prior_weight: float = 1.0, random_state: int = 42):
        self.cols = list(cols)
        self.prior_weight = float(prior_weight)
        self.random_state = int(random_state)
        self.global_mean_: float | None = None
        self.mappings_: dict[str, dict] = {}

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(y).astype(float).reset_index(drop=True)
        X = X.reset_index(drop=True)
        self.global_mean_ = float(y.mean())
        self.mappings_ = {}

        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(len(X))

        for col in self.cols:
            encoded = np.empty(len(X), dtype=float)
            counts: dict = {}
            sums: dict = {}

            # Online target mean in random order (CatBoost-style)
            for idx in order:
                key = X.at[idx, col]
                n = counts.get(key, 0)
                s = sums.get(key, 0.0)
                if n == 0:
                    encoded[idx] = self.global_mean_
                else:
                    encoded[idx] = (s + self.prior_weight * self.global_mean_) / (
                        n + self.prior_weight
                    )
                counts[key] = n + 1
                sums[key] = s + float(y.iloc[idx])

            # Final mapping for transform() = full-data smoothed means
            mapping = {}
            for key, n in counts.items():
                mapping[key] = (sums[key] + self.prior_weight * self.global_mean_) / (
                    n + self.prior_weight
                )
            self.mappings_[col] = mapping

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        out = pd.DataFrame(index=X.index)
        for col in self.cols:
            mapping = self.mappings_[col]
            out[col] = X[col].map(mapping).fillna(self.global_mean_).astype(float)
        return out

    def fit_transform(self, X, y=None, **fit_params):
        X = pd.DataFrame(X).copy().reset_index(drop=True)
        y = pd.Series(y).astype(float).reset_index(drop=True)
        self.global_mean_ = float(y.mean())
        self.mappings_ = {}

        rng = np.random.RandomState(self.random_state)
        order = rng.permutation(len(X))
        result = pd.DataFrame(index=X.index)

        for col in self.cols:
            encoded = np.empty(len(X), dtype=float)
            counts: dict = {}
            sums: dict = {}

            for idx in order:
                key = X.at[idx, col]
                n = counts.get(key, 0)
                s = sums.get(key, 0.0)
                if n == 0:
                    encoded[idx] = self.global_mean_
                else:
                    encoded[idx] = (s + self.prior_weight * self.global_mean_) / (
                        n + self.prior_weight
                    )
                counts[key] = n + 1
                sums[key] = s + float(y.iloc[idx])

            mapping = {}
            for key, n in counts.items():
                mapping[key] = (sums[key] + self.prior_weight * self.global_mean_) / (
                    n + self.prior_weight
                )
            self.mappings_[col] = mapping
            result[col] = encoded

        return result


def _get_catboost_encoder(cols: list[str]):
    """Prefer category_encoders if it imports cleanly; otherwise use local impl."""
    try:
        from category_encoders.cat_boost import CatBoostEncoder

        return CatBoostEncoder(cols=cols)
    except Exception:
        return CatBoostStyleEncoder(cols=cols)


def evaluate_onehot(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    categorical_columns: list[str],
    threshold: float = 0.5,
) -> dict:
    """Current project approach: one-hot all categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", RobustScaler(), _numeric_columns(X_train, categorical_columns)),
        ]
    )
    model = _make_model()

    t0 = time.perf_counter()
    X_train_t = preprocessor.fit_transform(X_train, y_train)
    model.fit(X_train_t, y_train)
    train_seconds = time.perf_counter() - t0

    X_test_t = preprocessor.transform(X_test)
    y_prob = model.predict_proba(X_test_t)[:, 1]
    return {
        "encoding": "one_hot",
        **_metrics(y_test, y_prob, threshold),
        "train_seconds": round(train_seconds, 3),
        "n_features_out": int(X_train_t.shape[1]),
    }


def evaluate_catboost_encoder(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    categorical_columns: list[str],
    threshold: float = 0.5,
    id_columns: list[str] | None = None,
) -> dict:
    """
    CatBoost-encode the four ID columns (fit on train only), then one-hot
    the remaining categoricals and scale numerics — including the encoded IDs.
    """
    id_columns = list(id_columns or ID_COLUMNS)
    other_cats = [c for c in categorical_columns if c not in id_columns]

    missing = [c for c in id_columns if c not in X_train.columns]
    if missing:
        raise ValueError(f"Missing ID columns for CatBoost encoding: {missing}")

    encoder = _get_catboost_encoder(id_columns)

    t0 = time.perf_counter()
    train_ids = encoder.fit_transform(X_train[id_columns], y_train)
    test_ids = encoder.transform(X_test[id_columns])

    # category_encoders returns DataFrame; local encoder does too
    train_ids = pd.DataFrame(train_ids)
    test_ids = pd.DataFrame(test_ids)
    if list(train_ids.columns) != id_columns:
        train_ids.columns = id_columns
        test_ids.columns = id_columns

    X_train_enc = X_train.drop(columns=id_columns).copy()
    X_test_enc = X_test.drop(columns=id_columns).copy()
    for col in id_columns:
        X_train_enc[col] = train_ids[col].to_numpy()
        X_test_enc[col] = test_ids[col].to_numpy()

    numeric_cols = _numeric_columns(X_train_enc, other_cats)
    preprocessor = ColumnTransformer(
        transformers=[
            ("other_cats", OneHotEncoder(handle_unknown="ignore"), other_cats),
            ("numeric", RobustScaler(), numeric_cols),
        ]
    )
    X_train_t = preprocessor.fit_transform(X_train_enc)
    model = _make_model()
    model.fit(X_train_t, y_train)
    train_seconds = time.perf_counter() - t0

    X_test_t = preprocessor.transform(X_test_enc)
    y_prob = model.predict_proba(X_test_t)[:, 1]
    return {
        "encoding": "catboost_encoder_ids",
        **_metrics(y_test, y_prob, threshold),
        "train_seconds": round(train_seconds, 3),
        "n_features_out": int(X_train_t.shape[1]),
        "encoder_backend": type(encoder).__name__,
    }


def compare_encodings(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_test,
    categorical_columns: list[str],
    threshold: float = 0.5,
    id_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Side-by-side: full one-hot vs CatBoost-style encoding on the four ID columns."""
    onehot = evaluate_onehot(
        X_train, X_test, y_train, y_test, categorical_columns, threshold
    )
    catboost_enc = evaluate_catboost_encoder(
        X_train,
        X_test,
        y_train,
        y_test,
        categorical_columns,
        threshold,
        id_columns=id_columns,
    )
    # Keep table columns aligned
    onehot["encoder_backend"] = "OneHotEncoder"
    return pd.DataFrame([onehot, catboost_enc])
