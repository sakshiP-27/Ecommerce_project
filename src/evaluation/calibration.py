"""Probability calibration / reliability-curve helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss


def reliability_bins(
    y_true,
    y_probs,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> pd.DataFrame:
    """Return calibration_curve points as a small DataFrame."""
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    frac_pos, mean_pred = calibration_curve(
        y_true,
        y_probs,
        n_bins=n_bins,
        strategy=strategy,
    )
    return pd.DataFrame(
        {
            "mean_predicted_probability": mean_pred,
            "fraction_of_positives": frac_pos,
        }
    )


def brier_score(y_true, y_probs) -> float:
    """Brier score (lower is better calibrated + accurate)."""
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(y_probs)))


def fit_isotonic_calibrator(base_model, X_train, y_train, cv: int = 5):
    """
    Wrap a classifier with isotonic CalibratedClassifierCV.

    Fits on the same feature space the base model expects
    (already preprocessed arrays, not raw DataFrames).
    """
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated
