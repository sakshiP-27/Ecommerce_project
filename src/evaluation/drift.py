"""Simple Kolmogorov–Smirnov data-drift checks."""

from __future__ import annotations

import pandas as pd
from scipy.stats import ks_2samp

from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_drift(train_col, new_col, threshold: float = 0.05) -> bool:
    """
    Return True if the KS test says the two samples likely differ
    (p-value < threshold).
    """
    train_col = pd.Series(train_col).dropna().astype(float)
    new_col = pd.Series(new_col).dropna().astype(float)
    if len(train_col) < 2 or len(new_col) < 2:
        return False
    _stat, p_value = ks_2samp(train_col, new_col)
    return bool(p_value < threshold)


def check_dataframe_drift(
    train_df: pd.DataFrame,
    new_df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.05,
    min_drifted_features: int = 3,
) -> dict:
    """
    Run KS drift checks on numeric columns.

    A batch is flagged as drifting when at least `min_drifted_features`
    columns have p-value < threshold.
    """
    if columns is None:
        columns = [
            c
            for c in train_df.select_dtypes(include="number").columns
            if c in new_df.columns
        ]

    rows = []
    drifted_cols: list[str] = []
    for col in columns:
        train_col = train_df[col].dropna().astype(float)
        new_col = new_df[col].dropna().astype(float)
        if len(train_col) < 2 or len(new_col) < 2:
            continue
        stat, p_value = ks_2samp(train_col, new_col)
        drifted = bool(p_value < threshold)
        if drifted:
            drifted_cols.append(col)
        rows.append(
            {
                "feature": col,
                "ks_statistic": float(stat),
                "p_value": float(p_value),
                "drifted": drifted,
            }
        )

    details = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame()
    batch_drifted = len(drifted_cols) >= min_drifted_features

    if batch_drifted:
        logger.warning(
            "Data drift detected: %s/%s numeric features flagged (threshold=%.3f). "
            "Examples: %s",
            len(drifted_cols),
            len(rows),
            threshold,
            ", ".join(drifted_cols[:5]),
        )
    else:
        logger.info(
            "No batch-level drift: %s/%s features flagged (need >= %s).",
            len(drifted_cols),
            len(rows),
            min_drifted_features,
        )

    return {
        "batch_drifted": batch_drifted,
        "n_features_checked": len(rows),
        "n_drifted": len(drifted_cols),
        "drifted_features": drifted_cols,
        "details": details,
        "message": (
            "Some features have drifted vs training data"
            if batch_drifted
            else "Data looks consistent with training data"
        ),
    }
