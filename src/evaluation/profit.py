"""Profit / expected-value threshold helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_value(
    y_true,
    y_probs,
    threshold: float,
    intervention_cost: float,
    conversion_value: float,
) -> float:
    """
    Net expected value at a decision threshold.

    Assumes we intervene when pred == 1:
    - true positives earn conversion_value each
    - false positives cost intervention_cost each
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=float)
    preds = (y_probs >= threshold).astype(int)

    true_positives = int(((preds == 1) & (y_true == 1)).sum())
    false_positives = int(((preds == 1) & (y_true == 0)).sum())
    return (true_positives * conversion_value) - (false_positives * intervention_cost)


def profit_curve(
    y_true,
    y_probs,
    intervention_cost: float,
    conversion_value: float,
    step: float = 0.01,
) -> pd.DataFrame:
    """Sweep thresholds and return net expected value for each."""
    thresholds = np.round(np.arange(0.0, 1.0 + step, step), 4)
    rows = []
    for t in thresholds:
        value = expected_value(
            y_true,
            y_probs,
            threshold=float(t),
            intervention_cost=intervention_cost,
            conversion_value=conversion_value,
        )
        rows.append({"threshold": float(t), "expected_value": float(value)})

    curve = pd.DataFrame(rows)
    best_idx = curve["expected_value"].idxmax()
    curve["is_optimal"] = curve.index == best_idx
    return curve
