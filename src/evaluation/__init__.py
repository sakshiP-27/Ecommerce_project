"""Evaluation helpers (stretch goals / business metrics)."""

from src.evaluation.ablation import run_pagevalues_ablation
from src.evaluation.calibration import (
    brier_score,
    fit_isotonic_calibrator,
    reliability_bins,
)
from src.evaluation.drift import check_dataframe_drift, check_drift
from src.evaluation.profit import expected_value, profit_curve

__all__ = [
    "expected_value",
    "profit_curve",
    "reliability_bins",
    "brier_score",
    "fit_isotonic_calibrator",
    "run_pagevalues_ablation",
    "compare_encodings",
    "check_drift",
    "check_dataframe_drift",
]


def __getattr__(name: str):
    # Lazy import so other stretch-goal helpers work even if category_encoders
    # is not installed yet.
    if name == "compare_encodings":
        from src.evaluation.encoding_compare import compare_encodings

        return compare_encodings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
