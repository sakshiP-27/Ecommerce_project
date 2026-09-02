"""Quick deliberate-shift test for the drift checker."""

from __future__ import annotations

from sklearn.model_selection import train_test_split

from src.config.settings import load_config
from src.data.loader import load_data
from src.evaluation.drift import check_dataframe_drift, check_drift
from src.features.engineering import add_engineered_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_drift_self_test() -> None:
    config = load_config()
    df = add_engineered_features(load_data())
    train_df, holdout_df = train_test_split(
        df,
        test_size=config["test_size"],
        random_state=config["random_seed"],
    )

    # Control: same distribution family → should NOT flag batch drift
    control = check_dataframe_drift(train_df, holdout_df)
    assert control["batch_drifted"] is False, (
        f"Unexpected drift on unshifted holdout: {control['drifted_features']}"
    )
    logger.info("Control OK: %s", control["message"])

    # Deliberate shift: push a few numeric columns hard
    shifted = holdout_df.copy()
    for col in ["PageValues", "BounceRates", "ExitRates", "ProductRelated"]:
        if col in shifted.columns:
            shifted[col] = shifted[col] * 5 + 10

    assert check_drift(train_df["PageValues"], shifted["PageValues"]) is True
    drifted = check_dataframe_drift(train_df, shifted)
    assert drifted["batch_drifted"] is True, "Deliberate shift should flag batch drift"
    logger.info(
        "Shifted OK: %s (drifted=%s)",
        drifted["message"],
        drifted["drifted_features"],
    )
    print("Drift self-test passed.")
    print("  control:", control["message"], f"({control['n_drifted']} flagged)")
    print("  shifted:", drifted["message"], f"({drifted['n_drifted']} flagged)")


if __name__ == "__main__":
    run_drift_self_test()
