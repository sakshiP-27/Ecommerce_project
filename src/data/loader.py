from pathlib import Path

import pandas as pd
from src.config.settings import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_raw_path = config.get("data_path", "")
dataset_path = (
    str((_PROJECT_ROOT / _raw_path).resolve())
    if _raw_path and not Path(_raw_path).is_absolute()
    else _raw_path
)


def load_data() -> pd.DataFrame:
    if not dataset_path:
        logger.error("Dataset path is not specified in the configuration.")
        raise ValueError("Dataset path is not specified in the configuration.")

    df = pd.read_csv(dataset_path)
    logger.info(f"Data loaded successfully from {dataset_path}.")

    expected_columns = [
        "Administrative",
        "Informational",
        "ProductRelated",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "Month",
        "VisitorType",
        "Weekend",
        "Converted",
    ]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing expected columns in the dataset: {missing_columns}")
        raise ValueError(f"Missing expected columns in the dataset: {missing_columns}")
    logger.info("All expected columns are present in the dataset.")

    return df
