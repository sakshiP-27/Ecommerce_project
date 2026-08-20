import pandas as pd
from src.config.settings import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()
dataset_path = config.get("data_path", "")


def load_data() -> pd.DataFrame:
    if not dataset_path:
        logger.error("Dataset path is not specified in the configuration.")
        raise ValueError("Dataset path is not specified in the configuration.")

    df = pd.read_csv(dataset_path)
    logger.info(f"Data loaded successfully from {dataset_path}.")

    expected_columns = ["Administrative", "Informational", "ProductRelated", "BounceRates", "ExitRates", "PageValues", "Month", "VisitorType", "Weekend", "Converted"]

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing expected columns in the dataset: {missing_columns}")
        raise ValueError(f"Missing expected columns in the dataset: {missing_columns}")
    logger.info("All expected columns are present in the dataset.")

    return df

# if __name__ == "__main__":
#     try:
#         data = load_data()
#         logger.info("Data loading completed successfully.")
#     except Exception as e:
#         logger.exception("An error occurred while loading the data.")
