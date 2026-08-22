import pandas as pd
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
import numpy as np

from src.config.settings import load_config
from src.data.loader import load_data
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

def build_pipeline():
    categorical_columns = config["categorical_columns"]

    # All numeric columns (everything that's not categorical or the target)
    df = load_data()
    numeric_columns = [col for col in df.select_dtypes(include="number").columns
                       if col not in categorical_columns and col != config["target_column"]]

    preprocessor = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numeric", RobustScaler(), numeric_columns),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
    ])

    return pipeline, df, numeric_columns, categorical_columns

def run_preprocessing():
    pipeline, df, numeric_columns, categorical_columns = build_pipeline()

    X = df.drop(config["target_column"], axis=1)
    y = df[config["target_column"]]

    # Split — fit only on training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_seed"]
    )

    # Fit on training data only
    pipeline.fit(X_train)
    logger.info("Pipeline fitted on training data")

    # Transform both sets
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    logger.info(f"Training shape: {X_train_processed.shape}, Test shape: {X_test_processed.shape}")

    # Save the fitted pipeline
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "src/models/preprocessing_pipeline.pkl")
    logger.info("Fitted pipeline saved to models/preprocessing_pipeline.pkl")

    return X_train_processed, X_test_processed, y_train, y_test

if __name__ == "__main__":
    run_preprocessing()

