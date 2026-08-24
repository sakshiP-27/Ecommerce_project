import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data.loader import load_data
from src.features.engineering import add_engineered_features
from src.config.settings import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

def save_final_artifacts():
    # Load and prepare data
    df = load_data()
    df = add_engineered_features(df)

    X = df.drop(config["target_column"], axis=1)
    y = df[config["target_column"]]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["random_seed"]
    )

    # Build preprocessing pipeline
    categorical_columns = config["categorical_columns"]
    numeric_columns = [col for col in X_train.select_dtypes(include="number").columns
                       if col not in categorical_columns]

    preprocessor = ColumnTransformer(transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numeric", RobustScaler(), numeric_columns),
    ])

    # Fit preprocessing on training data only
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    logger.info(f"Preprocessing fitted. Output shape: {X_train_processed.shape}")

    # Train final model with tuned parameters
    final_model = CatBoostClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        verbose=0
    )
    final_model.fit(X_train_processed, y_train)
    logger.info("Final CatBoost model trained")

    # Save both artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, "src/models/final_model.pkl")
    joblib.dump(preprocessor, "src/models/preprocessing_pipeline.pkl")

    logger.info("Saved: models/final_model.pkl")
    logger.info("Saved: models/preprocessing_pipeline.pkl")

if __name__ == "__main__":
    save_final_artifacts() 