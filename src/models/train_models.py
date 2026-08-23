import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from src.data.loader import load_data
from src.features.engineering import add_engineered_features
from src.models.trainer import train_and_evaluate   
from src.config.settings import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

def prepare_data():
    df = load_data()
    df = add_engineered_features(df)
    df = pd.get_dummies(df, columns=config['categorical_columns'])
    X = df.drop(columns=[config['target_column']],axis = 1)
    y = df[config['target_column']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_seed'])
    return X_train, X_test, y_train, y_test

def train_models():
    X_train, X_test, y_train, y_test = prepare_data()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=5000, class_weight='balanced'),
        "DecisionTree": DecisionTreeClassifier(class_weight='balanced'),
        "RandomForest": RandomForestClassifier(class_weight='balanced', n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "LightGBM": LGBMClassifier(verbose= -1),
        "CatBoost": CatBoostClassifier(verbose=0)
    }

    all_results = []
    for model_name, model in models.items():
        logger.info(f"Starting training for {model_name}.")
        result = train_and_evaluate(model, X_train, y_train, model_name=model_name)
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('average_precision', ascending=False)
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    logger.info(results_df.to_string(index=False))
    print("=" * 80)

    # Save results
    results_df.to_csv("src/models/model_comparison.csv", index=False)
    logger.info("Results saved to models/model_comparison.csv")

    return results_df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    train_models()

