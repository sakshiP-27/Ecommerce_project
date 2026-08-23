import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.data.loader import load_data
from src.features.engineering import add_engineered_features
from src.config.settings import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = load_config()

def prepare_data():
    df = load_data()
    df = add_engineered_features(df)
    df = pd.get_dummies(df, columns=config["categorical_columns"])

    X = df.drop(config["target_column"], axis=1)
    y = df[config["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["test_size"],
        random_state=config["random_seed"]
    )
    return X_train, X_test, y_train, y_test

def tune_model():
    X_train, X_test, y_train, y_test = prepare_data()
    cv = StratifiedKFold(n_splits= 5, shuffle=True, random_state=config["random_seed"])

    param_grid = {
        "XGBoost": {
            "model": XGBClassifier(eval_metric='logloss', use_label_encoder=False),
            "params": {
                "n_estimators": [100, 200, 500],
                "max_depth": [3, 5, 7, 9],
                "learning_rate": [0.01,0.05, 0.1, 0.2],
            }
        },
        "LightGBM": {
            "model": LGBMClassifier(verbose=-1),
            "params": {
                "n_estimators": [100, 200, 500],
                "max_depth": [-1,3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
            }
        },
        "CatBoost": {
            "model": CatBoostClassifier(verbose=0),
            "params": {
                "n_estimators": [100, 200, 500],
                "max_depth": [4, 6, 8, 10],
                "learning_rate": [0.01, 0.05, 0.1, 0.2]
            }
        }
    }

    tuned_results = []
    for name , config_item in param_grid.items():
        logger.info(f"Tuning {name}")

        search = RandomizedSearchCV(
            estimator=config_item["model"],
            param_distributions=config_item["params"],
            n_iter=20,
            scoring='average_precision',
            cv = cv,
            verbose=1,
            random_state=config['random_seed'],
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        logger.info(f"{name} — Best PR-AUC: {search.best_score_:.4f}")
        logger.info(f"{name} — Best params: {search.best_params_}")

        tuned_results.append({
            "model": name,
            "best_pr_auc": search.best_score_,
            "best_params": search.best_params_,
        })

    # Print comparison
    print("\n" + "=" * 80)
    print("TUNING RESULTS (sorted by PR-AUC)")
    print("=" * 80)
    for r in sorted(tuned_results, key=lambda x: x["best_pr_auc"], reverse=True):
        print(f"{r['model']:20s} | PR-AUC: {r['best_pr_auc']:.4f} | Params: {r['best_params']}")
    print("=" * 80)

    return tuned_results

if __name__ == "__main__":
    tune_model()



