import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.utils.logger import get_logger
from src.config.settings import load_config

config = load_config()
logger = get_logger(__name__)

def train_and_evaluate(model, X_train, y_train, model_name = "model", cv_folds = 5):
    cv= StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config['random_seed'])
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
    logger.info(f"Training {model_name} with {cv_folds}-fold cross-validation.")
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    results = {
        'model_name': model_name,
        'accuracy': np.mean(scores['test_accuracy']),
        'precision': np.mean(scores['test_precision']),
        'recall': np.mean(scores['test_recall']),
        'f1': np.mean(scores['test_f1']),
        'roc_auc': np.mean(scores['test_roc_auc']),
        'average_precision': np.mean(scores['test_average_precision'])
    }
 
    logger.info(f"Completed training {model_name}. Results: {results}")
    return results





