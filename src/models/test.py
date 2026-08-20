import joblib
import pandas as pd
from src.data.loader import load_data
from src.config.settings import load_config
from src.utils.logger import get_logger 

logger = get_logger(__name__)
config = load_config()

# Load the saved model
model = joblib.load("src/models/model_v0.pkl")

# Load data and grab one sample row
df = load_data()
df = pd.get_dummies(df, columns=config["categorical_columns"])
X = df.drop("Converted", axis=1)

sample = X.iloc[[0]]  # first row as a DataFrame

# Predict
probability = model.predict_proba(sample)[0][1]
logger.info(f"Predicted purchase probability: {probability:.2f}")
