import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.data.loader import load_data
from src.utils.logger import get_logger
from src.config.settings import load_config


logger = get_logger(__name__)
config = load_config()
df = load_data()


def first_prediction():
    logger.info("dropping the target column")
    X = df.drop("Converted", axis = 1)
    y = df["Converted"]

    logger.info("Encoding categorical variables")
    X = pd.get_dummies(X, columns = config.get("categorical_columns", []))

    logger.info("Splitting the data into train and test sets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get("test_size", 0.2), random_state=config.get("random_seed", 42))

    logger.info("Training the Logistic Regression model")
    lr = LogisticRegression(max_iter=config.get("max_iter", 1000))
    lr.fit(X_train, y_train)

    logger.info("Calculating accuracy")
    y_pred = lr.predict(X_test)
    logger.info(f"accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%") 

    logger.info("Saving the trained model")
    joblib.dump(lr, "src/models/model_v0.pkl")

if __name__ == "__main__":
    first_prediction()
