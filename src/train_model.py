from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

TRAINING_DATA_PATH = BASE_DIR / "data" / "customer_spending_training.csv"
MODEL_PARAMS_PATH = BASE_DIR / "model" / "model_params.json"

FEATURES = [
    "customer_age",
    "Income",
    "children_home",
    "Recency",
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
]

TARGET = "total_spending"


def train_and_save_model():
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {TRAINING_DATA_PATH}")

    df = pd.read_csv(TRAINING_DATA_PATH)

    required_columns = FEATURES + [TARGET]
    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.dropna(subset=required_columns)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    model_params = {
        "features": FEATURES,
        "intercept": float(model.intercept_),
        "coefficients": {
            feature: float(coef)
            for feature, coef in zip(FEATURES, model.coef_)
        },
        "validation_mae": float(mae),
        "validation_rmse": float(rmse)
    }

    MODEL_PARAMS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PARAMS_PATH, "w") as f:
        json.dump(model_params, f, indent=4)

    print("Model training completed.")
    print(f"Validation MAE: {mae:.2f}")
    print(f"Validation RMSE: {rmse:.2f}")
    print(f"Model parameters saved to: {MODEL_PARAMS_PATH}")

    return model_params


if __name__ == "__main__":
    train_and_save_model()
