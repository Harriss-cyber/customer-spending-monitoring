from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PARAMS_PATH = BASE_DIR / "model" / "model_params.json"


def load_model_params():
    if not MODEL_PARAMS_PATH.exists():
        raise FileNotFoundError(f"Model parameter file not found: {MODEL_PARAMS_PATH}")

    with open(MODEL_PARAMS_PATH, "r") as f:
        model_params = json.load(f)

    return model_params


def predict_customer_spending(input_data, model_params=None):
    if model_params is None:
        model_params = load_model_params()

    prediction = model_params["intercept"]

    for feature in model_params["features"]:
        prediction += float(input_data[feature]) * float(model_params["coefficients"][feature])

    return float(prediction)
