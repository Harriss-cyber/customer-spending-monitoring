import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from predict_utils import load_model_params, predict_customer_spending
from log_utils import log_prediction


def test_model_params_exist():
    params = load_model_params()

    assert "intercept" in params
    assert "coefficients" in params
    assert "features" in params


def test_prediction_output_is_numeric():
    params = load_model_params()

    input_data = {
        "customer_age": 40,
        "Income": 50000,
        "children_home": 1,
        "Recency": 30,
        "NumDealsPurchases": 2,
        "NumWebPurchases": 4,
        "NumCatalogPurchases": 2,
        "NumStorePurchases": 5,
        "NumWebVisitsMonth": 6,
    }

    prediction = predict_customer_spending(input_data, params)

    assert isinstance(prediction, float)


def test_log_prediction_creates_correct_error():
    row = log_prediction(
        customer_age=40,
        income=50000,
        children_home=1,
        recency=30,
        num_deals_purchases=2,
        num_web_purchases=4,
        num_catalog_purchases=2,
        num_store_purchases=5,
        num_web_visits_month=6,
        prediction=1000.0,
        actual_spending=850.0,
        latency_ms=12.5,
        feedback_score=4,
        feedback_text="Test log",
    )

    assert row["absolute_error"] == 150.0
    assert row["feedback_score"] == 4
