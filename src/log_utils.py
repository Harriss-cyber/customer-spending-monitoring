from datetime import datetime
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "data" / "monitoring_logs.csv"


def log_prediction(
    customer_age,
    income,
    children_home,
    recency,
    num_deals_purchases,
    num_web_purchases,
    num_catalog_purchases,
    num_store_purchases,
    num_web_visits_month,
    prediction,
    actual_spending,
    latency_ms,
    feedback_score,
    feedback_text,
):
    error = float(actual_spending) - float(prediction)
    absolute_error = abs(error)

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_age": customer_age,
        "Income": income,
        "children_home": children_home,
        "Recency": recency,
        "NumDealsPurchases": num_deals_purchases,
        "NumWebPurchases": num_web_purchases,
        "NumCatalogPurchases": num_catalog_purchases,
        "NumStorePurchases": num_store_purchases,
        "NumWebVisitsMonth": num_web_visits_month,
        "prediction": float(prediction),
        "actual_spending": float(actual_spending),
        "error": error,
        "absolute_error": absolute_error,
        "latency_ms": float(latency_ms),
        "feedback_score": int(feedback_score),
        "feedback_text": feedback_text or "",
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_new = pd.DataFrame([row])

    if LOG_PATH.exists():
        df_new.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(LOG_PATH, index=False)

    return row


def load_monitoring_logs():
    if not LOG_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(LOG_PATH, parse_dates=["timestamp"])
