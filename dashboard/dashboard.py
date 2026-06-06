import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))

from predict_utils import load_model_params, predict_customer_spending
from log_utils import log_prediction, load_monitoring_logs

TRAINING_DATA_PATH = BASE_DIR / "data" / "customer_spending_training.csv"
PRODUCTION_DATA_PATH = BASE_DIR / "data" / "customer_spending_production.csv"

st.set_page_config(
    page_title="Customer Spending Monitoring Dashboard",
    layout="wide"
)

st.title("Customer Spending Prediction and Monitoring Dashboard")

st.write(
    "This dashboard demonstrates how a deployed prediction model can be monitored "
    "using actual outcomes, latency, manual business feedback, and input drift indicators."
)

@st.cache_data
def load_training_data():
    return pd.read_csv(TRAINING_DATA_PATH)

@st.cache_data
def load_production_data():
    return pd.read_csv(PRODUCTION_DATA_PATH)

@st.cache_data
def load_params():
    return load_model_params()

training_data = load_training_data()
production_data = load_production_data()
model_params = load_params()

tab1, tab2, tab3 = st.tabs([
    "Prediction and Manual Feedback",
    "Monitoring Dashboard",
    "Agile Retrospective"
])

with tab1:
    st.header("Prediction and Manual Feedback")

    st.write(
        "Use this section to simulate a business user generating a prediction and "
        "manually providing feedback after comparing the prediction with actual customer spending."
    )

    st.sidebar.header("Customer Profile")

    customer_age = st.sidebar.slider("Customer Age", 18, 90, 45)

    income = st.sidebar.number_input(
        "Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    children_home = st.sidebar.slider("Children at Home", 0, 3, 1)
    recency = st.sidebar.slider("Recency", 0, 100, 30)

    num_deals_purchases = st.sidebar.slider("Number of Deal Purchases", 0, 20, 2)
    num_web_purchases = st.sidebar.slider("Number of Web Purchases", 0, 20, 4)
    num_catalog_purchases = st.sidebar.slider("Number of Catalog Purchases", 0, 20, 2)
    num_store_purchases = st.sidebar.slider("Number of Store Purchases", 0, 20, 5)
    num_web_visits_month = st.sidebar.slider("Number of Web Visits per Month", 0, 25, 5)

    input_data = {
        "customer_age": customer_age,
        "Income": income,
        "children_home": children_home,
        "Recency": recency,
        "NumDealsPurchases": num_deals_purchases,
        "NumWebPurchases": num_web_purchases,
        "NumCatalogPurchases": num_catalog_purchases,
        "NumStorePurchases": num_store_purchases,
        "NumWebVisitsMonth": num_web_visits_month,
    }

    input_df = pd.DataFrame([input_data])

    st.subheader("Input Data")
    st.dataframe(input_df)

    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None

    if "latency_ms" not in st.session_state:
        st.session_state["latency_ms"] = None

    if st.button("Run Prediction"):
        start_time = time.time()

        prediction = predict_customer_spending(
            input_data=input_data,
            model_params=model_params
        )

        latency_ms = (time.time() - start_time) * 1000

        st.session_state["prediction"] = float(prediction)
        st.session_state["latency_ms"] = float(latency_ms)

    if st.session_state["prediction"] is not None:
        st.subheader("Prediction Result")

        st.metric(
            "Predicted Customer Spending",
            f"RM {st.session_state['prediction']:,.2f}"
        )

        st.metric(
            "Prediction Latency",
            f"{st.session_state['latency_ms']:.2f} ms"
        )

        st.subheader("Manual Actual Outcome and Business Feedback")

        actual_spending = st.number_input(
            "Enter actual customer spending",
            min_value=0.0,
            value=0.0,
            step=10.0
        )

        feedback_score = st.slider(
            "Business feedback score (1 = Poor, 5 = Excellent)",
            1,
            5,
            4
        )

        feedback_text = st.text_area(
            "Business feedback comment",
            placeholder="Example: Prediction is useful for campaign planning."
        )

        if st.button("Submit Monitoring Log"):
            log_prediction(
                customer_age=customer_age,
                income=income,
                children_home=children_home,
                recency=recency,
                num_deals_purchases=num_deals_purchases,
                num_web_purchases=num_web_purchases,
                num_catalog_purchases=num_catalog_purchases,
                num_store_purchases=num_store_purchases,
                num_web_visits_month=num_web_visits_month,
                prediction=st.session_state["prediction"],
                actual_spending=actual_spending,
                latency_ms=st.session_state["latency_ms"],
                feedback_score=feedback_score,
                feedback_text=feedback_text,
            )

            st.success("Monitoring log saved successfully. Open the Monitoring Dashboard tab to view results.")
    else:
        st.info("Click Run Prediction to generate a prediction.")

with tab2:
    st.header("Model Monitoring Dashboard")

    logs = load_monitoring_logs()

    if logs.empty:
        st.warning(
            "No monitoring logs found yet. Please use the Prediction tab and submit at least one monitoring log."
        )
    else:
        st.subheader("Key Monitoring Metrics")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total Predictions Logged", len(logs))
        col2.metric("Average Feedback Score", f"{logs['feedback_score'].mean():.2f}")
        col3.metric("Average Latency", f"{logs['latency_ms'].mean():.2f} ms")
        col4.metric("Mean Absolute Error", f"{logs['absolute_error'].mean():.2f}")

        st.markdown("---")

        monitor_tab1, monitor_tab2, monitor_tab3, monitor_tab4 = st.tabs([
            "Model Performance",
            "Operational Health",
            "Business Feedback",
            "Input Drift"
        ])

        with monitor_tab1:
            st.subheader("Prediction Error Monitoring")

            mae = logs["absolute_error"].mean()
            rmse = np.sqrt(np.mean(logs["error"] ** 2))

            col_a, col_b = st.columns(2)
            col_a.metric("MAE", f"{mae:.2f}")
            col_b.metric("RMSE", f"{rmse:.2f}")

            error_chart = logs[["timestamp", "absolute_error"]].set_index("timestamp")
            st.line_chart(error_chart)

            st.write(
                "A rising error trend may indicate model performance degradation."
            )

        with monitor_tab2:
            st.subheader("Latency Monitoring")

            latency_chart = logs[["timestamp", "latency_ms"]].set_index("timestamp")
            st.line_chart(latency_chart)

            st.write(
                "Latency monitoring helps determine whether the prediction service remains responsive."
            )

        with monitor_tab3:
            st.subheader("Manual Business Feedback Monitoring")

            feedback_chart = logs[["timestamp", "feedback_score"]].set_index("timestamp")
            st.line_chart(feedback_chart)

            st.subheader("Recent Business Feedback Comments")

            comments = logs[logs["feedback_text"].astype(str).str.strip() != ""]
            comments = comments.sort_values("timestamp", ascending=False).head(10)

            if comments.empty:
                st.info("No feedback comments available.")
            else:
                for _, row in comments.iterrows():
                    st.write(f"**{row['timestamp']} | Score: {row['feedback_score']}**")
                    st.write(row["feedback_text"])
                    st.markdown("---")

        with monitor_tab4:
            st.subheader("Simple Input Drift Check")

            drift_features = [
                "Income",
                "customer_age",
                "children_home",
                "NumWebPurchases",
                "NumStorePurchases",
                "NumWebVisitsMonth",
            ]

            drift_rows = []

            for feature in drift_features:
                train_mean = training_data[feature].mean()
                prod_mean = production_data[feature].mean()
                difference = prod_mean - train_mean
                percent_change = (difference / train_mean) * 100 if train_mean != 0 else 0

                drift_rows.append({
                    "Feature": feature,
                    "Training Mean": train_mean,
                    "Production Mean": prod_mean,
                    "Difference": difference,
                    "Percent Change": percent_change,
                })

            drift_df = pd.DataFrame(drift_rows)

            st.dataframe(
                drift_df.style.format({
                    "Training Mean": "{:.2f}",
                    "Production Mean": "{:.2f}",
                    "Difference": "{:.2f}",
                    "Percent Change": "{:.2f}%"
                })
            )

            major_drift = drift_df[drift_df["Percent Change"].abs() > 20]

            if not major_drift.empty:
                st.error("Potential input drift detected in one or more features.")
            else:
                st.success("No major input drift detected using this simple mean comparison.")

        st.markdown("---")
        st.subheader("Raw Monitoring Logs")
        st.dataframe(logs)

with tab3:
    st.header("Agile Retrospective")

    st.subheader("Model Validation Result from Training Data")

    col1, col2 = st.columns(2)
    col1.metric("Validation MAE", f"{model_params['validation_mae']:.2f}")
    col2.metric("Validation RMSE", f"{model_params['validation_rmse']:.2f}")

    st.markdown("""
    Use the monitoring results to support the next sprint discussion.

### What went well?
- Was the model able to generate customer spending predictions?
- Was the prediction latency acceptable?
- Did business users provide useful feedback?

### What did not go well?
- Were prediction errors high?
- Was there evidence of input drift?
- Did business users report low confidence in predictions?

### What should be improved next?
- Should the model be retrained using newer customer data?
- Should additional customer features be added?
- Should data validation be improved?
- Should the feedback collection process be improved?

### Example Backlog Item
As a marketing manager, I want the customer spending prediction model retrained using recent customer behaviour data so that spending forecasts 
