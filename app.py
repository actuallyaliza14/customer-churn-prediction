import streamlit as st
import pandas as pd
import joblib

# Page setup
st.set_page_config(page_title="ChurnGuard", page_icon="🎬")

st.title("🎬 ChurnGuard")
st.caption("Subscription Cancellation Risk Checker")

# Load model + columns
model = joblib.load("churn_model.joblib")
model_columns = joblib.load("model_columns.joblib")

st.subheader("Enter subscriber information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.slider("Monthly Charges ($)", 0.0, 150.0, 70.0)
    paperless = st.selectbox("Paperless Billing?", ["Yes", "No"])
    tech = st.selectbox("Tech Support?", ["Yes", "No"])

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    total = st.number_input("Total Charges ($)", value=float(tenure * monthly))

def build_input():
    raw = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "PaperlessBilling": paperless,
        "TechSupport": tech,
        "Contract": contract,
        "PaymentMethod": payment,
        "InternetService": internet
    }])

    encoded = pd.get_dummies(raw, drop_first=True)
    aligned = encoded.reindex(columns=model_columns, fill_value=0)
    return aligned

st.divider()

if st.button("🎯 Predict Cancellation Risk"):
    X_input = build_input()
    prob = float(model.predict_proba(X_input)[0][1])
    pct = prob * 100

    if pct < 25:
        label = "🟢 Low Risk"
    elif pct < 60:
        label = "🟡 Medium Risk"
    else:
        label = "🔴 High Risk"

    st.subheader(f"{label}: {pct:.1f}% chance of cancellation")
    st.progress(min(int(pct), 100))
