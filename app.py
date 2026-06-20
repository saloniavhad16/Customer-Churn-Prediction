import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("model/churn_model.pkl")

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure", min_value=0)
monthlycharges = st.number_input("Monthly Charges", min_value=0.0)
totalcharges = st.number_input("Total Charges", min_value=0.0)

if st.button("Predict"):

    input_data = pd.DataFrame([{
        'SeniorCitizen': 0,
        'tenure': tenure,
        'MonthlyCharges': monthlycharges,
        'TotalCharges': totalcharges,
        'gender_Male': 0,
        'Partner_Yes': 0,
        'Dependents_Yes': 0,
        'PhoneService_Yes': 1,
        'MultipleLines_No phone service': 0,
        'MultipleLines_Yes': 0,
        'InternetService_Fiber optic': 0,
        'InternetService_No': 0,
        'OnlineSecurity_No internet service': 0,
        'OnlineSecurity_Yes': 0,
        'OnlineBackup_No internet service': 0,
        'OnlineBackup_Yes': 0,
        'DeviceProtection_No internet service': 0,
        'DeviceProtection_Yes': 0,
        'TechSupport_No internet service': 0,
        'TechSupport_Yes': 0,
        'StreamingTV_No internet service': 0,
        'StreamingTV_Yes': 0,
        'StreamingMovies_No internet service': 0,
        'StreamingMovies_Yes': 0,
        'Contract_One year': 0,
        'Contract_Two year': 0,
        'PaperlessBilling_Yes': 1,
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0,
        'PaymentMethod_Mailed check': 0
    }])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer is likely to leave.")
    else:
        st.success("Customer is likely to stay.")