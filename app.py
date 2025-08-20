import streamlit as st
import pandas as pd
import joblib

# Load models
xgb_model = joblib.load("model_xgb.pkl")
cat_model = joblib.load("model_cat.pkl")

st.title("Loan Approval Prediction")

# User inputs
income = st.number_input("Applicant Income")
loan_amount = st.number_input("Loan Amount")
# Add other fields as needed

# Predict button
if st.button("Predict"):
    data = pd.DataFrame([[income, loan_amount]], columns=['Income', 'LoanAmount'])
    pred_xgb = xgb_model.predict(data)[0]
    pred_cat = cat_model.predict(data)[0]
    st.write(f"XGBoost Prediction: {pred_xgb}")
    st.write(f"CatBoost Prediction: {pred_cat}")
