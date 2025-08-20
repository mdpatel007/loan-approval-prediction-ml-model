import streamlit as st
import pandas as pd
from model_utils import load_data, train_models, predict, evaluate_models, plot_feature_importance
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")
st.title("🏦 Loan Approval Prediction App")
st.markdown("Compare XGBoost and CatBoost models on loan approval prediction.")

# Sidebar for user input
st.sidebar.header("📋 Enter Applicant Details")
cibil = st.sidebar.slider("CIBIL Score", 300, 900, 650)
dependents = st.sidebar.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
movable = st.sidebar.number_input("Movable Assets (₹)", min_value=0)
immovable = st.sidebar.number_input("Immovable Assets (₹)", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", min_value=0)
tenure = st.sidebar.slider("Loan Tenure (months)", 6, 240, 60)

# Calculate total assets
assets = movable + immovable

# Load and train models
data = load_data()
xgb_model, cat_model = train_models(data)

# Prepare input
input_df = pd.DataFrame([[cibil, dependents, assets, loan_amount, tenure]],
    columns=["cibil_score", "no_of_dependents", "total_assets", "loan_amount", "loan_term"])

# Predictions
xgb_pred = predict(xgb_model, input_df)
cat_pred = predict(cat_model, input_df)

st.subheader("🔍 Prediction Results")
st.write(f"**XGBoost Prediction:** {'✅ Approved' if xgb_pred[0]==1 else '❌ Rejected'}")
st.write(f"**CatBoost Prediction:** {'✅ Approved' if cat_pred[0]==1 else '❌ Rejected'}")

# Evaluation
st.subheader("📊 Model Performance")
metrics = evaluate_models(data)
st.dataframe(metrics)

# Feature Importance
st.subheader("📈 Feature Importance (XGBoost)")
plot_feature_importance(xgb_model)
st.pyplot(plt)

# Conclusion
with st.expander("📘 Business Insights from Model"):
    st.markdown("""
    - ✅ **CIBIL Score**: Higher score → higher approval chance  
    - 👨‍👩‍👧 **Dependents**: More dependents → lower approval chance  
    - 💰 **Assets**: More assets → higher approval chance  
    - 📊 **Loan Amount & Tenure**: Higher amount with shorter tenure → more likely approved  
    - 🤖 **Model Performance**: Both models ~98% accurate — either is reliable
    """)

