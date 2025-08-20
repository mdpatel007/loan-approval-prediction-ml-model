# 🏦 Loan Approval Prediction App

A Streamlit-powered web app that predicts loan approval based on applicant details using a trained machine learning model. Built for recruiters, interviewers, and anyone curious about real-world ML deployment.

## 🚀 Live Demo

🔗 https://loan-approval-prediction-ml-model-pcp8fwof2pszzbm9npr7ls.streamlit.app/#model-performance

---

## 📌 Features

- ✅ Predicts loan approval status based on user input
- 📊 Uses CatBoostClassifier for high accuracy on tabular data
- 🧠 Includes preprocessing, feature engineering, and model evaluation
- 🌐 Deployed on Streamlit Cloud for instant access
- 📁 Modular codebase with reusable components

---

## 📂 Project Structure

Loan_approval_prediction/ 
├── app.py # Streamlit app 
├── model_utils.py # ML pipeline and prediction logic 
├── loan_approval_model.ipynb # Notebook for training & evaluation 
├── data/ # Dataset folder 
├── requirements.txt # Dependencies 


---

## 🧪 Model Details

- **Algorithm:** CatBoostClassifier  
- **Accuracy:** ~98% on test set  
- **Features Used:** Gender, Married, Education, ApplicantIncome, LoanAmount, Credit_History, etc.

---

## 📥 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/mdpatel007/Loan_approval_prediction.git
cd Loan_approval_prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py


