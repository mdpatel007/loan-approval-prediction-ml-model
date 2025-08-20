import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("data/loan_data.csv")

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Clean target labels
    df["loan_status"] = df["loan_status"].str.strip().map({"Approved": 1, "Rejected": 0})

    # Create total assets
    df["total_assets"] = (
        df["residential_assets_value"] +
        df["commercial_assets_value"] +
        df["luxury_assets_value"] +
        df["bank_asset_value"]
    )

    X = df[["cibil_score", "no_of_dependents", "total_assets", "loan_amount", "loan_term"]]
    y = df["loan_status"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_models(data):
    X_train, X_test, y_train, y_test = data
    xgb = XGBClassifier(eval_metric="logloss")
    cat = CatBoostClassifier(verbose=0)
    xgb.fit(X_train, y_train)
    cat.fit(X_train, y_train)
    return xgb, cat

def predict(model, input_df):
    return model.predict(input_df)

def evaluate_models(data):
    X_train, X_test, y_train, y_test = data
    xgb, cat = train_models(data)
    models = {"XGBoost": xgb, "CatBoost": cat}
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 2),
            "Precision": round(precision_score(y_test, y_pred), 2),
            "Recall": round(recall_score(y_test, y_pred), 2),
            "F1 Score": round(f1_score(y_test, y_pred), 2)
        })
    return pd.DataFrame(results)

def plot_feature_importance(model):
    importance = model.feature_importances_
    features = ["cibil_score", "no_of_dependents", "total_assets", "loan_amount", "loan_term"]
    plt.figure(figsize=(8, 4))
    plt.bar(features, importance)
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
