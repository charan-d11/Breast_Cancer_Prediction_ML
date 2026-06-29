"""
model.py
Breast Cancer Prediction - KNN Model Training Script
Trains a KNN classifier on the Wisconsin Breast Cancer Dataset and saves the model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/dataset.csv"
MODEL_PATH = "model/knn_model.pkl"
SCALER_PATH = "model/scaler.pkl"

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the breast cancer CSV dataset."""
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame):
    """
    Preprocess the dataset:
      - Drop unnecessary columns (id, Unnamed)
      - Encode target: M → 1, B → 0
      - Split into features and target
    """
    # Drop id column and any unnamed trailing columns
    drop_cols = [c for c in df.columns if c.lower() in ("id", "unnamed: 32")]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode diagnosis
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    return X, y


def train(X_train, y_train, n_neighbors: int = 5) -> tuple:
    """Scale features and train a KNN classifier."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def evaluate(model, scaler, X_test, y_test) -> dict:
    """Evaluate the trained model on the test set."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return results


def save_artifacts(model, scaler, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """Save the trained model and scaler to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Model  saved → {model_path}")
    print(f"✅ Scaler saved → {scaler_path}")


def load_artifacts(model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
    """Load a previously saved model and scaler."""
    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict_single(input_dict: dict, model=None, scaler=None):
    """
    Predict for a single sample given as a dict of feature_name → value.
    Loads saved artifacts if model/scaler are not provided.
    Returns 'Malignant' or 'Benign' along with confidence probabilities.
    """
    if model is None or scaler is None:
        model, scaler = load_artifacts()

    df_input = pd.DataFrame([input_dict])
    df_scaled = scaler.transform(df_input)

    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0]   # [P(Benign), P(Malignant)]

    label = "Malignant" if prediction == 1 else "Benign"
    return label, proba


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("📂 Loading data …")
    df = load_data()
    print(f"   Shape: {df.shape}")

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🏋️  Training KNN model …")
    model, scaler = train(X_train, y_train, n_neighbors=5)

    print("📊 Evaluation on test set:")
    results = evaluate(model, scaler, X_test, y_test)
    print(f"   Accuracy : {results['accuracy'] * 100:.2f}%")
    print(results["report"])

    save_artifacts(model, scaler)
    print("\n✨ Done! Run app.py with: streamlit run app.py")
