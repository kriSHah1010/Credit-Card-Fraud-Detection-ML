"""
Prediction utility for credit card fraud detection.

Loads the best trained model and the fitted scaler, then predicts the fraud
probability for a single transaction supplied as a dict of raw features
(V1-V28, Amount, Time).
"""
import os
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

FEATURE_ORDER = (
    [f"V{i}" for i in range(1, 29)] + ["Scaled_Amount", "Hour", "Log_Amount"]
)


def load_model(
    model_path=os.path.join(MODELS_DIR, "best_model.pkl"),
    scaler_path=os.path.join(MODELS_DIR, "scaler.pkl"),
):
    """Load the pickled best model and scaler."""
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_transaction(model, scaler, transaction_dict):
    """Predict fraud for a single transaction.

    Args:
        model: fitted classifier with ``predict_proba``.
        scaler: fitted StandardScaler for ``Amount``.
        transaction_dict: dict with keys V1-V28, Amount, Time.

    Returns:
        dict with ``prediction`` (0/1) and ``fraud_probability`` (float).
    """
    df = pd.DataFrame([transaction_dict])
    df["Scaled_Amount"] = scaler.transform(df[["Amount"]])
    df["Hour"] = (df["Time"] / 3600).astype(int) % 24
    df["Log_Amount"] = np.log1p(df["Amount"])
    df.drop(columns=["Time", "Amount"], inplace=True)
    df = df.reindex(columns=FEATURE_ORDER, fill_value=0.0)

    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(model.predict(df)[0])
    return {"prediction": pred, "fraud_probability": round(prob, 4)}


def main():
    model, scaler = load_model()
    rng = np.random.default_rng(42)
    sample = {f"V{i}": float(rng.standard_normal()) for i in range(1, 29)}
    sample.update({"Amount": 25.50, "Time": 45000})
    result = predict_transaction(model, scaler, sample)
    print(f"Prediction: {'FRAUD' if result['prediction'] else 'LEGIT'}")
    print(f"Fraud Probability: {result['fraud_probability']:.2%}")


if __name__ == "__main__":
    main()
