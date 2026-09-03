"""
Data preprocessing pipeline for credit card fraud detection.

Loads the raw Kaggle dataset, performs feature engineering, and produces a
stratified train/test split saved to ``data/processed/``. The fitted
``StandardScaler`` is persisted to ``models/scaler.pkl`` for reuse at
prediction time.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Resolve paths relative to the project root so scripts work from any cwd.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "creditcard.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_data(path=RAW_PATH):
    """Load the raw dataset into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download it from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place creditcard.csv in data/raw/."
        )
    return pd.read_csv(path)


def preprocess(df):
    """Feature-engineer the raw DataFrame.

    Steps:
        1. Scale ``Amount`` with StandardScaler (V1-V28 are already PCA-scaled).
        2. Derive ``Hour`` of day from ``Time`` (seconds since first txn).
        3. Add ``Log_Amount`` = log1p(Amount).
        4. Drop the original ``Time`` and ``Amount`` columns.

    Returns:
        (transformed_df, fitted_scaler)
    """
    df = df.copy()
    scaler = StandardScaler()
    df["Scaled_Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Hour"] = (df["Time"] / 3600).astype(int) % 24
    df["Log_Amount"] = np.log1p(df["Amount"])
    df.drop(columns=["Time", "Amount"], inplace=True)
    return df, scaler


def split_and_save(df, test_size=0.2, random_state=42):
    """Stratified 80/20 train/test split, saved to ``data/processed/``."""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    return X_train, X_test, y_train, y_test


def main():
    """Run the full preprocessing pipeline and persist artifacts."""
    df = load_data()
    df, scaler = preprocess(df)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    X_train, X_test, y_train, y_test = split_and_save(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(
        f"Fraud rate - Train: {y_train.mean():.4f}, "
        f"Test: {y_test.mean():.4f}"
    )


if __name__ == "__main__":
    main()
