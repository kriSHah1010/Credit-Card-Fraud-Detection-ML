"""
Model training pipeline for credit card fraud detection.

Loads the processed training set, balances it with SMOTE, trains three
classifiers (Logistic Regression, Random Forest, XGBoost) with 5-fold
cross-validation, and saves each as a pickle. The model with the best CV
ROC-AUC is also copied to ``models/best_model.pkl``.
"""
import os
import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def build_models():
    """Return a fresh dict of model name -> estimator."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost_model": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="aucpr",
            n_jobs=-1,
        ),
    }


def load_train_data():
    """Load the processed training features and target."""
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    X = train.drop(columns=["Class"])
    y = train["Class"]
    return X, y


def apply_smote(X, y, random_state=42):
    """Balance classes with SMOTE oversampling."""
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X, y)


def train_all_models(X_train, y_train):
    """Train all models, run CV, persist pickles, and pick the best one."""
    models = build_models()

    # scale_pos_weight helps XGBoost when the resampled data still leans neg.
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    models["xgboost_model"].set_params(scale_pos_weight=neg / max(pos, 1))

    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1
        )
        print(f"  CV ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        with open(os.path.join(MODELS_DIR, f"{name}.pkl"), "wb") as f:
            pickle.dump(model, f)
        results[name] = {"model": model, "cv_auc": cv_scores.mean()}

    best_name = max(results, key=lambda k: results[k]["cv_auc"])
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(results[best_name]["model"], f)
    print(
        f"\nBest model: {best_name} "
        f"(CV AUC: {results[best_name]['cv_auc']:.4f})"
    )
    return results


def main():
    X, y = load_train_data()
    X_sm, y_sm = apply_smote(X, y)
    print(f"After SMOTE - Fraud: {int((y_sm == 1).sum())}, "
          f"Legit: {int((y_sm == 0).sum())}")
    train_all_models(X_sm, y_sm)


if __name__ == "__main__":
    main()
