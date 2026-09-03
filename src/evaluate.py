"""
Evaluation and explainability for credit card fraud detection.

Provides metric computation, comparison plots (confusion matrices, ROC and
precision-recall curves, feature importance) and SHAP explainability, all
saved to ``reports/``.
"""
import os
import pickle

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def _ensure_dirs():
    os.makedirs(FIGURES_DIR, exist_ok=True)


def load_test_data():
    """Load the processed test features and target."""
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    X = test.drop(columns=["Class"])
    y = test["Class"]
    return X, y


def evaluate_model(model, X_test, y_test, model_name):
    """Return a metrics dict for one fitted model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_prob),
        "Avg_Precision": average_precision_score(y_test, y_prob),
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def plot_confusion_matrices(results, y_test):
    """Side-by-side confusion matrices for all models."""
    _ensure_dirs()
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{res['Model']}\nF1: {res['F1']:.4f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=150)
    plt.close()


def plot_roc_curves(results, y_test):
    """ROC curves for all models on one plot."""
    _ensure_dirs()
    plt.figure(figsize=(8, 6))
    for res in results:
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        plt.plot(fpr, tpr, label=f"{res['Model']} (AUC={res['ROC_AUC']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close()


def plot_precision_recall_curves(results, y_test):
    """Precision-recall curves for all models on one plot."""
    _ensure_dirs()
    plt.figure(figsize=(8, 6))
    for res in results:
        prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
        plt.plot(rec, prec, label=f"{res['Model']} (AP={res['Avg_Precision']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "precision_recall_curves.png"), dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, top_n=15):
    """Horizontal bar chart of top-N importances (tree-based models)."""
    _ensure_dirs()
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_; skipping.")
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importances[idx])
    plt.yticks(range(len(idx)), [feature_names[i] for i in idx])
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=150)
    plt.close()


def plot_shap_summary(model, X_test, model_name="Best Model", sample_size=1000):
    """SHAP summary plot for a tree-based model on a sample of test rows."""
    _ensure_dirs()
    try:
        import shap
    except ImportError:
        print("shap not installed; skipping SHAP summary.")
        return

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.title(f"SHAP Summary - {model_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIGURES_DIR, "shap_summary.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_comparison_table(results):
    """Persist the model comparison table to CSV and return it."""
    _ensure_dirs()
    df = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ["y_pred", "y_prob"]}
            for r in results
        ]
    )
    df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print("\nModel Comparison:")
    print(df.to_string(index=False))
    return df


def main():
    """Load all models, evaluate, and generate every report artifact."""
    X_test, y_test = load_test_data()
    model_names = ["logistic_regression", "random_forest", "xgboost_model"]

    results = []
    loaded = {}
    for name in model_names:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            print(f"Missing {path}; skipping.")
            continue
        with open(path, "rb") as f:
            model = pickle.load(f)
        loaded[name] = model
        results.append(evaluate_model(model, X_test, y_test, name))

    if not results:
        print("No models found. Run src/train.py first.")
        return

    plot_confusion_matrices(results, y_test)
    plot_roc_curves(results, y_test)
    plot_precision_recall_curves(results, y_test)

    # Feature importance + SHAP from the best available tree model.
    for name in ["xgboost_model", "random_forest"]:
        if name in loaded:
            plot_feature_importance(loaded[name], X_test.columns.tolist())
            plot_shap_summary(loaded[name], X_test, model_name=name)
            break

    comparison = save_comparison_table(results)
    best_row = comparison.loc[comparison["ROC_AUC"].idxmax()]
    best_name = best_row["Model"]
    print(f"\nClassification report - {best_name}:")
    best_res = next(r for r in results if r["Model"] == best_name)
    print(classification_report(y_test, best_res["y_pred"]))


if __name__ == "__main__":
    main()
