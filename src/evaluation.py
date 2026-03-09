"""
evaluation.py
-------------
Evaluate trained models and return comparison metrics.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Generate predictions and compute Accuracy, Precision, Recall, F1.

    Returns a dict with all metrics + the predictions for visualization.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\n{'='*60}")
    print(f" Model: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"\nClassification Report:\n{report}")

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "y_pred": y_pred,
        "report": report,
    }


def evaluate_all_models(models: dict, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate all models and return:
      - comparison_df : DataFrame with one row per model
      - results       : dict of per-model result dicts
    """
    results = {}
    rows = []

    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results[name] = result
        rows.append({
            "Model": name,
            "Accuracy": round(result["accuracy"], 4),
            "Precision": round(result["precision"], 4),
            "Recall": round(result["recall"], 4),
            "F1-Score": round(result["f1_score"], 4),
        })

    comparison_df = pd.DataFrame(rows).set_index("Model")

    print(f"\n{'='*60}")
    print(" MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comparison_df.to_string())
    print()

    return comparison_df, results
