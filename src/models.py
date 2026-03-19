"""
models.py
---------
Train and return three ML classifiers for sentiment analysis:
  1. Naive Bayes (MultinomialNB)
  2. Logistic Regression
  3. Support Vector Machine (LinearSVC)
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os


def train_all_models(X_train, y_train) -> dict:
    """
    Train all three models and return them in a dict.

    Parameters
    ----------
    X_train : Sparse TF-IDF matrix (training)
    y_train : Target labels (training)

    Returns
    -------
    dict mapping model name → fitted estimator
    """
    models = {}

    # ── 1. Naive Bayes ────────────────────────────────────────────────────────
    print("[Models] Training Naive Bayes...")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, y_train)
    models["Naive Bayes"] = nb
    print("[Models] Naive Bayes trained.\n")

    # ── 2. Logistic Regression ────────────────────────────────────────────────
    print("[Models] Training Logistic Regression...")
    lr = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    models["Logistic Regression"] = lr
    print("[Models] Logistic Regression trained.\n")

    # ── 3. SVM (LinearSVC) ────────────────────────────────────────────────────
    print("[Models] Training SVM (LinearSVC)...")
    svc = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    # Wrap with CalibratedClassifierCV so predict_proba is available
    svm = CalibratedClassifierCV(svc, cv=3)
    svm.fit(X_train, y_train)
    models["SVM"] = svm
    print("[Models] SVM trained.\n")

    return models


def save_model(model, path: str):
    """Persist a trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[Models] Model saved to: {path}")


def load_model(path: str):
    """Load a previously saved model."""
    return joblib.load(path)
