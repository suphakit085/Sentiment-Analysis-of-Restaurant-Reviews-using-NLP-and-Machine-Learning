"""
feature_extraction.py
---------------------
TF-IDF vectorization for text features.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import joblib
import os


def build_tfidf_vectorizer(max_features: int = 10000, ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    """
    Create a TF-IDF vectorizer instance with the given settings.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,        # Apply log normalization to term frequencies
        min_df=2,                 # Ignore terms appearing in fewer than 2 docs
        max_df=0.95,              # Ignore terms appearing in more than 95% of docs
        strip_accents="unicode",
        analyzer="word",
    )


def fit_transform_train(
    vectorizer: TfidfVectorizer, X_train: list
) -> csr_matrix:
    """Fit the vectorizer on training data and transform it."""
    print("[FeatureExtraction] Fitting TF-IDF on training data...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"[FeatureExtraction] Training matrix shape: {X_train_tfidf.shape}\n")
    return X_train_tfidf


def transform_test(vectorizer: TfidfVectorizer, X_test: list) -> csr_matrix:
    """Transform test data using the already-fitted vectorizer."""
    print("[FeatureExtraction] Transforming test data with TF-IDF...")
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"[FeatureExtraction] Test matrix shape: {X_test_tfidf.shape}\n")
    return X_test_tfidf


def save_vectorizer(vectorizer: TfidfVectorizer, path: str):
    """Persist the fitted vectorizer to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"[FeatureExtraction] Vectorizer saved to: {path}")


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load a previously saved vectorizer."""
    return joblib.load(path)
