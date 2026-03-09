"""
main.py
-------
End-to-end NLP Sentiment Analysis Pipeline for Yelp Restaurant Reviews

Usage:
    python main.py
"""

import os
import sys
import time
import warnings

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "yelp_sentiment_master_dataset.csv")
OUT_DIR   = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Add src to Python path
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from sklearn.model_selection import train_test_split

from src.data_loader       import load_data
from src.preprocessing     import preprocess_dataframe
from src.feature_extraction import (
    build_tfidf_vectorizer,
    fit_transform_train,
    transform_test,
)
from src.models         import train_all_models
from src.evaluation     import evaluate_all_models
from src.visualization  import (
    plot_sentiment_distribution,
    plot_model_comparison,
    plot_all_confusion_matrices,
    plot_wordclouds,
    plot_per_class_f1,
)


def print_header(title: str):
    print(f"\n{'#'*65}")
    print(f"#  {title}")
    print(f"{'#'*65}\n")


def main():
    start_time = time.time()

    # ── Step 1: Load Data ──────────────────────────────────────────────────────
    print_header("Step 1: Load Data")
    df = load_data(DATA_PATH, sample_size=10000, random_state=42)

    # ── Step 2: Visualize Raw Distribution ────────────────────────────────────
    print_header("Step 2: Sentiment Distribution")
    dist_path = plot_sentiment_distribution(df)
    print(f"  Chart saved → {dist_path}")

    # ── Step 3: Preprocess Text ───────────────────────────────────────────────
    print_header("Step 3: Text Preprocessing")
    df = preprocess_dataframe(df, text_col="text")

    # ── Step 4: Train / Test Split ────────────────────────────────────────────
    print_header("Step 4: Train / Test Split (80 / 20)")
    X = df["cleaned_text"].tolist()
    y = df["sentiment"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")

    from collections import Counter
    print(f"\n  Train class distribution: {dict(Counter(y_train))}")
    print(f"  Test  class distribution: {dict(Counter(y_test))}")

    # ── Step 5: TF-IDF Feature Extraction ────────────────────────────────────
    print_header("Step 5: TF-IDF Feature Extraction")
    vectorizer = build_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = fit_transform_train(vectorizer, X_train)
    X_test_tfidf  = transform_test(vectorizer, X_test)

    # ── Step 6: Train Models ──────────────────────────────────────────────────
    print_header("Step 6: Train ML Models")
    models = train_all_models(X_train_tfidf, y_train)

    # ── Step 7: Evaluate Models ───────────────────────────────────────────────
    print_header("Step 7: Model Evaluation")
    comparison_df, results = evaluate_all_models(models, X_test_tfidf, y_test)

    # Attach y_true so visualization can use it
    for name in results:
        results[name]["y_true"] = y_test

    # ── Step 8: Visualize Results ─────────────────────────────────────────────
    print_header("Step 8: Save Result Charts")
    cmp_path = plot_model_comparison(comparison_df)
    print(f"  Model comparison chart → {cmp_path}")
    plot_all_confusion_matrices(results)

    # Word cloud for each sentiment class
    wc_path = plot_wordclouds(df)
    print(f"  Word clouds          → {wc_path}")

    # Per-class F1 breakdown
    f1_path = plot_per_class_f1(results)
    print(f"  Per-class F1 chart   → {f1_path}")

    # ── Step 9: Final Summary ─────────────────────────────────────────────────
    elapsed = time.time() - start_time
    best_model = comparison_df["F1-Score"].idxmax()
    best_f1    = comparison_df.loc[best_model, "F1-Score"]

    print_header("Step 9: Final Summary")
    print(f"  Dataset size       : {len(df):,} reviews")
    print(f"  Best Model (F1)    : {best_model} — F1={best_f1:.4f}")
    print(f"  Output directory   : {OUT_DIR}")
    print(f"  Total time elapsed : {elapsed:.1f} seconds")
    print("\n  Saved charts:")
    for fname in os.listdir(OUT_DIR):
        if fname.endswith(".png"):
            print(f"    • {fname}")

    print(f"\n{'='*65}")
    print(" Pipeline completed successfully!")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
