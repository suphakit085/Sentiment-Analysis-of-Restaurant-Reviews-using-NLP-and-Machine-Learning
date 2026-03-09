"""
visualization.py
----------------
Generate and save all plots for the sentiment analysis pipeline.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud
from collections import Counter

# ─── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

COLORS = {
    "Positive": "#2ecc71",
    "Neutral":  "#3498db",
    "Negative": "#e74c3c",
}

WORDCLOUD_CM = {
    "Positive": "Greens",
    "Neutral":  "Blues",
    "Negative": "Reds",
}

MODEL_COLORS = ["#5e81f4", "#56ccf2", "#f4a261"]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def _save(fig, filename: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Visualization] Saved: {path}")
    return path


# ── 1. Sentiment Distribution ──────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame, sentiment_col: str = "sentiment"):
    """Bar chart showing count & percentage of each sentiment class."""
    counts = df[sentiment_col].value_counts()
    total = counts.sum()
    labels = ["Positive", "Neutral", "Negative"]
    values = [counts.get(l, 0) for l in labels]
    colors = [COLORS[l] for l in labels]
    pcts = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=2)
    ax.set_title("Sentiment Distribution of Reviews", fontweight="bold", pad=15)
    ax.set_ylabel("Number of Reviews")

    for bar, pct, val in zip(bars, pcts, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, max(values) * 1.2)
    fig.tight_layout()
    return _save(fig, "sentiment_distribution.png")


# ── 2. Model Comparison ────────────────────────────────────────────────────────

def plot_model_comparison(comparison_df: pd.DataFrame):
    """Grouped bar chart comparing Accuracy / Precision / Recall / F1 across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = comparison_df.index.tolist()
    x = np.arange(len(metrics))
    width = 0.22
    n = len(models)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_facecolor("#f8f9fa")
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=2)

    for i, (model, color) in enumerate(zip(models, MODEL_COLORS)):
        vals = [comparison_df.loc[model, m] for m in metrics]
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=color,
                      edgecolor="white", linewidth=0.5, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_title("Model Performance Comparison", fontweight="bold", pad=15)
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    return _save(fig, "model_comparison.png")


# ── 3. Confusion Matrices ──────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """Heatmap confusion matrix for a single model."""
    labels = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, data, fmt, title_suffix in zip(
        axes,
        [cm, cm_norm],
        [".0f", ".2f"],
        ["(Counts)", "(Normalized)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, linecolor="#e0e0e0",
            ax=ax, cbar_kws={"shrink": 0.8},
        )
        ax.set_title(f"{model_name}\nConfusion Matrix {title_suffix}", fontweight="bold")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")

    fig.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    return _save(fig, f"confusion_matrix_{safe_name}.png")


def plot_all_confusion_matrices(results: dict):
    """Plot confusion matrix for every model in results dict."""
    for model_name, res in results.items():
        plot_confusion_matrix(res["y_true"], res["y_pred"], model_name)


# ── 4. Word Cloud per Sentiment ────────────────────────────────────────────────

def plot_wordclouds(df: pd.DataFrame,
                    text_col: str = "cleaned_text",
                    sentiment_col: str = "sentiment"):
    """Generate a side-by-side word cloud for each sentiment class."""
    labels = ["Positive", "Neutral", "Negative"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Most Common Words by Sentiment", fontsize=16, fontweight="bold", y=1.02)

    for ax, label in zip(axes, labels):
        subset = df[df[sentiment_col] == label][text_col]
        text = " ".join(subset.dropna().tolist())

        if not text.strip():
            ax.set_title(label, fontsize=13, fontweight="bold",
                         color=COLORS[label])
            ax.axis("off")
            continue

        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap=WORDCLOUD_CM[label],
            max_words=100,
            collocations=False,
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(label, fontsize=13, fontweight="bold",
                     color=COLORS[label], pad=10)
        ax.axis("off")

    fig.tight_layout()
    return _save(fig, "wordclouds.png")


# ── 5. Per-Class F1 Score ──────────────────────────────────────────────────────

def plot_per_class_f1(results: dict):
    """
    Horizontal bar chart showing per-class F1 score for each model.
    """
    from sklearn.metrics import f1_score

    classes = ["Negative", "Neutral", "Positive"]
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, res), color in zip(axes, results.items(), MODEL_COLORS):
        y_true = res["y_true"]
        y_pred = res["y_pred"]
        f1s = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

        bar_colors = [COLORS[c] for c in classes]
        bars = ax.barh(classes, f1s, color=bar_colors, edgecolor="white",
                       linewidth=0.5, zorder=3)
        ax.set_facecolor("#f8f9fa")
        ax.grid(axis="x", color="white", linewidth=1.2, zorder=2)
        ax.set_xlim(0, 1.05)
        ax.set_title(model_name, fontweight="bold", pad=10, color=color)
        ax.set_xlabel("F1-Score")

        for bar, val in zip(bars, f1s):
            ax.text(
                val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold",
            )

    fig.suptitle("Per-Class F1 Score by Model", fontsize=15, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "per_class_f1.png")
