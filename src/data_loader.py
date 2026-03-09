"""
data_loader.py
--------------
Load and prepare the Yelp restaurant review dataset.
"""

import pandas as pd
import os


def load_data(filepath: str, sample_size: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Load the CSV dataset, keep relevant columns, drop missing text rows,
    and return a stratified sample.

    Parameters
    ----------
    filepath    : Path to the raw CSV file.
    sample_size : Number of reviews to sample (default 10,000).
    random_state: Reproducibility seed.

    Returns
    -------
    DataFrame with columns ['text', 'sentiment']
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    print(f"[DataLoader] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath, on_bad_lines="skip")

    # Rename for clarity
    df = df.rename(columns={"text": "text", "rating_review": "sentiment"})

    # Keep only needed columns
    df = df[["text", "sentiment"]].dropna()

    # Remove any rows with empty text
    df = df[df["text"].str.strip().str.len() > 0]

    # Validate sentiment labels
    valid_labels = {"Positive", "Neutral", "Negative"}
    df = df[df["sentiment"].isin(valid_labels)]

    print(f"[DataLoader] Total usable reviews: {len(df)}")
    print(f"[DataLoader] Sentiment distribution (full):\n{df['sentiment'].value_counts()}\n")

    # Stratified sampling
    actual_sample = min(sample_size, len(df))
    df_sampled = df.groupby("sentiment", group_keys=False).apply(
        lambda x: x.sample(
            frac=actual_sample / len(df),
            random_state=random_state
        )
    ).reset_index(drop=True)

    # Ensure exactly sample_size rows (rounding artefact fix)
    df_sampled = df_sampled.sample(
        n=min(actual_sample, len(df_sampled)),
        random_state=random_state
    ).reset_index(drop=True)

    print(f"[DataLoader] Sampled {len(df_sampled)} reviews.")
    print(f"[DataLoader] Sampled distribution:\n{df_sampled['sentiment'].value_counts()}\n")

    return df_sampled
