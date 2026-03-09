"""
preprocessing.py
----------------
Text cleaning and NLP preprocessing pipeline.
"""

import re
import string
import nltk
import pandas as pd

# Download required NLTK resources (only on first run)
def _ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"[Preprocessing] Downloading NLTK resource: {name}")
            nltk.download(name, quiet=True)


_ensure_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Keep negation words as they carry sentiment signal
KEEP_WORDS = {"not", "no", "never", "nor", "neither", "without"}
STOPWORDS = STOPWORDS - KEEP_WORDS


def clean_text(text: str) -> str:
    """
    Apply full NLP preprocessing to a single review string:
    1. Lowercase
    2. Remove URLs
    3. Remove HTML tags
    4. Remove non-alphabetic characters (keep spaces)
    5. Tokenize
    6. Remove stopwords
    7. Lemmatize
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and short tokens, then lemmatize
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and len(tok) > 1
    ]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Apply clean_text to every row of the given DataFrame column.
    Adds a 'cleaned_text' column and drops rows where result is empty.
    """
    print("[Preprocessing] Cleaning text...")
    df = df.copy()
    df["cleaned_text"] = df[text_col].apply(clean_text)

    before = len(df)
    df = df[df["cleaned_text"].str.strip().str.len() > 0].reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"[Preprocessing] Dropped {before - after} empty rows after cleaning.")

    print(f"[Preprocessing] Done. {after} reviews ready.\n")
    return df
