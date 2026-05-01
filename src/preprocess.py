"""
preprocess.py – Load, clean, and featurise the SMS Spam Collection dataset.

Pipeline:
    1. Download the dataset (if not already cached in data/).
    2. Clean text (lowercase, strip punctuation/numbers, remove stopwords).
    3. Vectorise with TF-IDF.
    4. Split into train / test sets and persist artefacts for downstream use.
"""

import os
import re
import string
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)
RAW_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
ZIP_FILE = os.path.join(DATA_DIR, "smsspamcollection.zip")


# ── download ─────────────────────────────────────────────────────────────
def download_dataset() -> str:
    """Download the SMS Spam Collection from UCI if it isn't already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(RAW_FILE):
        print(f"Dataset already exists at {RAW_FILE}")
        return RAW_FILE

    print("Downloading SMS Spam Collection dataset …")
    urllib.request.urlretrieve(DATASET_URL, ZIP_FILE)
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        zf.extractall(DATA_DIR)
    print(f"Dataset extracted to {DATA_DIR}")
    return RAW_FILE


# ── load ─────────────────────────────────────────────────────────────────
def load_data(path: str | None = None) -> pd.DataFrame:
    """Read the tab-separated SMS file into a DataFrame."""
    path = path or RAW_FILE
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="latin-1",
    )
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    print(f"Loaded {len(df)} messages  (spam={df['label'].sum()}, ham={len(df) - df['label'].sum()})")
    return df


# ── clean ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip URLs, punctuation, digits, and collapse whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)          # URLs
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # punctuation
    text = re.sub(r"\d+", "", text)                         # digits
    text = re.sub(r"\s+", " ", text).strip()                # whitespace
    return text


# ── featurise ────────────────────────────────────────────────────────────
def build_features(
    df: pd.DataFrame,
    max_features: int = 5000,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Clean text → TF-IDF → train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test, vectorizer
    """
    df = df.copy()
    df["clean"] = df["message"].apply(clean_text)

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
    )

    X = vectorizer.fit_transform(df["clean"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # persist the vectorizer so inference can reuse it
    os.makedirs(MODELS_DIR, exist_ok=True)
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print(f"TF-IDF vectorizer saved → {vec_path}")

    print(f"Train set: {X_train.shape[0]} samples  |  Test set: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test, vectorizer


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_path = download_dataset()
    df = load_data(raw_path)
    X_train, X_test, y_train, y_test, vec = build_features(df)

    # save processed splits for train.py / evaluate.py
    import scipy.sparse as sp

    sp.save_npz(os.path.join(DATA_DIR, "X_train.npz"), X_train)
    sp.save_npz(os.path.join(DATA_DIR, "X_test.npz"), X_test)
    joblib.dump(y_train, os.path.join(DATA_DIR, "y_train.joblib"))
    joblib.dump(y_test, os.path.join(DATA_DIR, "y_test.joblib"))
    print("Preprocessed data saved to data/")
