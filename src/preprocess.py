"""
preprocess.py – Load, clean, and featurise a spam/ham classification dataset.

Expects any CSV with at least two columns:
    - "Message"  — the text to classify
    - "Spam/Ham" — the label ("spam" or "ham", case-insensitive)

Preprocessed artefacts are saved under data/<dataset_name>/ so multiple
datasets can coexist.

Usage:
    # SMS Spam Collection (auto-downloaded)
    python src/preprocess.py --dataset sms

    # Custom CSV
    python src/preprocess.py --dataset enron --csv data/enron_labeled.csv
"""

import argparse
import os
import re
import string
import urllib.request
import zipfile

import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# ── paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

SMS_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)


# ── dataset loaders ──────────────────────────────────────────────────────

def download_sms_dataset():
    """Download the SMS Spam Collection from UCI and return a standardised DataFrame."""
    sms_dir = os.path.join(DATA_DIR, "sms")
    raw_file = os.path.join(sms_dir, "SMSSpamCollection")
    zip_file = os.path.join(sms_dir, "smsspamcollection.zip")
    os.makedirs(sms_dir, exist_ok=True)

    if not os.path.exists(raw_file):
        print("Downloading SMS Spam Collection dataset ...")
        urllib.request.urlretrieve(SMS_URL, zip_file)
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(sms_dir)
        print("Extracted to %s" % sms_dir)

    df = pd.read_csv(
        raw_file, sep="\t", header=None,
        names=["Spam/Ham", "Message"], encoding="latin-1",
    )
    return df


def load_csv_dataset(csv_path):
    """Load a CSV that contains 'Message' and 'Spam/Ham' columns."""
    df = pd.read_csv(csv_path)

    # case-insensitive column matching
    col_map = {c.lower(): c for c in df.columns}
    msg_col = col_map.get("message")
    label_col = col_map.get("spam/ham")

    if msg_col is None:
        raise ValueError("'Message' column not found. Available: %s" % list(df.columns))
    if label_col is None:
        raise ValueError("'Spam/Ham' column not found. Available: %s" % list(df.columns))

    df = df.rename(columns={msg_col: "Message", label_col: "Spam/Ham"})
    return df


def prepare_labels(df):
    """Standardise the DataFrame: keep Message + binary label (spam=1, ham=0)."""
    df = df[["Message", "Spam/Ham"]].copy()
    df = df.dropna(subset=["Message", "Spam/Ham"])
    df["label"] = df["Spam/Ham"].astype(str).str.strip().str.lower().map({"spam": 1, "ham": 0})

    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        unique_vals = df.loc[df["label"].isna(), "Spam/Ham"].unique()[:5]
        raise ValueError(
            "%d rows have unrecognised labels (expected 'spam' or 'ham'). "
            "Sample values: %s" % (unmapped, list(unique_vals))
        )

    df["label"] = df["label"].astype(int)
    return df


# ── text cleaning ────────────────────────────────────────────────────────

def clean_text(text):
    """Lowercase, strip URLs, punctuation, digits, and collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── feature engineering ──────────────────────────────────────────────────

def build_features(df, dataset_name, max_features=5000, test_size=0.2, random_state=42):
    """Clean text -> TF-IDF -> train/test split.

    Returns: X_train, X_test, y_train, y_test, vectorizer
    """
    df = df.copy()
    df["clean"] = df["Message"].apply(clean_text)

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

    # persist vectorizer
    model_dir = os.path.join(MODELS_DIR, dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print("TF-IDF vectorizer saved: %s" % vec_path)

    print("Train set: %d samples  |  Test set: %d samples" % (X_train.shape[0], X_test.shape[0]))
    return X_train, X_test, y_train, y_test, vectorizer


# ── CLI entry point ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess a spam/ham classification dataset.")
    parser.add_argument(
        "--dataset", required=True,
        help="Name for this dataset (used to namespace saved artefacts). "
             "Use 'sms' to auto-download the SMS Spam Collection.",
    )
    parser.add_argument("--csv", default=None, help="Path to a CSV file (required for custom datasets).")
    parser.add_argument("--max-features", type=int, default=5000, help="Max TF-IDF features (default: 5000).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # load data
    if args.dataset == "sms":
        print("Using built-in SMS Spam Collection dataset")
        df = download_sms_dataset()
    else:
        if args.csv is None:
            raise ValueError("--csv is required for custom datasets (dataset != 'sms').")
        print("Loading dataset '%s' from %s" % (args.dataset, args.csv))
        df = load_csv_dataset(args.csv)

    df = prepare_labels(df)

    n_spam = df["label"].sum()
    n_ham = len(df) - n_spam
    print("Loaded %d samples  (spam=%d, ham=%d, %.1f%% spam)" % (
        len(df), n_spam, n_ham, 100 * n_spam / len(df),
    ))

    # build features
    X_train, X_test, y_train, y_test, vec = build_features(
        df, args.dataset,
        max_features=args.max_features,
        test_size=args.test_size,
    )

    # save processed splits
    data_dir = os.path.join(DATA_DIR, args.dataset)
    os.makedirs(data_dir, exist_ok=True)
    sp.save_npz(os.path.join(data_dir, "X_train.npz"), X_train)
    sp.save_npz(os.path.join(data_dir, "X_test.npz"), X_test)
    joblib.dump(y_train, os.path.join(data_dir, "y_train.joblib"))
    joblib.dump(y_test, os.path.join(data_dir, "y_test.joblib"))
    print("Preprocessed data saved to data/%s/" % args.dataset)
