"""
train.py – Train Logistic Regression, Random Forest, and XGBoost on a
preprocessed dataset and save each model to models/<dataset>/.

Usage:
    python src/train.py --dataset sms
    python src/train.py --dataset enron
    python src/train.py --dataset sms --from-db
"""

import argparse
import os
import time

import joblib
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from db import load_dataset
from preprocess import clean_text

# ── paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def load_splits(dataset_name):
    """Load the preprocessed train/test splits produced by preprocess.py."""
    data_dir = os.path.join(DATA_DIR, dataset_name)
    X_train = sp.load_npz(os.path.join(data_dir, "X_train.npz"))
    X_test = sp.load_npz(os.path.join(data_dir, "X_test.npz"))
    y_train = joblib.load(os.path.join(data_dir, "y_train.joblib"))
    y_test = joblib.load(os.path.join(data_dir, "y_test.joblib"))
    return X_train, X_test, y_train, y_test


def load_splits_from_db(dataset_name, max_features=5000, test_size=0.2, random_state=42):
    """Load raw messages from Postgres, vectorize on-demand, and split."""
    rows = load_dataset(dataset_name)
    messages = [row["message"] for row in rows]
    labels = [row["label"] for row in rows]

    cleaned = [clean_text(m) for m in messages]
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(cleaned)
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # persist vectorizer for later use
    model_dir = os.path.join(MODELS_DIR, dataset_name)
    os.makedirs(model_dir, exist_ok=True)
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    joblib.dump(vectorizer, vec_path)
    print("TF-IDF vectorizer saved: %s" % vec_path)

    return X_train, X_test, y_train, y_test


def get_models():
    """Return a dict of model name -> untrained estimator."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="liblinear",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        ),
    }


def train_all(dataset_name, from_db=False, max_features=5000, test_size=0.2):
    """Train every model, print timing info, and persist to disk."""
    if from_db:
        X_train, X_test, y_train, y_test = load_splits_from_db(
            dataset_name, max_features=max_features, test_size=test_size
        )
    else:
        X_train, X_test, y_train, y_test = load_splits(dataset_name)
    model_dir = os.path.join(MODELS_DIR, dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    trained = {}
    for name, model in get_models().items():
        print("\n" + "=" * 50)
        print("Training %s ..." % name)
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print("  Trained in %.2fs" % elapsed)

        path = os.path.join(model_dir, "%s.joblib" % name)
        joblib.dump(model, path)
        print("  Saved: %s" % path)
        trained[name] = model

    return trained


# ── CLI entry point ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train classification models.")
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (must match the name used in preprocess.py).",
    )
    parser.add_argument("--from-db", action="store_true", help="Load raw messages from PostgreSQL instead of preprocessed vectors.")
    parser.add_argument("--max-features", type=int, default=5000, help="Max TF-IDF features when using --from-db (default: 5000).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction when using --from-db (default: 0.2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_all(args.dataset, from_db=args.from_db, max_features=args.max_features, test_size=args.test_size)
    print("\nAll models trained and saved to models/%s/" % args.dataset)
