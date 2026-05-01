"""
train.py – Train Logistic Regression, Random Forest, and XGBoost on a
preprocessed dataset and save each model to models/<dataset>/.

Usage:
    python src/train.py --dataset sms
    python src/train.py --dataset enron
"""

import argparse
import os
import time

import joblib
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

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


def train_all(dataset_name):
    """Train every model, print timing info, and persist to disk."""
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_all(args.dataset)
    print("\nAll models trained and saved to models/%s/" % args.dataset)
