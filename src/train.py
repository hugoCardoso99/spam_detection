"""
train.py – Train Logistic Regression, Random Forest, and XGBoost on the
preprocessed SMS Spam Collection data and save each model to models/.
"""

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


def load_splits():
    """Load the preprocessed train/test splits produced by preprocess.py."""
    X_train = sp.load_npz(os.path.join(DATA_DIR, "X_train.npz"))
    X_test = sp.load_npz(os.path.join(DATA_DIR, "X_test.npz"))
    y_train = joblib.load(os.path.join(DATA_DIR, "y_train.joblib"))
    y_test = joblib.load(os.path.join(DATA_DIR, "y_test.joblib"))
    return X_train, X_test, y_train, y_test


def get_models() -> dict:
    """Return a dict of model name → untrained estimator."""
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


def train_all():
    """Train every model, print timing info, and persist to disk."""
    X_train, X_test, y_train, y_test = load_splits()
    os.makedirs(MODELS_DIR, exist_ok=True)

    trained = {}
    for name, model in get_models().items():
        print(f"\n{'='*50}")
        print(f"Training {name} …")
        start = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - start
        print(f"  ✓ Trained in {elapsed:.2f}s")

        path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, path)
        print(f"  ✓ Saved → {path}")
        trained[name] = model

    return trained


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    train_all()
    print("\nAll models trained and saved to models/")
