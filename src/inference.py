"""
inference.py – Load the best trained model + TF-IDF vectorizer and predict
whether new SMS messages are spam or ham.

Usage:
    python src/inference.py "Congratulations! You've won a free iPhone"
    python src/inference.py "Hey, are we still meeting at 5?"
"""

import os
import sys

import joblib

from preprocess import clean_text

# ── paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Default to Random Forest (best PR-AUC and F1 on this dataset).
DEFAULT_MODEL = "RandomForest"


def load_pipeline(model_name: str | None = None):
    """Load the TF-IDF vectorizer and a trained model."""
    model_name = model_name or DEFAULT_MODEL
    vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")

    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer not found at {vec_path}. Run preprocess.py first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    print(f"Loaded model: {model_name}")
    return vectorizer, model


def predict(messages: list[str], vectorizer=None, model=None, model_name: str | None = None):
    """
    Predict spam/ham for a list of raw SMS messages.

    Returns a list of dicts with keys: message, prediction, confidence.
    """
    if vectorizer is None or model is None:
        vectorizer, model = load_pipeline(model_name)

    cleaned = [clean_text(m) for m in messages]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]

    results = []
    for msg, pred, prob in zip(messages, preds, probas):
        label = "SPAM" if pred == 1 else "HAM"
        results.append({
            "message": msg,
            "prediction": label,
            "confidence": round(float(prob if pred == 1 else 1 - prob), 4),
        })
    return results


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py \"<message>\" [\"<message2>\" ...]")
        sys.exit(1)

    messages = sys.argv[1:]
    vectorizer, model = load_pipeline()

    print(f"\n{'='*60}")
    print("SPAM DETECTION — Inference")
    print("=" * 60)

    for r in predict(messages, vectorizer, model):
        icon = "🚫" if r["prediction"] == "SPAM" else "✅"
        print(f"\n{icon}  [{r['prediction']}]  (confidence: {r['confidence']:.1%})")
        print(f"   \"{r['message']}\"")
