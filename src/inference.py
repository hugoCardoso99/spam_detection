"""
inference.py – Load a trained model + TF-IDF vectorizer and predict
whether new text samples belong to the positive or negative class.

Usage:
    python src/inference.py --dataset sms "Congratulations! You've won a free iPhone"
    python src/inference.py --dataset sms "Hey, are we still meeting at 5?"
    python src/inference.py --dataset enron --model LogisticRegression "urgent wire transfer"
"""

import argparse
import os

import joblib

from preprocess import clean_text

# ── paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT_DIR, "models")

DEFAULT_MODEL = "RandomForest"


def load_pipeline(dataset_name, model_name=None):
    """Load the TF-IDF vectorizer and a trained model for a given dataset."""
    model_name = model_name or DEFAULT_MODEL
    model_dir = os.path.join(MODELS_DIR, dataset_name)
    vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(model_dir, "%s.joblib" % model_name)

    if not os.path.exists(vec_path):
        raise FileNotFoundError("Vectorizer not found at %s. Run preprocess.py first." % vec_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found at %s. Run train.py first." % model_path)

    vectorizer = joblib.load(vec_path)
    model = joblib.load(model_path)
    print("Loaded model: %s (dataset: %s)" % (model_name, dataset_name))
    return vectorizer, model


def predict(messages, vectorizer=None, model=None, dataset_name=None, model_name=None):
    """Predict positive/negative for a list of raw text messages.

    Returns a list of dicts with keys: message, prediction, confidence.
    """
    if vectorizer is None or model is None:
        vectorizer, model = load_pipeline(dataset_name, model_name)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on new text samples.")
    parser.add_argument("--dataset", required=True, help="Dataset name (must match preprocess/train).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model to use (default: %s)." % DEFAULT_MODEL)
    parser.add_argument("messages", nargs="+", help="One or more text messages to classify.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vectorizer, model = load_pipeline(args.dataset, args.model)

    print("\n" + "=" * 60)
    print("TEXT CLASSIFICATION — Inference (%s)" % args.dataset)
    print("=" * 60)

    for r in predict(args.messages, vectorizer, model):
        print("\n  [%s]  (confidence: %.1f%%)" % (r["prediction"], r["confidence"] * 100))
        print("   \"%s\"" % r["message"])
