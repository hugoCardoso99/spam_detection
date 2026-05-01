"""
evaluate.py – Compare models on the test set.

Usage:
    python src/evaluate.py --dataset sms
    python src/evaluate.py --dataset sms --from-db
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from db import load_dataset
from preprocess import clean_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAMES = ["LogisticRegression", "RandomForest", "XGBoost"]
CLS = ["Ham", "Spam"]
CLR = {"LogisticRegression": "#6366f1", "RandomForest": "#22c55e", "XGBoost": "#f59e0b"}


def load_test(ds):
    d = os.path.join(ROOT, "data", ds)
    return sp.load_npz(os.path.join(d, "X_test.npz")), joblib.load(os.path.join(d, "y_test.joblib"))


def load_test_from_db(dataset_name, test_size=0.2, random_state=42):
    """Load raw messages from Postgres, vectorize on-demand, and return test split."""
    rows = load_dataset(dataset_name)
    messages = [row["message"] for row in rows]
    labels = [row["label"] for row in rows]

    cleaned = [clean_text(m) for m in messages]

    # load saved vectorizer
    vec_path = os.path.join(ROOT, "models", dataset_name, "tfidf_vectorizer.joblib")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(
            "Vectorizer not found at %s. Run train.py --from-db first." % vec_path
        )
    vectorizer = joblib.load(vec_path)
    X = vectorizer.transform(cleaned)
    y = labels

    # reproduce the same train/test split
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    return X_test, y_test


def eval_one(mdl, X, y):
    yp = mdl.predict(X)
    ypr = mdl.predict_proba(X)[:, 1]
    return dict(accuracy=accuracy_score(y, yp), precision=precision_score(y, yp),
                recall=recall_score(y, yp), f1=f1_score(y, yp),
                roc_auc=roc_auc_score(y, ypr), pr_auc=average_precision_score(y, ypr),
                y_pred=yp, y_proba=ypr)


def best_f1_thr(y, ypr):
    p, r, t = precision_recall_curve(y, ypr)
    p, r = p[:-1], r[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where((p+r) > 0, 2*p*r/(p+r), 0.0)
    i = np.argmax(f)
    return float(t[i]), float(f[i]), (ypr >= t[i]).astype(int)


def plot_cm(res, yt, path):
    n = len(res)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]
    for ax, (nm, m) in zip(axes, res.items()):
        thr, f1v, yp = best_f1_thr(yt, m["y_proba"])
        cm = confusion_matrix(yt, yp)
        ConfusionMatrixDisplay(cm, display_labels=CLS).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title("%s\nThreshold = %.2f  |  F1 = %.4f" % (nm, thr, f1v),
                     fontsize=12, fontweight="bold")
    fig.suptitle("Confusion Matrices (at best-F1 threshold per model)",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % path)


def plot_pr(res, yt, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for nm, m in res.items():
        p, r, _ = precision_recall_curve(yt, m["y_proba"])
        ax.plot(r, p, label="%s  (PR-AUC = %.4f)" % (nm, m["pr_auc"]),
                color=CLR.get(nm), linewidth=2)
    bl = yt.sum() / len(yt)
    ax.axhline(y=bl, color="gray", linestyle="--", linewidth=1,
               label="Baseline (%.2f)" % bl)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=[0, 1], ylim=[0, 1.05])
    ax.set_title("Precision-Recall Curves", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + path)


def plot_f1(res, yt, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    for nm, m in res.items():
        p, r, t = precision_recall_curve(yt, m["y_proba"])
        p, r = p[:-1], r[:-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            fv = np.where((p+r) > 0, 2*p*r/(p+r), 0.0)
        bi = np.argmax(fv)
        ax.plot(t, fv, label="%s  (best F1 = %.4f @ %.2f)" % (nm, fv[bi], t[bi]),
                color=CLR.get(nm), linewidth=2)
        ax.scatter(t[bi], fv[bi], color=CLR.get(nm), s=80, zorder=5,
                   edgecolors="white", linewidths=1.5)
    ax.set(xlabel="Classification Threshold", ylabel="F1-Score",
           xlim=[0, 1], ylim=[0, 1.05])
    ax.set_title("F1-Score Across Thresholds", fontsize=15, fontweight="bold")
    ax.legend(loc="lower center", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: %s" % path)


def load_models(dataset_name):
    """Load all trained models for a given dataset."""
    model_dir = os.path.join(ROOT, "models", dataset_name)
    models = {}
    for name in NAMES:
        path = os.path.join(model_dir, "%s.joblib" % name)
        if os.path.exists(path):
            models[name] = joblib.load(path)
        else:
            raise FileNotFoundError("Model not found: %s" % path)
    return models


def evaluate_all(dataset_name, from_db=False, test_size=0.2, random_state=42):
    """Load test data and models, evaluate, and generate plots."""
    print("\nLoading test data for '%s'..." % dataset_name)
    if from_db:
        X_test, y_test = load_test_from_db(dataset_name, test_size=test_size, random_state=random_state)
    else:
        X_test, y_test = load_test(dataset_name)
    print("  Test set shape: %s" % (X_test.shape,))
    print("  Spam ratio: %.2f%%" % (100 * y_test.mean()))

    print("\nLoading trained models...")
    models = load_models(dataset_name)
    print("  Loaded %d models" % len(models))

    print("\nEvaluating models...")
    results = {}
    for name, model in models.items():
        results[name] = eval_one(model, X_test, y_test)
        print("  %s: accuracy=%.4f, f1=%.4f, roc_auc=%.4f, pr_auc=%.4f" %
              (name, results[name]["accuracy"], results[name]["f1"], results[name]["roc_auc"], results[name]["pr_auc"]))

    # Create output directory
    out_dir = os.path.join(ROOT, "results", dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating plots...")
    plot_cm(results, y_test, os.path.join(out_dir, "confusion_matrix.png"))
    plot_pr(results, y_test, os.path.join(out_dir, "precision_recall.png"))
    plot_f1(results, y_test, os.path.join(out_dir, "f1_threshold.png"))

    # Print classification reports
    print("\nClassification Reports:")
    print("=" * 70)
    for name, res in results.items():
        print("\n%s:" % name)
        print(classification_report(y_test, res["y_pred"], target_names=CLS, digits=4))

    # Save metrics summary
    summary_path = os.path.join(out_dir, "metrics_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Model Evaluation Summary - Dataset: %s\n" % dataset_name)
        f.write("=" * 70 + "\n\n")
        for name, res in results.items():
            f.write("%s:\n" % name)
            f.write("  Accuracy:  %.4f\n" % res["accuracy"])
            f.write("  Precision: %.4f\n" % res["precision"])
            f.write("  Recall:    %.4f\n" % res["recall"])
            f.write("  F1-Score:  %.4f\n" % res["f1"])
            f.write("  ROC-AUC:   %.4f\n" % res["roc_auc"])
            f.write("  PR-AUC:    %.4f\n" % res["pr_auc"])
            f.write("\n")
    print("  Saved: %s" % summary_path)

    print("\nEvaluation complete. Results saved to: %s" % out_dir)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained classification models.")
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name (must match the name used in preprocess.py and train.py).",
    )
    parser.add_argument("--from-db", action="store_true", help="Load raw messages from PostgreSQL instead of preprocessed vectors.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction when using --from-db (default: 0.2).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_all(args.dataset, from_db=args.from_db, test_size=args.test_size)
