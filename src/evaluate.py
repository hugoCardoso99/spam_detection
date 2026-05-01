"""
evaluate.py – Load every trained model and compare them on the held-out
test set using accuracy, precision, recall, F1-score, ROC-AUC, and PR-AUC.

Produces three figures saved to figures/:
    1. confusion_matrices.png   — side-by-side confusion matrices
    2. pr_auc_curves.png        — precision-recall curves
    3. f1_vs_threshold.png      — F1-score across classification thresholds
"""

import os

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to file
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── paths ────────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

MODEL_NAMES = ["LogisticRegression", "RandomForest", "XGBoost"]


def load_test_data():
    X_test = sp.load_npz(os.path.join(DATA_DIR, "X_test.npz"))
    y_test = joblib.load(os.path.join(DATA_DIR, "y_test.joblib"))
    return X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Return a dict of metric name -> value for a single model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # PR-AUC (average_precision_score) is more informative than ROC-AUC
    # for imbalanced datasets because it focuses on the minority class
    # and is not inflated by the large number of true negatives.
    pr_auc = average_precision_score(y_test, y_proba)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": pr_auc,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


COLORS = {
    "LogisticRegression": "#6366f1",
    "RandomForest": "#22c55e",
    "XGBoost": "#f59e0b",
}
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")


# ── plotting helpers ─────────────────────────────────────────────────────

def _find_best_f1_threshold(y_test, y_proba):
    """Return (best_threshold, best_f1, y_pred_at_best) for the threshold
    that maximises F1-score."""
    prec, rec, thresholds = precision_recall_curve(y_test, y_proba)
    prec = prec[:-1]
    rec = rec[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_vals = np.where(
            (prec + rec) > 0,
            2 * prec * rec / (prec + rec),
            0.0,
        )
    best_idx = np.argmax(f1_vals)
    best_thr = float(thresholds[best_idx])
    best_f1 = float(f1_vals[best_idx])
    y_pred_best = (y_proba >= best_thr).astype(int)
    return best_thr, best_f1, y_pred_best


def plot_confusion_matrices(results, y_test, save_path):
    """Side-by-side confusion matrices using each model's optimal F1 threshold."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, m) in zip(axes, results.items()):
        best_thr, best_f1, y_pred_best = _find_best_f1_threshold(y_test, m["y_proba"])
        cm = confusion_matrix(y_test, y_pred_best)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Ham", "Spam"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
        ax.set_title(
            "%s\nThreshold = %.2f  |  F1 = %.4f" % (name, best_thr, best_f1),
            fontsize=12, fontweight="bold",
        )

    fig.suptitle(
        "Confusion Matrices (at best-F1 threshold per model)",
        fontsize=15, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + save_path)


def plot_pr_curves(results, y_test, save_path):
    """Overlay precision-recall curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, m in results.items():
        prec, rec, _ = precision_recall_curve(y_test, m["y_proba"])
        pr_auc_val = m["pr_auc"]
        label = "%s  (PR-AUC = %.4f)" % (name, pr_auc_val)
        ax.plot(rec, prec, label=label, color=COLORS.get(name), linewidth=2)

    # baseline: proportion of positives
    baseline = y_test.sum() / len(y_test)
    ax.axhline(
        y=baseline, color="gray", linestyle="--", linewidth=1,
        label="Baseline (%.2f)" % baseline,
    )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=15, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + save_path)


def plot_f1_vs_threshold(results, y_test, save_path):
    """F1-score as a function of the classification threshold for each model."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, m in results.items():
        prec, rec, thresholds = precision_recall_curve(y_test, m["y_proba"])
        # precision_recall_curve returns n+1 precision/recall values;
        # thresholds has length n. Trim to align.
        prec = prec[:-1]
        rec = rec[:-1]

        # F1 = 2 * (P * R) / (P + R), guarding against division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_vals = np.where(
                (prec + rec) > 0,
                2 * prec * rec / (prec + rec),
                0.0,
            )

        # find and mark the optimal threshold
        best_idx = np.argmax(f1_vals)
        best_f1 = f1_vals[best_idx]
        best_thr = thresholds[best_idx]
        label = "%s  (best F1 = %.4f @ %.2f)" % (name, best_f1, best_thr)
        ax.plot(thresholds, f1_vals, label=label, color=COLORS.get(name), linewidth=2)
        ax.scatter(
            best_thr, best_f1,
            color=COLORS.get(name),
            s=80, zorder=5, edgecolors="white", linewidths=1.5,
        )

    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("F1-Score Across Thresholds", fontsize=15, fontweight="bold")
    ax.legend(loc="lower center", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: " + save_path)


# ── main comparison ──────────────────────────────────────────────────────

def compare_models():
    """Evaluate every saved model, print a comparison table, and save plots."""
    X_test, y_test = load_test_data()

    results = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, name + ".joblib")
        if not os.path.exists(path):
            print("Warning: Model not found: %s  — skipping" % path)
            continue
        model = joblib.load(path)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

    # ── summary table ────────────────────────────────────────────────
    header = "%-22s %9s %10s %8s %8s %9s %8s" % (
        "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC",
    )
    print("\n" + "=" * len(header))
    print("MODEL COMPARISON — SMS Spam Detection")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for name, m in results.items():
        print("%-22s %9.4f %10.4f %8.4f %8.4f %9.4f %8.4f" % (
            name, m["accuracy"], m["precision"],
            m["recall"], m["f1"], m["roc_auc"], m["pr_auc"],
        ))
    print("-" * len(header))

    # ── pick the best model by F1 ────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["f1"])
    best = results[best_name]
    print("\nBest model (by F1-score): %s  —  F1 = %.4f" % (best_name, best["f1"]))

    # ── detailed classification reports ──────────────────────────────
    for name, m in results.items():
        print("\n" + "-" * 50)
        print("Classification Report — %s" % name)
        print("-" * 50)
        print(classification_report(y_test, m["y_pred"], target_names=["ham", "spam"]))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, m["y_pred"]))

    # ── generate and save figures ────────────────────────────────────
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print("\nGenerating figures in %s/" % FIGURES_DIR)

    plot_confusion_matrices(
        results, y_test,
        os.path.join(FIGURES_DIR, "confusion_matrices.png"),
    )
    plot_pr_curves(
        results, y_test,
        os.path.join(FIGURES_DIR, "pr_auc_curves.png"),
    )
    plot_f1_vs_threshold(
        results, y_test,
        os.path.join(FIGURES_DIR, "f1_vs_threshold.png"),
    )

    return results


# ── CLI entry point ──────────────────────────────────────────────────────
if __name__ == "__main__":
    compare_models()
