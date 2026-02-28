"""
Evaluation utilities — confusion matrix, classification report, ROC-AUC.

Usage
-----
    python -m pneumonia_detection.evaluate
"""

import os

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from pneumonia_detection import config
from pneumonia_detection.preprocess import get_test_generator


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model: tf.keras.Model) -> dict:
    """
    Run the model over the test set and return a dictionary of metrics.

    Returns
    -------
    dict with keys: accuracy, auc, precision, recall, f1, report
    """
    test_gen = get_test_generator()
    test_gen.reset()

    y_prob = model.predict(test_gen, verbose=config.VERBOSE).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    y_true = test_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=config.CLASSES, digits=4
    )
    auc = roc_auc_score(y_true, y_prob)

    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

    print("\n" + "=" * 60)
    print("Test-set evaluation")
    print("=" * 60)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print("\nClassification Report:\n")
    print(report)

    return dict(
        accuracy=accuracy,
        auc=auc,
        precision=precision,
        recall=recall,
        f1=f1,
        report=report,
        y_true=y_true,
        y_prob=y_prob,
        confusion_matrix=cm,
    )


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, save_path: str | None = None):
    """Plot and optionally save a confusion matrix heat-map."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    tick_marks = np.arange(len(config.CLASSES))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(config.CLASSES, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(config.CLASSES)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved to {save_path}")
    return fig


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str | None = None):
    """Plot and optionally save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"ROC curve saved to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = tf.keras.models.load_model(config.MODEL_PATH)
    results = evaluate_model(model)
    plot_confusion_matrix(
        results["confusion_matrix"],
        save_path=os.path.join(config.MODEL_DIR, "confusion_matrix.png"),
    )
    plot_roc_curve(
        results["y_true"],
        results["y_prob"],
        save_path=os.path.join(config.MODEL_DIR, "roc_curve.png"),
    )
