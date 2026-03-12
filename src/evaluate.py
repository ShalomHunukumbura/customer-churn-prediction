from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
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
    roc_curve,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, output_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_model_comparison(all_metrics: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(all_metrics, indent=2))


def save_roc_pr_curves(y_true, y_proba, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr, tpr, label="ROC")
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    ax1.set_title("ROC Curve")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")

    ax2.plot(recall, precision, label="PR")
    ax2.set_title("Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_report_markdown(all_metrics: dict, output_path: Path) -> None:
    lines = ["# Model Comparison", ""]
    for name, metric in all_metrics.items():
        lines.append(f"## {name}")
        lines.append(f"- Accuracy: {metric['accuracy']}")
        lines.append(f"- Precision: {metric['precision']}")
        lines.append(f"- Recall: {metric['recall']}")
        lines.append(f"- F1: {metric['f1']}")
        lines.append(f"- ROC-AUC: {metric['roc_auc']}")
        lines.append(f"- PR-AUC: {metric['pr_auc']}")
        lines.append("")
    output_path.write_text("\n".join(lines))
