
import matplotlib.pyplot as plt
from ..metrics.metrics import roc_pr_curves

def plot_roc(y_true, y_score, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    curves = roc_pr_curves(y_true, y_score)
    ax.plot(curves["fpr"], curves["tpr"], label="ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax

def plot_pr(y_true, y_score, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    curves = roc_pr_curves(y_true, y_score)
    ax.plot(curves["recall"], curves["precision"], label="PR")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    return ax

def save_curves(y_true, y_score, out_prefix: str = "curves"):
    ax = plot_roc(y_true, y_score)
    ax.figure.savefig(f"{out_prefix}_roc.png", dpi=200, bbox_inches="tight")
    plt.close(ax.figure)
    ax = plot_pr(y_true, y_score)
    ax.figure.savefig(f"{out_prefix}_pr.png", dpi=200, bbox_inches="tight")
    plt.close(ax.figure)
