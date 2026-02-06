from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False


def _plot_curve(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, out_png: Path):
    if not _HAS_MPL:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)


def _plot_reliability(bins_info: List[Dict[str, Any]], out_png: Path):
    if not _HAS_MPL:
        return
    xs = []
    ys = []
    for b in bins_info:
        if b.get("conf") is None or b.get("acc") is None:
            continue
        xs.append(float(b["conf"]))
        ys.append(float(b["acc"]))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.scatter(xs, ys, s=30)
    ax.set_title("Calibration (Reliability Diagram)")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", linewidth=0.5)
    fig.savefig(str(out_png), bbox_inches="tight", dpi=150)
    plt.close(fig)
