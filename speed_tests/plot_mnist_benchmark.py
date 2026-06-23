"""Generate plots from the MNIST SecAgg+ benchmark JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = Path("speed_tests/flower_secagg_mnist_benchmark.json")
OUT_DIR = Path("speed_tests")


def _load_data() -> list[dict[str, Any]]:
    """Load benchmark results from JSON."""
    data: list[dict[str, Any]] = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    return data


def _group(
    data: list[dict[str, Any]],
) -> dict[tuple[int, float], dict[str, dict[str, Any]]]:
    """Group results by (n, dropout_rate) and implementation."""
    grouped: dict[tuple[int, float], dict[str, dict[str, Any]]] = {}
    for row in data:
        key = (row["n"], row["target_dropout_rate"])
        grouped.setdefault(key, {})[row["impl"]] = row
    return grouped


def _plot_metric(
    grouped: dict[tuple[int, float], dict[str, dict[str, Any]]],
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    log_y: bool = False,
) -> None:
    """Create a 2x2 subplot grid, one panel per dropout rate."""
    rates = sorted({key[1] for key in grouped})
    ns = sorted({key[0] for key in grouped})
    impls = ("classical", "pq_single")
    colors = {"classical": "tab:blue", "pq_single": "tab:orange"}
    markers = {"classical": "o", "pq_single": "s"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, rate in zip(axes, rates, strict=True):
        for impl in impls:
            xs = []
            ys = []
            for n in ns:
                row = grouped[(n, rate)].get(impl)
                if row and row["status"] == "success":
                    xs.append(n)
                    ys.append(row[metric])
            ax.plot(
                xs,
                ys,
                label=impl,
                color=colors[impl],
                marker=markers[impl],
                linewidth=2,
                markersize=8,
            )
        ax.set_title(f"Dropout rate = {rate:.0%}")
        ax.set_xlabel("Number of clients")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        if log_y:
            ax.set_yscale("log")
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def _plot_accuracy_heatmap(
    grouped: dict[tuple[int, float], dict[str, dict[str, Any]]],
) -> None:
    """Create heatmaps of final accuracy for each implementation."""
    rates = sorted({key[1] for key in grouped})
    ns = sorted({key[0] for key in grouped})
    impls = ("classical", "pq_single")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, impl in zip(axes, impls, strict=True):
        matrix = np.zeros((len(rates), len(ns)))
        for i, rate in enumerate(rates):
            for j, n in enumerate(ns):
                row = grouped[(n, rate)].get(impl)
                matrix[i, j] = row["final_accuracy"] if row else 0.0
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.8, vmax=0.92)
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns)
        ax.set_yticks(range(len(rates)))
        ax.set_yticklabels([f"{r:.0%}" for r in rates])
        ax.set_xlabel("Number of clients")
        ax.set_ylabel("Dropout rate")
        ax.set_title(impl)
        for i in range(len(rates)):
            for j in range(len(ns)):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        fig.colorbar(im, ax=ax, label="Final accuracy")

    fig.suptitle("MNIST Final Accuracy Heatmaps")
    fig.tight_layout()
    out_path = OUT_DIR / "mnist_accuracy_heatmaps.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main() -> None:
    """Generate all benchmark plots."""
    data = _load_data()
    grouped = _group(data)

    _plot_metric(
        grouped,
        metric="time_seconds",
        ylabel="Wall time (s)",
        title="MNIST SecAgg+ Training Time",
        filename="mnist_time_comparison.png",
    )
    _plot_metric(
        grouped,
        metric="final_accuracy",
        ylabel="Final test accuracy",
        title="MNIST SecAgg+ Final Accuracy",
        filename="mnist_accuracy_comparison.png",
    )
    _plot_accuracy_heatmap(grouped)


if __name__ == "__main__":
    main()
