#!/usr/bin/env python3
"""Render manuscript Figure 1 exclusively from canonical figure-data tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DATA_ROOT = Path("outputs/manuscript/figure_data")
OUTPUT_ROOT = Path("outputs/manuscript/figures/figure1")
SAMPLES_PATH = DATA_ROOT / "figure1_endpoint_samples.csv"
SUMMARY_PATH = DATA_ROOT / "figure1_endpoint_summary.csv"
DATASETS = ("RetinaMNIST", "UTKFace", "Solar")
CLASS_LABELS = ("0", "1", "2", "3", "4")
UPPER_CLASS = 4


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing canonical Figure 1 table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty canonical Figure 1 table: {path}")
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} for {row['dataset']} {row['endpoint']}")
    return value


def validate(samples: list[dict[str, str]], summary: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    expected = {(dataset, endpoint) for dataset in DATASETS for endpoint in ("lower", "upper")}
    observed = {(row["dataset"], row["endpoint"]) for row in summary}
    if observed != expected:
        raise ValueError(f"unexpected Figure 1 summary groups: {observed}")
    if {row["dataset"] for row in samples} != set(DATASETS):
        raise ValueError("Figure 1 samples do not contain exactly the three expected datasets")
    if any(row["model_objective"] != "CE" or row["seed"] != "0" for row in samples + summary):
        raise ValueError("Figure 1 must contain original CE seed-0 rows only")
    if any(int(row["true_label"]) not in (0, UPPER_CLASS) for row in samples):
        raise ValueError("Figure 1 sample table contains a non-endpoint row")

    index = {(row["dataset"], row["endpoint"]): row for row in summary}
    if len(index) != len(summary):
        raise ValueError("duplicate Figure 1 summary groups")
    for dataset in DATASETS:
        upper = index[(dataset, "upper")]
        lower = index[(dataset, "lower")]
        if int(upper["true_label"]) != UPPER_CLASS or int(lower["true_label"]) != 0:
            raise ValueError(f"endpoint labels are invalid for {dataset}")
        upper_routing = [int(upper[f"routing_{klass}"]) for klass in range(5)]
        support = int(upper["support"])
        if support <= 0 or sum(upper_routing) != support:
            raise ValueError(f"rare-end routing does not sum to support for {dataset}")
        if not np.isclose(sum(value / support for value in upper_routing), 1.0, atol=1e-12):
            raise ValueError(f"rare-end routing fractions do not sum to one for {dataset}")
        for row in (upper, lower):
            for field in ("l1_mae", "predictive_mean", "inward_shrinkage"):
                as_float(row, field)

        selected = [row for row in samples if row["dataset"] == dataset and row["endpoint"] == "upper"]
        if len(selected) != support:
            raise ValueError(f"sample/summary rare support mismatch for {dataset}")
        sample_routing = np.bincount([int(row["l1_prediction"]) for row in selected], minlength=5)
        if not np.array_equal(sample_routing, np.asarray(upper_routing)):
            raise ValueError(f"sample/summary rare routing mismatch for {dataset}")
    return index


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def panel_heading(axis: plt.Axes, letter: str, title: str, letter_x: float = -0.04) -> None:
    """Place a panel letter and title on one baseline above the plot area."""
    y_position = 1.16
    axis.text(letter_x, y_position, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="baseline")
    axis.text(0.0, y_position, title, transform=axis.transAxes, fontsize=10, fontweight="semibold", va="baseline")


def draw_routing(axis: plt.Axes, dataset: str, row: dict[str, str]) -> None:
    support = int(row["support"])
    fractions = np.asarray([int(row[f"routing_{klass}"]) / support for klass in range(5)])
    colors = ["#d0d5da", "#aeb7c0", "#8996a3", "#637282", "#193b4b"]
    bars = axis.bar(np.arange(5), fractions, color=colors, edgecolor="#26333b", linewidth=0.55, width=0.72)
    bars[-1].set_hatch("///")
    for index, value in enumerate(fractions):
        if value > 0.035:
            axis.text(index, value + 0.025, f"{value:.0%}", ha="center", va="bottom", fontsize=7.5)
    axis.set_title(dataset, pad=11, fontweight="semibold")
    axis.set_xticks(range(5), CLASS_LABELS)
    axis.set_ylim(0, 1.08)
    axis.set_xlabel("L1 decision class")
    axis.axvspan(3.58, 4.42, color="#193b4b", alpha=0.06, zorder=0)
    axis.text(4, 1.025, "true endpoint", ha="center", va="bottom", color="#193b4b", fontsize=7.3)
    axis.text(0.02, 0.94, f"true class 4, n={support}", transform=axis.transAxes, ha="left", va="top", fontsize=7.4)
    axis.grid(axis="y", linewidth=0.45, color="#d9dfe3", alpha=0.8)


def draw_location(axis: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    y_positions = np.arange(len(DATASETS))[::-1]
    for y, dataset in zip(y_positions, DATASETS):
        row = summary[(dataset, "upper")]
        location = as_float(row, "predictive_mean")
        axis.plot([location, UPPER_CLASS], [y, y], color="#6e7b85", linewidth=2.2, solid_capstyle="round")
        axis.scatter(location, y, s=56, color="#193b4b", edgecolor="white", linewidth=0.8, zorder=3)
        axis.text(location, y + 0.18, f"μ={location:.2f}", ha="center", va="bottom", fontsize=8)
        axis.text((location + UPPER_CLASS) / 2, y - 0.22, f"S={as_float(row, 'inward_shrinkage'):.2f}", ha="center", va="top", fontsize=7.8, color="#4f5b63")
    axis.set_xlim(-0.1, 4.25)
    axis.set_xticks(range(5), CLASS_LABELS)
    axis.set_yticks(y_positions, DATASETS)
    axis.set_xlabel("ordinal predictive location")
    axis.axvline(UPPER_CLASS, color="#a03d32", linewidth=0.8, linestyle="--", alpha=0.65)
    axis.grid(axis="x", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    axis.text(UPPER_CLASS, -0.62, "true endpoint", ha="center", va="top", fontsize=7.5, color="#a03d32")


def draw_endpoint_mae(axis: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    y_positions = np.arange(len(DATASETS))[::-1]
    for y, dataset in zip(y_positions, DATASETS):
        lower = as_float(summary[(dataset, "lower")], "l1_mae")
        upper = as_float(summary[(dataset, "upper")], "l1_mae")
        axis.plot([lower, upper], [y, y], color="#7b8790", linewidth=1.8, zorder=1)
        axis.scatter(lower, y, s=48, marker="o", color="#c8d0d6", edgecolor="#26333b", linewidth=0.7, zorder=2)
        axis.scatter(upper, y, s=48, marker="D", color="#193b4b", edgecolor="#193b4b", linewidth=0.7, zorder=2)
        axis.text(lower, y + 0.18, f"{lower:.2f}", ha="center", va="bottom", fontsize=7.8)
        axis.text(upper, y + 0.18, f"{upper:.2f}", ha="center", va="bottom", fontsize=7.8)
    axis.set_xlim(-0.04, 1.85)
    axis.set_yticks(y_positions, DATASETS)
    axis.set_xlabel("class-conditional L1 MAE")
    axis.grid(axis="x", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    axis.legend(handles=[
        Line2D([0], [0], marker="o", color="#7b8790", markerfacecolor="#c8d0d6", markeredgecolor="#26333b", markersize=6, linewidth=0, label="lower endpoint (class 0)"),
        Line2D([0], [0], marker="D", color="#193b4b", markerfacecolor="#193b4b", markersize=6, linewidth=0, label="upper endpoint (class 4)"),
    ], frameon=False, fontsize=7.0, loc="lower right")


def render(summary: dict[tuple[str, str], dict[str, str]]) -> None:
    set_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(12.6, 8.0), constrained_layout=True)
    grid = figure.add_gridspec(3, 3, height_ratios=(1.55, 0.88, 0.88), hspace=0.18)
    routing_axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    for axis, dataset in zip(routing_axes, DATASETS):
        draw_routing(axis, dataset, summary[(dataset, "upper")])
    routing_axes[0].set_ylabel("fraction of true upper-end samples")
    panel_heading(
        routing_axes[0],
        "A",
        "Upper-extreme L1 decision routing",
        letter_x=-0.12,
    )

    location_axis = figure.add_subplot(grid[1, :])
    draw_location(location_axis, summary)
    panel_heading(location_axis, "B", "Predictive location and inward shrinkage")

    mae_axis = figure.add_subplot(grid[2, :])
    draw_endpoint_mae(mae_axis, summary)
    panel_heading(mae_axis, "C", "Endpoint-specific L1 MAE")

    figure.savefig(OUTPUT_ROOT / "figure1.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT_ROOT / "figure1.png", dpi=360, bbox_inches="tight")
    plt.close(figure)


def write_caption() -> None:
    caption = (
        "**Figure 1 | Rare upper extremes exhibit inward localization across the studied ordinal tasks.** "
        "Under the original CE classifiers, exact discrete L1 decisions for true upper-end samples are shifted "
        "toward interior classes (A), and the predictive distributions remain centered below the true endpoint (B). "
        "The lower-versus-upper endpoint comparison provides an asymmetry control: upper-end failure is particularly "
        "pronounced in the strongest settings, while its severity varies across datasets."
    )
    (OUTPUT_ROOT / "caption.md").write_text(caption + "\n")


def main() -> None:
    samples = read_csv(SAMPLES_PATH)
    summary_rows = read_csv(SUMMARY_PATH)
    summary = validate(samples, summary_rows)
    render(summary)
    write_caption()
    for path in (OUTPUT_ROOT / "figure1.pdf", OUTPUT_ROOT / "figure1.png", OUTPUT_ROOT / "caption.md"):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"failed to create non-empty Figure 1 output: {path}")
    print(f"Figure 1 rendered from canonical tables into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
