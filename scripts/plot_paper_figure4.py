#!/usr/bin/env python3
"""Render manuscript Figure 4 exclusively from canonical Figure 4 tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DATA_ROOT = Path("outputs/manuscript/figure_data")
OUTPUT_ROOT = Path("outputs/manuscript/figures/figure4")
FACTORIAL_PATH = DATA_ROOT / "figure4_factorial_summary.csv"
PER_SEED_PATH = DATA_ROOT / "figure4_severity_per_seed.csv"
SEVERITY_PATH = DATA_ROOT / "figure4_severity_summary.csv"
SUPPORTS = (66, 50, 33, 16, 8)
SEEDS = (0, 1, 2, 3, 4)
OBJECTIVE_STYLE = {"CE": ("#1e4d5c", "o"), "RPS": ("#728391", "s")}
SEED_MARKERS = ("o", "s", "^", "D", "P")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing canonical Figure 4 table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty canonical Figure 4 table: {path}")
    return rows


def number(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}")
    return value


def validate(factorial: list[dict[str, str]], per_seed: list[dict[str, str]], severity: list[dict[str, str]]) -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[int, int], dict[str, str]], dict[int, dict[str, str]]]:
    expected_factorial = {(sampling, objective) for sampling in ("natural", "balanced") for objective in ("CE", "RPS")}
    observed_factorial = {(row["sampling"], row["objective"]) for row in factorial}
    if observed_factorial != expected_factorial or len(factorial) != 4:
        raise ValueError("Figure 4 must contain exactly four factorial cells")
    if any(row["representation_objective"] != "RPS" or row["population"] != "training_only_oof" or row["backbone_seed"] != "0" or int(row["rare_n"]) != 66 for row in factorial):
        raise ValueError("unexpected Phase 3.19 factorial protocol")
    for row in factorial:
        number(row, "c4_l1_mae"); number(row, "z4_z3_mean")
    factorial_index = {(row["sampling"], row["objective"]): row for row in factorial}

    expected_cells = {(seed, support) for seed in SEEDS for support in SUPPORTS}
    observed_cells = {(int(row["seed"]), int(row["n4_support"])) for row in per_seed}
    if observed_cells != expected_cells or len(per_seed) != 25:
        raise ValueError("Figure 4 severity table is not a complete 25-cell grid")
    for row in per_seed:
        for field in ("mean_p4", "margin_z4_z3", "shrinkage"):
            number(row, field)
    per_seed_index = {(int(row["seed"]), int(row["n4_support"])): row for row in per_seed}
    summary_index = {int(row["n4_support"]): row for row in severity}
    if set(summary_index) != set(SUPPORTS) or len(summary_index) != 5:
        raise ValueError("unexpected severity summary supports")
    for support in SUPPORTS:
        rows = [per_seed_index[(seed, support)] for seed in SEEDS]
        for source, aggregate in (("mean_p4", "mean_p4_mean"), ("margin_z4_z3", "margin_z4_z3_mean"), ("shrinkage", "shrinkage_mean")):
            if not math.isclose(float(np.mean([number(row, source) for row in rows])), number(summary_index[support], aggregate), abs_tol=1e-12):
                raise ValueError(f"severity aggregate mismatch for N4={support}, {source}")
    return factorial_index, per_seed_index, summary_index


def set_style() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})


def heading(axis: plt.Axes, letter: str, title: str) -> None:
    axis.text(-0.12, 1.14, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="baseline")
    axis.text(0, 1.14, title, transform=axis.transAxes, fontsize=10, fontweight="semibold", va="baseline")


def draw_factorial(axis: plt.Axes, table: dict[tuple[str, str], dict[str, str]], field: str, ylabel: str, zero: bool = False) -> None:
    for position, sampling in enumerate(("natural", "balanced")):
        for offset, objective in ((-0.12, "CE"), (0.12, "RPS")):
            color, marker = OBJECTIVE_STYLE[objective]
            value = number(table[(sampling, objective)], field)
            axis.scatter(position + offset, value, s=58, marker=marker, color=color, edgecolor="white", linewidth=0.7, zorder=3)
            axis.text(position + offset, value + (0.035 if field == "c4_l1_mae" else 0.055), f"{value:.2f}", ha="center", va="bottom", fontsize=7.2)
    axis.set_xlim(-0.45, 1.45); axis.set_xticks((0, 1), ("Natural sampling", "Balanced sampling")); axis.set_ylabel(ylabel)
    axis.grid(axis="y", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    if field == "c4_l1_mae": axis.set_ylim(1.30, 1.755)
    if zero: axis.axhline(0, color="#59666f", linewidth=0.8, linestyle="--", zorder=0)


def draw_severity(axis: plt.Axes, per_seed: dict[tuple[int, int], dict[str, str]], summary: dict[int, dict[str, str]], field: str, aggregate: str, ylabel: str, zero: bool = False) -> None:
    positions = np.arange(len(SUPPORTS))
    for seed, marker in zip(SEEDS, SEED_MARKERS):
        values = [number(per_seed[(seed, support)], field) for support in SUPPORTS]
        axis.plot(positions, values, color="#aeb9c1", linewidth=0.8, marker=marker, markersize=3.8, alpha=0.9, zorder=1)
    means = [number(summary[support], aggregate) for support in SUPPORTS]
    axis.plot(positions, means, color="#1e4d5c", linewidth=2.1, marker="o", markersize=5.5, markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    axis.set_xticks(positions, [str(support) for support in SUPPORTS]); axis.set_xlabel("Class-4 training support N4"); axis.set_ylabel(ylabel)
    axis.grid(axis="y", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    if zero: axis.axhline(0, color="#59666f", linewidth=0.8, linestyle="--", zorder=0)


def render(factorial: dict[tuple[str, str], dict[str, str]], per_seed: dict[tuple[int, int], dict[str, str]], severity: dict[int, dict[str, str]]) -> None:
    set_style(); OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(13.4, 8.8))
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.84)
    grid = figure.add_gridspec(2, 6, height_ratios=(1, 1.05), hspace=0.60, wspace=0.32)
    a, b = figure.add_subplot(grid[0, :3]), figure.add_subplot(grid[0, 3:])
    c, d, e = figure.add_subplot(grid[1, :2]), figure.add_subplot(grid[1, 2:4]), figure.add_subplot(grid[1, 4:])
    draw_factorial(a, factorial, "c4_l1_mae", "class-4 L1 MAE")
    draw_factorial(b, factorial, "z4_z3_mean", "mean z4 − z3 margin", zero=True)
    draw_severity(c, per_seed, severity, "mean_p4", "mean_p4_mean", "mean p4 on true class 4")
    draw_severity(d, per_seed, severity, "margin_z4_z3", "margin_z4_z3_mean", "mean z4 − z3 margin", zero=True)
    draw_severity(e, per_seed, severity, "shrinkage", "shrinkage_mean", "inward shrinkage")
    heading(a, "A", "C4 MAE by sampling and objective"); heading(b, "B", "C4-vs-C3 logit margin")
    heading(c, "C", "Rare-class probability vs. support"); heading(d, "D", "Rare-vs-adjacent margin vs. support"); heading(e, "E", "Inward shrinkage vs. support")
    figure.text(0.5, 0.965, "Controlled direction-only mechanism study — frozen RPS representation, training-only OOF", ha="center", va="center", fontsize=11.5, fontweight="semibold", color="#4f5b63")
    figure.text(0.5, 0.500, "Independent full-model severity study — CE retraining, 5 seeds per support level", ha="center", va="center", fontsize=11.5, fontweight="semibold", color="#4f5b63")
    a.legend(handles=[Line2D([0], [0], marker=OBJECTIVE_STYLE[o][1], color="none", markerfacecolor=OBJECTIVE_STYLE[o][0], markersize=6.5, label=o) for o in ("CE", "RPS")], frameon=False, fontsize=7.5, loc="lower left")
    c.legend(handles=[Line2D([0], [0], color="#aeb9c1", marker="o", linewidth=1.0, markersize=4.8, label="individual seeds"), Line2D([0], [0], color="#1e4d5c", marker="o", linewidth=2.3, markersize=5.8, label="mean")], frameon=False, fontsize=7.8, loc="upper right")
    figure.savefig(OUTPUT_ROOT / "figure4.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(OUTPUT_ROOT / "figure4.png", dpi=360, bbox_inches="tight")
    plt.close(figure)


def write_caption() -> None:
    text = "**Figure 4 | Balanced sampling is the dominant adaptation factor in the controlled RetinaMNIST direction-only mechanism study, while reduced rare-class support shows only a partial relationship with localization severity.** In the frozen-RPS 2×2 factorial experiment, balanced sampling lowers class-4 MAE and improves the class-4-versus-class-3 margin under both CE and RPS objectives, while within-regime objective differences are small. In the separate five-seed full-CE severity study, reducing class-4 support lowers p4 and worsens z4-z3, but inward shrinkage is non-monotonic and seed-dependent. The blocks are distinct experimental regimes, not one causal chain."
    (OUTPUT_ROOT / "caption.md").write_text(text + "\n")


def main() -> None:
    factorial, per_seed, severity = validate(read_csv(FACTORIAL_PATH), read_csv(PER_SEED_PATH), read_csv(SEVERITY_PATH))
    render(factorial, per_seed, severity); write_caption()
    for path in (OUTPUT_ROOT / "figure4.pdf", OUTPUT_ROOT / "figure4.png", OUTPUT_ROOT / "caption.md"):
        if not path.exists() or path.stat().st_size == 0: raise RuntimeError(f"failed to create {path}")
    print(f"Figure 4 rendered from canonical tables into {OUTPUT_ROOT}")


if __name__ == "__main__": main()
