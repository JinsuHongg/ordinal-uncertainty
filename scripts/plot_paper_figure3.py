#!/usr/bin/env python3
"""Render manuscript Figure 3 exclusively from the canonical Figure 3 summary."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DATA_ROOT = Path("outputs/manuscript/figure_data")
OUTPUT_ROOT = Path("outputs/manuscript/figures/figure3")
SUMMARY_PATH = DATA_ROOT / "figure3_direction_summary.csv"
DATASETS = ("RetinaMNIST", "Solar")
OBJECTIVES = ("RPS", "CE")
CONDITIONS = ("A_original", "C_direction_only")
SETTING_ORDER = (
    ("RetinaMNIST", "RPS"),
    ("RetinaMNIST", "CE"),
    ("Solar", "RPS"),
    ("Solar", "CE"),
)
PROTOCOLS = {
    "RetinaMNIST": "training_only_oof",
    "Solar": "retained_aligned_future_test",
}
MARKERS = {"RetinaMNIST": "o", "Solar": "s"}
COLORS = {"RPS": "#728391", "CE": "#1e4d5c"}
LINESTYLES = {"RPS": "--", "CE": "-"}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing canonical Figure 3 table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty canonical Figure 3 table: {path}")
    return rows


def as_float(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field} for {row['dataset']} {row['representation_objective']} {row['condition']}")
    return value


def validate(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    expected = {(dataset, objective, condition) for dataset, objective in SETTING_ORDER for condition in CONDITIONS}
    observed = {(row["dataset"], row["representation_objective"], row["condition"]) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"unexpected Figure 3 setting rows: {observed}")
    if {row["dataset"] for row in rows} != set(DATASETS):
        raise ValueError("Figure 3 must contain exactly RetinaMNIST and Solar")
    if {row["representation_objective"] for row in rows} != set(OBJECTIVES):
        raise ValueError("Figure 3 must contain exactly CE and RPS")
    if {row["condition"] for row in rows} != set(CONDITIONS):
        raise ValueError("Figure 3 must contain exactly A original and C balanced direction-constrained rows")
    if any(row["seed"] != "0" or row["rare_class"] != "4" for row in rows):
        raise ValueError("Figure 3 contains an unexpected seed or rare class")
    if any(row["population"] != PROTOCOLS[row["dataset"]] for row in rows):
        raise ValueError("Figure 3 has an unexpected protocol population")

    index = {(row["dataset"], row["representation_objective"], row["condition"]): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("duplicate Figure 3 setting rows")
    for dataset, objective in SETTING_ORDER:
        original = index[(dataset, objective, "A_original")]
        direction = index[(dataset, objective, "C_direction_only")]
        if int(original["rare_n"]) != int(direction["rare_n"]):
            raise ValueError(f"A/C rare denominator mismatch for {dataset} {objective}")
        expected_n = 66 if dataset == "RetinaMNIST" else 921
        if int(original["rare_n"]) != expected_n:
            raise ValueError(f"unexpected rare denominator for {dataset} {objective}")
        for row in (original, direction):
            for field in ("rare_l1_mae", "rare_exact_fraction", "rare_inward_shrinkage", "rare_z4_z3_mean"):
                as_float(row, field)
            fraction = as_float(row, "rare_exact_fraction")
            count = int(row["rare_exact_count"])
            if not 0 <= fraction <= 1 or not math.isclose(fraction, count / int(row["rare_n"]), abs_tol=1e-12):
                raise ValueError(f"invalid exact fraction for {dataset} {objective} {row['condition']}")
        if not (as_float(direction, "rare_l1_mae") < as_float(original, "rare_l1_mae")):
            raise ValueError(f"rare-end MAE does not improve A to C for {dataset} {objective}")
        if not (as_float(direction, "rare_inward_shrinkage") < as_float(original, "rare_inward_shrinkage")):
            raise ValueError(f"inward shrinkage does not decrease A to C for {dataset} {objective}")
        if not (as_float(direction, "rare_z4_z3_mean") > as_float(original, "rare_z4_z3_mean")):
            raise ValueError(f"adjacent margin does not improve A to C for {dataset} {objective}")
    return index


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def panel_heading(axis: plt.Axes, letter: str, title: str) -> None:
    axis.text(-0.12, 1.13, letter, transform=axis.transAxes, fontsize=12, fontweight="bold", va="baseline")
    axis.text(0.0, 1.13, title, transform=axis.transAxes, fontsize=10, fontweight="semibold", va="baseline")


def setting_style(dataset: str, objective: str) -> dict[str, object]:
    return {"color": COLORS[objective], "linestyle": LINESTYLES[objective], "marker": MARKERS[dataset]}


def draw_paired_metric(
    axis: plt.Axes,
    summary: dict[tuple[str, str, str], dict[str, str]],
    field: str,
    ylabel: str,
    annotate_counts: bool = False,
    zero_line: bool = False,
) -> None:
    for dataset, objective in SETTING_ORDER:
        original = summary[(dataset, objective, "A_original")]
        direction = summary[(dataset, objective, "C_direction_only")]
        values = (as_float(original, field), as_float(direction, field))
        style = setting_style(dataset, objective)
        axis.plot((0, 1), values, linewidth=1.8, markersize=6.3, markeredgecolor="white", markeredgewidth=0.7, **style)
        if annotate_counts:
            axis.annotate(
                f"{int(direction['rare_exact_count'])}/{int(direction['rare_n'])}",
                xy=(1, values[1]), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7.6, color=style["color"],
            )
    axis.set_xlim(-0.12, 1.32 if annotate_counts else 1.12)
    axis.set_xticks((0, 1), ("A original", "C: balanced direction-constrained"))
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    if zero_line:
        axis.axhline(0, color="#59666f", linewidth=0.8, linestyle="--", zorder=0)


def shared_legend(figure: plt.Figure) -> None:
    handles = []
    for dataset, objective in SETTING_ORDER:
        style = setting_style(dataset, objective)
        handles.append(Line2D(
            [0], [0], label=f"{dataset} {objective}", linewidth=2.2, markersize=7.4,
            markeredgecolor="white", markeredgewidth=0.7, **style,
        ))
    figure.legend(handles=handles, frameon=False, fontsize=8.7, loc="outside lower center", ncol=4, columnspacing=1.55)


def render(summary: dict[tuple[str, str, str], dict[str, str]]) -> None:
    set_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(11.3, 7.8), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, hspace=0.24, wspace=0.26)
    mae_axis = figure.add_subplot(grid[0, 0])
    exact_axis = figure.add_subplot(grid[0, 1])
    shrinkage_axis = figure.add_subplot(grid[1, 0])
    margin_axis = figure.add_subplot(grid[1, 1])
    draw_paired_metric(mae_axis, summary, "rare_l1_mae", "rare-end L1 MAE")
    draw_paired_metric(exact_axis, summary, "rare_exact_fraction", "exact rare-end fraction", annotate_counts=True)
    draw_paired_metric(shrinkage_axis, summary, "rare_inward_shrinkage", "inward shrinkage, S")
    draw_paired_metric(margin_axis, summary, "rare_z4_z3_mean", "mean rare-vs-adjacent logit margin", zero_line=True)
    panel_heading(mae_axis, "A", "Rare-end L1 MAE")
    panel_heading(exact_axis, "B", "Exact rare-end recovery")
    panel_heading(shrinkage_axis, "C", "Inward shrinkage")
    panel_heading(margin_axis, "D", "Rare-end vs. adjacent-class logit margin")
    shared_legend(figure)
    figure.savefig(OUTPUT_ROOT / "figure3.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    figure.savefig(OUTPUT_ROOT / "figure3.png", dpi=360, bbox_inches="tight")
    plt.close(figure)


def write_caption() -> None:
    caption = (
        "**Figure 3 | Classifier-direction adaptation improves rare-end localization across the evaluated CE- and RPS-trained representations.** "
        "Across all four within-setting A→C comparisons, balanced direction-constrained adaptation reduces rare-end L1 MAE and inward shrinkage, "
        "increases exact endpoint recovery, and improves the rare-end-versus-adjacent-class logit margin. RetinaMNIST results use "
        "training-only OOF head evaluation, whereas Solar results use the predeclared archived confirmatory readout. The consistent "
        "response direction is not interpreted as objective independence, equivalence, universal correction, or global/UQ improvement."
    )
    (OUTPUT_ROOT / "caption.md").write_text(caption + "\n")


def main() -> None:
    rows = read_csv(SUMMARY_PATH)
    summary = validate(rows)
    render(summary)
    write_caption()
    for path in (OUTPUT_ROOT / "figure3.pdf", OUTPUT_ROOT / "figure3.png", OUTPUT_ROOT / "caption.md"):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"failed to create non-empty Figure 3 output: {path}")
    print(f"Figure 3 rendered from canonical table into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
