#!/usr/bin/env python3
"""Render manuscript Figure 2 exclusively from canonical Figure 2 tables."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DATA_ROOT = Path("outputs/manuscript/figure_data")
OUTPUT_ROOT = Path("outputs/manuscript/figures/figure2")
SAMPLES_PATH = DATA_ROOT / "figure2_mechanism_samples.csv"
SUMMARY_PATH = DATA_ROOT / "figure2_mechanism_summary.csv"
DATASETS = ("RetinaMNIST", "Solar")
GROUPS = ("representation_inward_raw_nearest_interior", "rare_end_like_raw_nearest_4")
GROUP_LABELS = {
    "representation_inward_raw_nearest_interior": "representation-inward",
    "rare_end_like_raw_nearest_4": "rare-end-like",
}
EXPECTED_SUPPORTS = {"RetinaMNIST": 66, "Solar": 921}
CLASS_COLORS = ("#f1f3f4", "#c7d4dc", "#92a9b7", "#5f7889", "#153f56")
INWARD_COLOR = "#bac4cb"
RARE_LIKE_COLOR = "#1e4d5c"
ORIGINAL_COLOR = "#cbd3d9"
DIRECTION_COLOR = "#1e4d5c"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing canonical Figure 2 table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty canonical Figure 2 table: {path}")
    return rows


def as_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key} in {row['dataset']} {row['representation_group']}")
    return value


def validate(
    samples: list[dict[str, str]], summary: list[dict[str, str]]
) -> dict[tuple[str, str], dict[str, str]]:
    expected_groups = {(dataset, group) for dataset in DATASETS for group in GROUPS}
    observed_groups = {(row["dataset"], row["representation_group"]) for row in summary}
    if observed_groups != expected_groups:
        raise ValueError(f"unexpected Figure 2 summary groups: {observed_groups}")
    if {row["dataset"] for row in samples} != set(DATASETS):
        raise ValueError("Figure 2 samples do not contain exactly RetinaMNIST and Solar")
    if any(row["representation_objective"] != "CE" or row["seed"] != "0" for row in samples + summary):
        raise ValueError("Figure 2 must contain CE seed-0 rows only")
    if any(int(row["true_label"]) != 4 for row in samples):
        raise ValueError("Figure 2 sample table contains a non-rare-end label")
    if any(row["representation_group"] not in GROUPS for row in samples + summary):
        raise ValueError("Figure 2 contains an unexpected representation group")
    if any(dataset not in DATASETS for dataset in (row["dataset"] for row in samples + summary)):
        raise ValueError("Figure 2 contains an unexpected dataset")

    index = {(row["dataset"], row["representation_group"]): row for row in summary}
    if len(index) != len(summary):
        raise ValueError("duplicate Figure 2 summary groups")
    for dataset in DATASETS:
        dataset_samples = [row for row in samples if row["dataset"] == dataset]
        if len(dataset_samples) != EXPECTED_SUPPORTS[dataset]:
            raise ValueError(f"unexpected rare-end support for {dataset}: {len(dataset_samples)}")
        sample_ids = [row["sample_id"] for row in dataset_samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"non-unique sample IDs for {dataset}")
        if len({row["true_label"] for row in dataset_samples}) != 1:
            raise ValueError(f"mismatched labels within {dataset}")

        group_support = 0
        for group in GROUPS:
            row = index[(dataset, group)]
            selected = [sample for sample in dataset_samples if sample["representation_group"] == group]
            support = int(row["support"])
            if support != len(selected) or support <= 0:
                raise ValueError(f"sample/summary support mismatch for {dataset} {group}")
            group_support += support
            routing = [int(row[f"routing_{klass}"]) for klass in range(5)]
            if sum(routing) != support or not np.isclose(sum(value / support for value in routing), 1.0, atol=1e-12):
                raise ValueError(f"original routing does not sum to one for {dataset} {group}")
            sample_routing = np.bincount([int(sample["a_l1_prediction"]) for sample in selected], minlength=5)
            if not np.array_equal(sample_routing, np.asarray(routing)):
                raise ValueError(f"sample/summary original routing mismatch for {dataset} {group}")
            a_exact = sum(int(sample["a_exact"]) for sample in selected)
            c_exact = sum(int(sample["c_exact"]) for sample in selected)
            recovery = sum(int(sample["a_to_c_exact_recovery"]) for sample in selected)
            if (a_exact, c_exact, recovery) != (int(row["a_exact"]), int(row["c_exact"]), int(row["a_to_c_exact_recovery"])):
                raise ValueError(f"sample/summary exact counts mismatch for {dataset} {group}")
            if any(not 0 <= int(sample[field]) <= 1 for sample in selected for field in ("a_exact", "c_exact", "a_to_c_exact_recovery")):
                raise ValueError(f"invalid exact indicator for {dataset} {group}")
            mean_delta = float(np.mean([as_float(sample, "delta_predictive_mean") for sample in selected]))
            if not np.isclose(mean_delta, as_float(row, "mean_delta_predictive_mean"), atol=1e-12):
                raise ValueError(f"sample/summary predictive-location delta mismatch for {dataset} {group}")
        if group_support != EXPECTED_SUPPORTS[dataset]:
            raise ValueError(f"subgroup supports do not sum for {dataset}")

    audited = {
        ("RetinaMNIST", "rare_end_like_raw_nearest_4"): (48, 11),
        ("RetinaMNIST", "representation_inward_raw_nearest_interior"): (18, 0),
        ("Solar", "rare_end_like_raw_nearest_4"): (724, 559),
        ("Solar", "representation_inward_raw_nearest_interior"): (197, 0),
    }
    for key, (support, recovery) in audited.items():
        row = index[key]
        if int(row["support"]) != support or int(row["a_to_c_exact_recovery"]) != recovery:
            raise ValueError(f"audited Figure 2 count mismatch for {key}")
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


def draw_representation_state(axis: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    y_positions = np.arange(len(DATASETS))[::-1]
    for y, dataset in zip(y_positions, DATASETS):
        inward = int(summary[(dataset, GROUPS[0])]["support"])
        rare_like = int(summary[(dataset, GROUPS[1])]["support"])
        total = inward + rare_like
        inward_fraction = inward / total
        axis.barh(y, inward_fraction, color=INWARD_COLOR, edgecolor="#35434c", linewidth=0.6, hatch="///", height=0.46)
        axis.barh(y, 1 - inward_fraction, left=inward_fraction, color=RARE_LIKE_COLOR, edgecolor="#35434c", linewidth=0.6, height=0.46)
        axis.text(inward_fraction / 2, y, f"{inward}\n({inward_fraction:.0%})", ha="center", va="center", fontsize=7.6)
        axis.text(inward_fraction + (1 - inward_fraction) / 2, y, f"{rare_like}\n({1 - inward_fraction:.0%})", ha="center", va="center", fontsize=7.6, color="white")
    axis.set_xlim(0, 1)
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1), ("0%", "25%", "50%", "75%", "100%"))
    axis.set_yticks(y_positions, DATASETS)
    axis.set_xlabel("fraction of true rare-end samples")
    axis.grid(axis="x", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    axis.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=INWARD_COLOR, edgecolor="#35434c", hatch="///", label="representation-inward"),
        plt.Rectangle((0, 0), 1, 1, facecolor=RARE_LIKE_COLOR, edgecolor="#35434c", label="rare-end-like"),
    ], frameon=False, fontsize=7.0, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)


def draw_original_routing(axis: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    bars = [(dataset, group) for dataset in DATASETS for group in GROUPS]
    y_positions = np.arange(len(bars))[::-1]
    for y, (dataset, group) in zip(y_positions, bars):
        row = summary[(dataset, group)]
        support = int(row["support"])
        fractions = [int(row[f"routing_{klass}"]) / support for klass in range(5)]
        left = 0.0
        for klass, fraction in enumerate(fractions):
            container = axis.barh(y, fraction, left=left, color=CLASS_COLORS[klass], edgecolor="#35434c", linewidth=0.45, height=0.48)
            if klass == 4:
                container[0].set_hatch("///")
            left += fraction
        class_three = fractions[3]
        annotation = f"{class_three:.0%} → class 3"
        if class_three >= 0.20:
            axis.text(sum(fractions[:3]) + class_three / 2, y, annotation, ha="center", va="center", fontsize=7.0, color="white")
        else:
            axis.text(1.015, y, annotation, ha="left", va="center", fontsize=7.0, color="#4f5b63", clip_on=False)
    axis.set_xlim(0, 1.15)
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1), ("0%", "25%", "50%", "75%", "100%"))
    axis.set_yticks(y_positions, [f"{dataset}\n{GROUP_LABELS[group]}" for dataset, group in bars])
    axis.set_xlabel("original-head L1 routing fraction")
    axis.grid(axis="x", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    axis.legend(handles=[
        plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLORS[klass], edgecolor="#35434c", hatch="///" if klass == 4 else None, label=f"class {klass}")
        for klass in range(5)
    ], frameon=False, fontsize=7.0, loc="lower center", bbox_to_anchor=(0.43, -0.34), ncol=5, columnspacing=0.7)


def draw_exact_recovery(axis: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    bars = [(dataset, group) for dataset in DATASETS for group in GROUPS]
    y_positions = np.arange(len(bars))[::-1]
    for y, (dataset, group) in zip(y_positions, bars):
        row = summary[(dataset, group)]
        support = int(row["support"])
        original = int(row["a_exact"]) / support
        direction = int(row["c_exact"]) / support
        original_y, direction_y = (y + 0.09, y - 0.09) if original == direction == 0 else (y, y)
        axis.plot([original, direction], [original_y, direction_y], color="#7b8790", linewidth=1.8, zorder=1)
        axis.scatter(original, original_y, s=48, marker="o", color=ORIGINAL_COLOR, edgecolor="#35434c", linewidth=0.7, zorder=2)
        axis.scatter(direction, direction_y, s=50, marker="D", color=DIRECTION_COLOR, edgecolor=DIRECTION_COLOR, zorder=3)
        if direction > 0:
            axis.text(direction, y + 0.18, f"{int(row['c_exact'])}/{support}\n({direction:.0%})", ha="center", va="bottom", fontsize=7.4)
        else:
            axis.text(0.018, original_y, "0", ha="left", va="center", fontsize=7.0, color="#4f5b63")
            axis.text(0.018, direction_y, "0", ha="left", va="center", fontsize=7.0, color="#4f5b63")
    axis.set_xlim(-0.02, 0.85)
    axis.set_xticks((0, 0.2, 0.4, 0.6, 0.8), ("0%", "20%", "40%", "60%", "80%"))
    axis.set_yticks(y_positions, [f"{dataset}\n{GROUP_LABELS[group]}" for dataset, group in bars])
    axis.set_xlabel("exact rare-end fraction")
    axis.grid(axis="x", linewidth=0.45, color="#d9dfe3", alpha=0.8)
    axis.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ORIGINAL_COLOR, markeredgecolor="#35434c", markersize=6, label="A original head"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=DIRECTION_COLOR, markeredgecolor=DIRECTION_COLOR, markersize=6, label="C direction-only head"),
    ], frameon=False, fontsize=7.0, loc="upper right")


def draw_outward_movement(axis: plt.Axes, summary: dict[tuple[str, str], dict[str, str]]) -> None:
    bars = [(dataset, group) for dataset in DATASETS for group in GROUPS]
    y_positions = np.arange(len(bars))[::-1]
    for y, (dataset, group) in zip(y_positions, bars):
        delta = as_float(summary[(dataset, group)], "mean_delta_predictive_mean")
        color = RARE_LIKE_COLOR if group == GROUPS[1] else INWARD_COLOR
        hatch = None if group == GROUPS[1] else "///"
        container = axis.barh(y, delta, color=color, edgecolor="#35434c", linewidth=0.55, height=0.48)
        container[0].set_hatch(hatch)
        axis.text(delta + 0.018, y, f"+{delta:.2f}", ha="left", va="center", fontsize=7.5)
    axis.axvline(0, color="#59666f", linewidth=0.8)
    axis.set_xlim(-0.03, 0.72)
    axis.set_yticks(y_positions, [f"{dataset}\n{GROUP_LABELS[group]}" for dataset, group in bars])
    axis.set_xlabel("Mean Δ predictive location (C − A)")
    axis.grid(axis="x", linewidth=0.45, color="#d9dfe3", alpha=0.8)


def render(summary: dict[tuple[str, str], dict[str, str]]) -> None:
    set_style()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(11.3, 7.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(0.92, 1.08), hspace=0.18, wspace=0.18)
    representation_axis = figure.add_subplot(grid[0, 0])
    routing_axis = figure.add_subplot(grid[0, 1])
    recovery_axis = figure.add_subplot(grid[1, 0])
    movement_axis = figure.add_subplot(grid[1, 1])
    draw_representation_state(representation_axis, summary)
    draw_original_routing(routing_axis, summary)
    draw_exact_recovery(recovery_axis, summary)
    draw_outward_movement(movement_axis, summary)
    panel_heading(representation_axis, "A", "Representation state")
    panel_heading(routing_axis, "B", "Original-head routing conditioned on state")
    panel_heading(recovery_axis, "C", "Exact rare-end recovery under direction adaptation")
    panel_heading(movement_axis, "D", "Outward movement beyond exact recovery")
    figure.savefig(
        OUTPUT_ROOT / "figure2.pdf",
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(OUTPUT_ROOT / "figure2.png", dpi=360, bbox_inches="tight")
    plt.close(figure)


def write_caption() -> None:
    caption = (
        "**Figure 2 | Rare-end localization failure contains both representation- and head-level components.** "
        "Among true rare upper-end samples, some are representation-inward under the audited training-centroid diagnostic, "
        "whereas others retain rare-end-like geometry (A). The original CE head still maps the latter predominantly to interior "
        "classes (B). In these audited RetinaMNIST training-only OOF and Solar aligned-test CE settings, exact rare-end recovery "
        "under direction-only adaptation occurs only in the rare-end-like subgroup (C). Representation-inward samples nevertheless "
        "show positive mean outward movement in predictive location without exact recovery (D)."
    )
    (OUTPUT_ROOT / "caption.md").write_text(caption + "\n")


def main() -> None:
    samples = read_csv(SAMPLES_PATH)
    summary_rows = read_csv(SUMMARY_PATH)
    summary = validate(samples, summary_rows)
    render(summary)
    write_caption()
    for path in (OUTPUT_ROOT / "figure2.pdf", OUTPUT_ROOT / "figure2.png", OUTPUT_ROOT / "caption.md"):
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError(f"failed to create non-empty Figure 2 output: {path}")
    print(f"Figure 2 rendered from canonical tables into {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
