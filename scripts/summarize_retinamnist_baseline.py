#!/usr/bin/env python3
"""Aggregate completed Experiment 0 seed directories without rerunning evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs")
    args = parser.parse_args()
    root = Path(args.output_root) / "retinamnist" / "single_model_baseline"
    seed_dirs = sorted(path for path in root.glob("seed_*") if (path / "metrics.json").exists())
    if not seed_dirs: raise FileNotFoundError(f"No completed seed outputs found in {root}")
    summary_dir = root / "summary"
    if summary_dir.exists(): raise FileExistsError(f"Refusing to overwrite existing summary directory: {summary_dir}")
    summary_dir.mkdir(parents=True)
    seed_rows = []
    uncertainty_rows = []
    for directory in seed_dirs:
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        seed = directory.name.removeprefix("seed_")
        seed_rows.append({"seed": seed, **metrics["prediction_metrics"], "test_count": metrics["test_count"], "any_error_count": metrics["any_error_count"], "severe_error_count": metrics["severe_error_count"]})
        for measure, association in metrics["association_with_ordinal_error"].items():
            uncertainty_rows.append({"seed": seed, "measure": measure, **association})
    for filename, rows in (("seed_summary.csv", seed_rows), ("uncertainty_summary.csv", uncertainty_rows)):
        with (summary_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = {"completed_seeds": [int(path.name.removeprefix("seed_")) for path in seed_dirs], "prediction_metric_mean_std": {metric: {"mean": float(np.mean([float(row[metric]) for row in seed_rows])), "std": float(np.std([float(row[metric]) for row in seed_rows], ddof=1)) if len(seed_rows) > 1 else None} for metric in ("accuracy", "mae", "quadratic_weighted_kappa", "nll", "brier_score", "ranked_probability_score", "ece")}}
    (summary_dir / "experiment_summary.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
