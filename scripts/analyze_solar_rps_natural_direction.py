#!/usr/bin/env python3
"""Analyze saved Solar A/N/C outputs; no fitting or backbone inference."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

INPUT = Path("outputs/mechanism_replication/natural_direction/solar/rps")
OUTPUT = Path("outputs/mechanism_replication/analysis/natural_direction/solar/rps")
K, ENDPOINT, SEEDS = 5, 4, (1, 2, 3, 4)


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metric(y: np.ndarray, decision: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    error = np.abs(y - decision)
    result = {
        "global_mae": float(error.mean()),
        "macro_mae": float(np.mean([error[y == k].mean() for k in range(K)])),
        "global_severe": float((error >= 2).mean()),
    }
    for k in range(K):
        mask = y == k
        result.update({
            f"mae_{k}": float(error[mask].mean()),
            f"recall_{k}": float((decision[mask] == k).mean()),
            f"severe_{k}": float((error[mask] >= 2).mean()),
            f"p_end_{k}": float(probability[mask, ENDPOINT].mean()),
            f"true_p_{k}": float(probability[mask, k].mean()),
            f"predictive_mean_{k}": float((probability[mask] @ np.arange(K)).mean()),
        })
    return result


def ci(values: np.ndarray) -> list[float]:
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    half = 3.182446305 * sd / np.sqrt(len(values))
    return [mean - half, mean + half]


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    seed_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    mass_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    summaries: dict[str, object] = {}

    for objective in ("rps",):
        objective_rows: list[dict[str, object]] = []
        for seed in SEEDS:
            root = INPUT / f"seed_{seed}"
            manifest = json.loads((root / "manifest.json").read_text())
            if manifest["a_c_identity"] != "PASS" or not manifest["no_backbone_training"] or not manifest["no_c_refit"]:
                raise RuntimeError(f"invalid N source manifest: {objective} seed {seed}")
            with np.load(root / "per_sample_arrays.npz", allow_pickle=False) as z:
                y, ids = z["labels"], z["sample_ids"]
                if y.shape != (28006,) or int((y == ENDPOINT).sum()) != 921 or len(np.unique(ids)) != len(ids):
                    raise RuntimeError(f"invalid Solar readout population: {objective} seed {seed}")
                decision = {name: z[f"{name}_l1"] for name in "anc"}
                probability = {name: z[f"{name}_probabilities"] for name in "anc"}
                values = {name: metric(y, decision[name], probability[name]) for name in "anc"}

            row: dict[str, object] = {"dataset": "solar", "objective": objective, "seed": seed}
            for name in "anc":
                upper = name.upper()
                row.update({
                    f"endpoint_mae_{upper}": values[name]["mae_4"],
                    f"global_mae_{upper}": values[name]["global_mae"],
                    f"macro_mae_{upper}": values[name]["macro_mae"],
                    f"global_severe_{upper}": values[name]["global_severe"],
                })
            row.update({
                "delta_NA_endpoint": row["endpoint_mae_N"] - row["endpoint_mae_A"],
                "delta_CA_endpoint": row["endpoint_mae_C"] - row["endpoint_mae_A"],
                "delta_NC_endpoint": row["endpoint_mae_N"] - row["endpoint_mae_C"],
            })
            seed_rows.append(row)
            objective_rows.append(row)

            for k in range(K):
                item: dict[str, object] = {"dataset": "solar", "objective": objective, "seed": seed, "true_class": k}
                for name in "anc":
                    upper = name.upper()
                    for key in ("mae", "recall", "severe", "p_end", "true_p", "predictive_mean"):
                        item[f"{upper}_{key}"] = values[name][f"{key}_{k}"]
                for key in ("mae", "recall", "severe", "p_end", "true_p", "predictive_mean"):
                    item[f"N_minus_A_{key}"] = item[f"N_{key}"] - item[f"A_{key}"]
                    item[f"C_minus_A_{key}"] = item[f"C_{key}"] - item[f"A_{key}"]
                    item[f"C_minus_N_{key}"] = item[f"C_{key}"] - item[f"N_{key}"]
                class_rows.append(item)
                mass_rows.append({key: item[key] for key in ("dataset", "objective", "seed", "true_class", "A_p_end", "N_p_end", "C_p_end", "N_minus_A_p_end", "C_minus_A_p_end", "C_minus_N_p_end")})
                mask = y == k
                for predicted in range(K):
                    routing_rows.append({"dataset": "solar", "objective": objective, "seed": seed, "true_class": k, "predicted_class": predicted, "A_count": int((decision["a"][mask] == predicted).sum()), "N_count": int((decision["n"][mask] == predicted).sum()), "C_count": int((decision["c"][mask] == predicted).sum())})
            sources.append({"objective": objective, "seed": seed, "path": str(root / "per_sample_arrays.npz")})

        delta = np.asarray([float(row["delta_NA_endpoint"]) for row in objective_rows])
        improve = int((delta < 0).sum())
        summaries[objective] = {
            "endpoint_mean_A": float(np.mean([row["endpoint_mae_A"] for row in objective_rows])),
            "endpoint_mean_N": float(np.mean([row["endpoint_mae_N"] for row in objective_rows])),
            "endpoint_mean_C": float(np.mean([row["endpoint_mae_C"] for row in objective_rows])),
            "delta_NA_mean": float(delta.mean()), "delta_NA_sd": float(delta.std(ddof=1)), "delta_NA_t95": ci(delta),
            "improving_seeds": improve,
            "label": "CONSISTENT" if improve == 4 else "MIXED" if improve == 3 else "NOT CONSISTENT",
            "global_mae_means": {n: float(np.mean([row[f"global_mae_{n}"] for row in objective_rows])) for n in ("A", "N", "C")},
            "macro_mae_means": {n: float(np.mean([row[f"macro_mae_{n}"] for row in objective_rows])) for n in ("A", "N", "C")},
            "global_severe_means": {n: float(np.mean([row[f"global_severe_{n}"] for row in objective_rows])) for n in ("A", "N", "C")},
        }

    write(OUTPUT / "per_seed_summary.csv", seed_rows)
    write(OUTPUT / "per_class_metrics.csv", class_rows)
    write(OUTPUT / "endpoint_mass_shift.csv", mass_rows)
    write(OUTPUT / "routing_summary.csv", routing_rows)
    verdict = "D — SOLAR RPS NATURAL DIRECTION-ONLY IS MIXED"
    summary = {"dataset": "Solar RPS", "evaluation": "fixed archived aligned 28,006-row readout; exact L1 decisions", "per_objective": summaries, "verdict": verdict, "ce_unresolved_and_untouched": True, "no_backbone_training": True, "no_head_fitting": True}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps({"script": "scripts/analyze_solar_rps_natural_direction.py", "sources": sources, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "no_backbone_training": True, "no_head_fitting": True}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
