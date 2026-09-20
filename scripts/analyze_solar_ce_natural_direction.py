#!/usr/bin/env python3
"""Analyze the frozen Solar CE A/N/C follow-up without fitting any head."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


N_ROOT = Path("outputs/mechanism_replication/natural_direction/solar/ce")
REPLACEMENT_ROOT = Path("outputs/solar/mechanism_replication/replacement_seed2/ce/n")
OUTPUT = Path("outputs/mechanism_replication/analysis/natural_direction/solar/ce")
PROTOCOL = Path("docs/research/solar_ce_natural_sampling_remaining_seeds_protocol.md")
K, ENDPOINT, SEEDS = 5, 4, (1, 2, 3, 4)
T95_N4 = 3.182446305


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_root(seed: int) -> Path:
    return REPLACEMENT_ROOT if seed == 2 else N_ROOT / f"seed_{seed}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
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


def read_source(seed: int) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    root = source_root(seed)
    manifest = json.loads((root / "manifest.json").read_text())
    required = ("no_backbone_training", "no_c_refit")
    if any(manifest.get(key) is not True for key in required):
        raise RuntimeError(f"invalid source manifest for seed {seed}: {root}")
    if seed != 2 and (manifest.get("a_c_identity") != "PASS" or manifest.get("no_a_refit") is not True):
        raise RuntimeError(f"unvalidated original-seed N source for seed {seed}: {root}")
    with np.load(root / "per_sample_arrays.npz", allow_pickle=False) as data:
        values = {key: data[key] for key in data.files}
    required_arrays = {"sample_ids", "labels"}
    required_arrays |= {f"{name}_{field}" for name in "anc" for field in ("probabilities", "l1", "predictive_mean")}
    absent = required_arrays.difference(values)
    if absent:
        raise RuntimeError(f"missing source arrays for seed {seed}: {sorted(absent)}")
    y, ids = values["labels"], values["sample_ids"]
    if y.shape != (28006,) or ids.shape != (28006,) or int((y == ENDPOINT).sum()) != 921:
        raise RuntimeError(f"invalid fixed readout population for seed {seed}")
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError(f"duplicate readout IDs for seed {seed}")
    for name in "anc":
        if values[f"{name}_probabilities"].shape != (28006, K) or values[f"{name}_l1"].shape != (28006,):
            raise RuntimeError(f"invalid A/N/C shape for seed {seed}, condition {name}")
    return manifest, values


def descriptive_label(improve_count: int) -> str:
    if improve_count == 4:
        return "consistent natural-direction response"
    if improve_count == 3:
        return "mixed/partial natural-direction response"
    return "natural-direction response not consistent"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not PROTOCOL.is_file():
        raise RuntimeError("frozen CE remaining-seeds protocol is absent")

    seed_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    mass_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []

    for seed in SEEDS:
        manifest, arrays = read_source(seed)
        y, ids = arrays["labels"], arrays["sample_ids"]
        values = {name: metric(y, arrays[f"{name}_l1"], arrays[f"{name}_probabilities"]) for name in "anc"}
        row: dict[str, object] = {
            "dataset": "solar", "objective": "ce", "seed": seed,
            "seed_provenance": "replacement" if seed == 2 else "original_validated",
            "endpoint_support": int((y == ENDPOINT).sum()),
        }
        for name in "anc":
            upper = name.upper()
            row.update({
                f"endpoint_mae_{upper}": values[name]["mae_4"],
                f"global_mae_{upper}": values[name]["global_mae"],
                f"macro_mae_{upper}": values[name]["macro_mae"],
                f"global_severe_{upper}": values[name]["global_severe"],
                f"endpoint_exact_routing_{upper}": int((arrays[f"{name}_l1"][y == ENDPOINT] == ENDPOINT).sum()),
            })
        row.update({
            "delta_NA_endpoint": row["endpoint_mae_N"] - row["endpoint_mae_A"],
            "delta_CA_endpoint": row["endpoint_mae_C"] - row["endpoint_mae_A"],
            "delta_NC_endpoint": row["endpoint_mae_N"] - row["endpoint_mae_C"],
        })
        seed_rows.append(row)
        for k in range(K):
            item: dict[str, object] = {
                "dataset": "solar", "objective": "ce", "seed": seed,
                "seed_provenance": row["seed_provenance"], "true_class": k,
            }
            for name in "anc":
                upper = name.upper()
                for key in ("mae", "recall", "severe", "p_end", "true_p", "predictive_mean"):
                    item[f"{upper}_{key}"] = values[name][f"{key}_{k}"]
            for key in ("mae", "recall", "severe", "p_end", "true_p", "predictive_mean"):
                item[f"N_minus_A_{key}"] = item[f"N_{key}"] - item[f"A_{key}"]
                item[f"C_minus_A_{key}"] = item[f"C_{key}"] - item[f"A_{key}"]
                item[f"C_minus_N_{key}"] = item[f"C_{key}"] - item[f"N_{key}"]
            class_rows.append(item)
            mass_rows.append({key: item[key] for key in (
                "dataset", "objective", "seed", "seed_provenance", "true_class",
                "A_p_end", "N_p_end", "C_p_end", "N_minus_A_p_end",
                "C_minus_A_p_end", "C_minus_N_p_end",
            )})
            mask = y == k
            for predicted in range(K):
                routing_rows.append({
                    "dataset": "solar", "objective": "ce", "seed": seed,
                    "seed_provenance": row["seed_provenance"], "true_class": k,
                    "predicted_class": predicted,
                    "A_count": int((arrays["a_l1"][mask] == predicted).sum()),
                    "N_count": int((arrays["n_l1"][mask] == predicted).sum()),
                    "C_count": int((arrays["c_l1"][mask] == predicted).sum()),
                })
        sources.append({
            "seed": seed, "seed_provenance": row["seed_provenance"],
            "root": str(source_root(seed)),
            "manifest": str(source_root(seed) / "manifest.json"),
            "per_sample_arrays": str(source_root(seed) / "per_sample_arrays.npz"),
            "per_sample_arrays_sha256": sha256(source_root(seed) / "per_sample_arrays.npz"),
            "cluster_job_id": manifest.get("cluster_job_id"),
        })

    delta = np.asarray([float(row["delta_NA_endpoint"]) for row in seed_rows])
    improve = int((delta < 0).sum())
    sd = float(delta.std(ddof=1))
    ci_half = T95_N4 * sd / np.sqrt(len(delta))
    means = {
        key: {name: float(np.mean([row[f"{key}_{name}"] for row in seed_rows])) for name in "ANC"}
        for key in ("endpoint_mae", "global_mae", "macro_mae", "global_severe")
    }
    synthesis_rows = [
        {"setting": "Retina CE", "n_improve_a": "2/4", "n_weaker_than_c": "4/4", "interpretation": "natural direction-only did not reproduce C; balanced sampling primary in this controlled setting"},
        {"setting": "Retina RPS", "n_improve_a": "2/4", "n_weaker_than_c": "4/4", "interpretation": "natural direction-only did not reproduce C; balanced sampling primary in this controlled setting"},
        {"setting": "Solar RPS", "n_improve_a": "3/4", "n_weaker_than_c": "4/4", "interpretation": "natural direction-only partly helped, but balanced sampling was important for the full response"},
        {"setting": "Solar CE", "n_improve_a": f"{improve}/4", "n_weaker_than_c": f"{int((np.asarray([row['delta_NC_endpoint'] for row in seed_rows]) > 0).sum())}/4", "interpretation": descriptive_label(improve)},
    ]
    summary = {
        "dataset": "Solar CE", "objective": "ce",
        "evaluation": "fixed 28,006-row readout; exact discrete L1 decisions",
        "seeds": [1, "replacement-2", 3, 4],
        "replacement_seed_2_only": True,
        "n_minus_a": {
            "per_seed": {str(row["seed"]): row["delta_NA_endpoint"] for row in seed_rows},
            "improving_seeds": improve, "mean": float(delta.mean()), "sample_sd": sd,
            "descriptive_t95": [float(delta.mean() - ci_half), float(delta.mean() + ci_half)],
            "label": descriptive_label(improve),
        },
        "n_weaker_than_c_endpoint": int((np.asarray([row["delta_NC_endpoint"] for row in seed_rows]) > 0).sum()),
        "means": means,
        "verdict": "C — SOLAR CE N FOLLOW-UP COMPLETE; NOT CONSISTENT" if improve <= 2 else ("B — SOLAR CE N FOLLOW-UP COMPLETE; MIXED/PARTIAL" if improve == 3 else "A — SOLAR CE N FOLLOW-UP COMPLETE; CONSISTENT"),
        "no_backbone_training": True, "no_a_refit": True, "no_c_refit": True,
        "no_tolerance_relaxation": True,
    }
    write_csv(OUTPUT / "per_seed_summary.csv", seed_rows)
    write_csv(OUTPUT / "per_class_metrics.csv", class_rows)
    write_csv(OUTPUT / "endpoint_mass_shift.csv", mass_rows)
    write_csv(OUTPUT / "routing_summary.csv", routing_rows)
    write_csv(OUTPUT / "cross_setting_synthesis.csv", synthesis_rows)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "script": "scripts/analyze_solar_ce_natural_direction.py",
        "protocol": str(PROTOCOL), "protocol_sha256": sha256(PROTOCOL),
        "git_commit": git_commit, "sources": sources,
        "source_feature_paths": {str(seed): f"outputs/mechanism_replication/features/solar/ce/seed_{seed}" for seed in (1, 3, 4)},
        "source_ac_paths": {str(seed): f"outputs/mechanism_replication/ac/solar/ce/seed_{seed}" for seed in (1, 3, 4)},
        "replacement_seed_2_source": str(REPLACEMENT_ROOT),
        "job_ids": [source["cluster_job_id"] for source in sources],
        "commands": ["python scripts/run_solar_natural_direction.py --objective ce --seed <1|3|4> --device cuda:0", "python scripts/analyze_solar_ce_natural_direction.py"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "no_fitting_in_analysis": True,
    }
    (OUTPUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
