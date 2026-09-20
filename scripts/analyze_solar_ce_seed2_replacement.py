#!/usr/bin/env python3
"""Read-only A/C/N and frozen-H2a analysis for replacement Solar CE seed 2."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_mechanism_replication_h1_h2 import (
    h2a_vectors, loocv_mse, model_matrix, ordinary_least_squares,
    sample_standardize, t_interval,
)


BASE = Path("outputs/solar/mechanism_replication/replacement_seed2/ce")
N_ROOT, OUT = BASE / "n", BASE / "analysis"
H1_OLD = Path("outputs/mechanism_replication/analysis/solar/ce/h1_per_seed.csv")
H2_OLD = Path("outputs/mechanism_replication/analysis/solar/ce/h2a_per_seed.csv")
K, ENDPOINT = 5, 4


def write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def metric(labels: np.ndarray, decision: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    error = np.abs(labels - decision)
    result = {"global_mae": float(error.mean()), "macro_mae": float(np.mean([error[labels == k].mean() for k in range(K)])), "global_severe": float((error >= 2).mean())}
    for k in range(K):
        mask = labels == k
        result.update({f"mae_{k}": float(error[mask].mean()), f"recall_{k}": float((decision[mask] == k).mean()), f"severe_{k}": float((error[mask] >= 2).mean()), f"p_end_{k}": float(probability[mask, ENDPOINT].mean()), f"true_p_{k}": float(probability[mask, k].mean()), f"predictive_mean_{k}": float((probability[mask] @ np.arange(K)).mean())})
    return result


def h2a(arrays: dict[str, np.ndarray]) -> dict[str, float | list[float]]:
    y_c, y_a, generic, ordinal = h2a_vectors(arrays)
    za, _, _ = sample_standardize(y_a); zg, _, _ = sample_standardize(generic); zo, _, _ = sample_standardize(ordinal)
    coefficients, _, standard_errors = ordinary_least_squares(model_matrix(za, zg, zo), y_c)
    beta, se = float(coefficients[-1]), float(standard_errors[-1])
    mse0, mse1 = loocv_mse(y_c, y_a, generic, None), loocv_mse(y_c, y_a, generic, ordinal)
    return {"beta_ord": beta, "hc3_se": se, "hc3_ci95": [beta - 1.959963984540054 * se, beta + 1.959963984540054 * se], "mse_loo_m0": mse0, "mse_loo_m1": mse1, "delta_loo": mse0 - mse1}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite replacement analysis: {OUT}")
    with np.load(N_ROOT / "per_sample_arrays.npz", allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    required = {"sample_ids", "labels", "a_l1", "c_l1", "n_l1", "a_probabilities", "c_probabilities", "n_probabilities", "a_predictive_mean", "c_predictive_mean", "endpoint_adj_margin", "generic_centroid_margin"}
    if not required.issubset(arrays) or arrays["labels"].shape != (28006,) or int((arrays["labels"] == ENDPOINT).sum()) != 921 or len(np.unique(arrays["sample_ids"])) != 28006:
        raise RuntimeError("replacement analysis input gate failed")
    values = {name: metric(arrays["labels"], arrays[f"{name}_l1"], arrays[f"{name}_probabilities"]) for name in "acn"}
    seed_row: dict[str, object] = {"dataset": "solar", "objective": "ce", "seed": "replacement_2"}
    for name in "acn":
        upper = name.upper(); seed_row.update({f"endpoint_mae_{upper}": values[name]["mae_4"], f"global_mae_{upper}": values[name]["global_mae"], f"macro_mae_{upper}": values[name]["macro_mae"], f"global_severe_{upper}": values[name]["global_severe"]})
    seed_row.update({"delta_NA_endpoint": seed_row["endpoint_mae_N"] - seed_row["endpoint_mae_A"], "delta_CA_endpoint": seed_row["endpoint_mae_C"] - seed_row["endpoint_mae_A"], "delta_NC_endpoint": seed_row["endpoint_mae_N"] - seed_row["endpoint_mae_C"]})
    class_rows: list[dict[str, object]] = []; mass_rows: list[dict[str, object]] = []; routing_rows: list[dict[str, object]] = []
    for k in range(K):
        item: dict[str, object] = {"dataset": "solar", "objective": "ce", "seed": "replacement_2", "true_class": k}
        for name in "acn":
            upper = name.upper()
            for field in ("mae", "recall", "severe", "p_end", "true_p", "predictive_mean"):
                item[f"{upper}_{field}"] = values[name][f"{field}_{k}"]
        for field in ("mae", "recall", "severe", "p_end", "true_p", "predictive_mean"):
            item[f"N_minus_A_{field}"] = item[f"N_{field}"] - item[f"A_{field}"]
            item[f"C_minus_A_{field}"] = item[f"C_{field}"] - item[f"A_{field}"]
            item[f"C_minus_N_{field}"] = item[f"C_{field}"] - item[f"N_{field}"]
        class_rows.append(item)
        mass_rows.append({field: item[field] for field in ("dataset", "objective", "seed", "true_class", "A_p_end", "N_p_end", "C_p_end", "N_minus_A_p_end", "C_minus_A_p_end", "C_minus_N_p_end")})
        mask = arrays["labels"] == k
        for predicted in range(K):
            routing_rows.append({"dataset": "solar", "objective": "ce", "seed": "replacement_2", "true_class": k, "predicted_class": predicted, "A_count": int((arrays["a_l1"][mask] == predicted).sum()), "N_count": int((arrays["n_l1"][mask] == predicted).sum()), "C_count": int((arrays["c_l1"][mask] == predicted).sum())})
    replacement_h2a = h2a(arrays)
    old_h1 = [row for row in read_csv(H1_OLD) if row["seed"] in {"1", "3", "4"}]
    old_h2 = [row for row in read_csv(H2_OLD) if row["seed"] in {"1", "3", "4"}]
    h1_rows = [{"seed": row["seed"], "a_mae": float(row["a_mae"]), "c_mae": float(row["c_mae"]), "delta_mae_c_minus_a": float(row["delta_mae_c_minus_a"]), "source": "original_valid"} for row in old_h1] + [{"seed": "replacement_2", "a_mae": seed_row["endpoint_mae_A"], "c_mae": seed_row["endpoint_mae_C"], "delta_mae_c_minus_a": seed_row["delta_CA_endpoint"], "source": "replacement"}]
    h2_rows = [{"seed": row["seed"], "beta_ord": float(row["beta_ord"]), "delta_loo": float(row["delta_loo"]), "source": "original_valid"} for row in old_h2] + [{"seed": "replacement_2", "beta_ord": replacement_h2a["beta_ord"], "delta_loo": replacement_h2a["delta_loo"], "source": "replacement"}]
    h1_delta = [float(row["delta_mae_c_minus_a"]) for row in h1_rows]; h2_beta = [float(row["beta_ord"]) for row in h2_rows]; h2_loo = [float(row["delta_loo"]) for row in h2_rows]
    cross = {"h1": {"seeds": [row["seed"] for row in h1_rows], "negative_seed_count": int(sum(v < 0 for v in h1_delta)), **t_interval(h1_delta), "verdict": "REPLICATED" if sum(v < 0 for v in h1_delta) == 4 else "PARTIAL" if sum(v < 0 for v in h1_delta) == 3 else "NOT REPLICATED"}, "h2a": {"seeds": [row["seed"] for row in h2_rows], "beta_ord_negative_count": int(sum(v < 0 for v in h2_beta)), "delta_loo_positive_count": int(sum(v > 0 for v in h2_loo)), "mean_delta_loo": float(np.mean(h2_loo)), "beta_ord_cross_seed": t_interval(h2_beta), "verdict": "STRONG" if sum(v < 0 for v in h2_beta) == 4 and sum(v > 0 for v in h2_loo) >= 3 and float(np.mean(h2_loo)) > 0 else "PARTIAL" if sum(v < 0 for v in h2_beta) >= 3 else "NOT SUPPORTED"}, "n_four_seed_consistency": "UNAVAILABLE: CE N artifacts for original seeds 1, 3, and 4 do not exist and this replacement task does not authorize modifying them"}
    OUT.mkdir(parents=True)
    write(OUT / "per_seed_summary.csv", [seed_row]); write(OUT / "per_class_metrics.csv", class_rows); write(OUT / "endpoint_mass_shift.csv", mass_rows); write(OUT / "routing_summary.csv", routing_rows); write(OUT / "cross_seed_h1.csv", h1_rows); write(OUT / "cross_seed_h2a.csv", h2_rows)
    summary = {"replacement_seed2": {"metrics": seed_row, "h2a": replacement_h2a}, "cross_seed_ac": cross, "n_cross_seed_status": "UNAVAILABLE", "no_training": True, "no_head_fitting": True}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "analysis_manifest.json").write_text(json.dumps({"script": "scripts/analyze_solar_ce_seed2_replacement.py", "input_n_arrays": str(N_ROOT / "per_sample_arrays.npz"), "old_ac_h1": str(H1_OLD), "old_ac_h2a": str(H2_OLD), "created_utc": datetime.now(timezone.utc).isoformat(), "no_training": True, "no_head_fitting": True}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
