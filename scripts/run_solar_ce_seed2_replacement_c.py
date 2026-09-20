#!/usr/bin/env python3
"""Fit the replacement seed-2 balanced direction-only C head from frozen features."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_solar_ce_seed2_replacement_features import l1_decisions, runtime_gate, sha256, softmax
from run_ac_mechanism_replication import fit_direction_head, geometry, make_outputs


BASE = Path("outputs/solar/mechanism_replication/replacement_seed2/ce")
FEATURES, A_ROOT, OUT = BASE / "features", BASE / "a", BASE / "c"
K, ENDPOINT, TOL = 5, 4, 1e-6


def load_archive(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"feature checksum mismatch: {path}")
    with np.load(path, allow_pickle=False) as source:
        result = {key: source[key] for key in ("features", "labels", "sample_ids")}
    if result["features"].dtype != np.float32 or result["features"].ndim != 2 or result["features"].shape[1] != 512:
        raise RuntimeError(f"invalid replacement feature archive: {path}")
    if len(np.unique(result["sample_ids"])) != len(result["sample_ids"]):
        raise RuntimeError(f"duplicate replacement IDs: {path}")
    return result


def head_step_count(train_rows: int) -> int:
    return int(np.ceil(train_rows / 64)) * 100


def endpoint_h1(labels: np.ndarray, a_l1: np.ndarray, c_l1: np.ndarray) -> dict[str, float]:
    endpoint = labels == ENDPOINT
    a = float(np.abs(labels[endpoint] - a_l1[endpoint]).mean())
    c = float(np.abs(labels[endpoint] - c_l1[endpoint]).mean())
    return {"endpoint_mae_A": a, "endpoint_mae_C": c, "delta_H1_C_minus_A": c - a}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite replacement C: {OUT}")
    runtime = runtime_gate()
    feature_manifest = json.loads((FEATURES / "feature_manifest.json").read_text())
    train = load_archive(FEATURES / "train_features.npz", feature_manifest["splits"]["train"]["sha256"])
    validation = load_archive(FEATURES / "val_features.npz", feature_manifest["splits"]["val"]["sha256"])
    evaluation = load_archive(FEATURES / "eval_features.npz", feature_manifest["splits"]["eval"]["sha256"])
    if [len(item["labels"]) for item in (train, validation, evaluation)] != [45047, 2431, 28006]:
        raise RuntimeError("replacement feature counts mismatch")
    if int((evaluation["labels"] == ENDPOINT).sum()) != 921:
        raise RuntimeError("replacement endpoint support mismatch")
    if set(train["sample_ids"]) & set(validation["sample_ids"]) or set(train["sample_ids"]) & set(evaluation["sample_ids"]) or set(validation["sample_ids"]) & set(evaluation["sample_ids"]):
        raise RuntimeError("replacement split overlap")
    head_a = torch.load(A_ROOT / "A_original_head.pt", map_location="cpu", weights_only=False)
    weight, bias = head_a["weight"].float(), head_a["bias"].float()
    with np.load(A_ROOT / "per_sample_arrays.npz", allow_pickle=False) as source:
        a = {key: source[key] for key in ("sample_ids", "labels", "a_logits", "a_probabilities", "a_mode", "a_l1", "a_predictive_mean")}
    if not np.array_equal(a["sample_ids"], evaluation["sample_ids"]) or not np.array_equal(a["labels"], evaluation["labels"]):
        raise RuntimeError("replacement A/evaluation alignment failure")
    device = torch.device("cuda:0")
    head, history, init_error, norm_error, bias_error = fit_direction_head(train["features"], train["labels"], weight, bias, seed=10_000, device=device)
    if max(norm_error, bias_error) > TOL:
        raise RuntimeError(f"replacement C fixed-parameter failure: norm={norm_error}, bias={bias_error}")
    with torch.no_grad():
        c_logits = head(torch.as_tensor(evaluation["features"], dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)
    values = make_outputs(evaluation["labels"], a["a_logits"], c_logits)
    c_mode = values["c_probabilities"].argmax(axis=1).astype(np.int64)
    c_l1 = values["c_l1"].astype(np.int64)
    if not np.array_equal(c_l1, l1_decisions(values["c_probabilities"])):
        raise RuntimeError("replacement C exact-L1 calculation mismatch")
    g = geometry(train["features"], train["labels"], evaluation["features"])
    h1 = endpoint_h1(evaluation["labels"], a["a_l1"], c_l1)
    OUT.mkdir(parents=True)
    torch.save({"condition": "C_replacement_balanced_direction_only", "state_dict": head.cpu().state_dict(), "fixed_norms": head.fixed_norms.cpu(), "fixed_bias": head.fixed_bias.cpu()}, OUT / "C_head.pt")
    write_csv(OUT / "training_history.csv", history)
    np.savez_compressed(OUT / "per_sample_arrays.npz", sample_ids=evaluation["sample_ids"], labels=evaluation["labels"], folds=np.full(len(evaluation["labels"]), -1, dtype=np.int64), nearest_centroid=g["nearest_centroid"], endpoint_adj_margin=g["endpoint_adj_margin"], generic_centroid_margin=g["generic_centroid_margin"], a_logits=a["a_logits"], c_logits=c_logits, a_probabilities=a["a_probabilities"], c_probabilities=values["c_probabilities"], a_mode=a["a_mode"], c_mode=c_mode, a_l1=a["a_l1"], c_l1=c_l1, a_predictive_mean=a["a_predictive_mean"], c_predictive_mean=values["c_predictive_mean"], a_inward_shrinkage=values["a_inward_shrinkage"], c_inward_shrinkage=values["c_inward_shrinkage"])
    manifest = {"dataset": "solar", "objective": "ce", "logical_seed": 2, "run_type": "replacement", "condition": "C balanced direction-only", "feature_manifest": str(FEATURES / "feature_manifest.json"), "feature_manifest_sha256": sha256(FEATURES / "feature_manifest.json"), "source_A": str(A_ROOT), "source_A_head_sha256": sha256(A_ROOT / "A_original_head.pt"), "source_A_arrays_sha256": sha256(A_ROOT / "per_sample_arrays.npz"), "protocol": {"objective": "cross_entropy", "sampling": "replacement class-balanced", "optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0, "batch_size": 64, "epochs": 100, "initialization": "replacement A direction", "fixed_original_norms": True, "fixed_original_biases": True, "validation_selection": "none; frozen archived C terminal-epoch protocol"}, "constraints": {"initialization_replay_max_abs_error": init_error, "max_norm_error": norm_error, "max_bias_error": bias_error, "only_directions_updated": True, "backbone_in_optimizer": False}, "runtime": runtime, "head_steps": head_step_count(len(train["labels"])), "h1": h1, "per_sample_arrays_sha256": sha256(OUT / "per_sample_arrays.npz"), "c_head_sha256": sha256(OUT / "C_head.pt"), "no_backbone_training": True, "created_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
