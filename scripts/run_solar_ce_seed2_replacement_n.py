#!/usr/bin/env python3
"""Fit replacement seed-2 N with natural sampling only."""
from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import natural_batch_indices
from export_solar_ce_seed2_replacement_features import l1_decisions, runtime_gate, sha256, softmax


BASE = Path("outputs/solar/mechanism_replication/replacement_seed2/ce")
FEATURES, A_ROOT, C_ROOT, OUT = BASE / "features", BASE / "a", BASE / "c", BASE / "n"
K, BATCH, EPOCHS, LR, TOL = 5, 64, 100, 1e-3, 1e-6


def expected_draw_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=K) * EPOCHS


def load_feature(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    if sha256(path) != expected_hash:
        raise RuntimeError(f"replacement feature hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in ("features", "labels", "sample_ids")}
    if values["features"].dtype != np.float32 or values["features"].shape[1] != 512 or len(np.unique(values["sample_ids"])) != len(values["sample_ids"]):
        raise RuntimeError(f"invalid replacement feature input: {path}")
    return values


def fit_natural(features: np.ndarray, labels: np.ndarray, weight: torch.Tensor, bias: torch.Tensor, device: torch.device) -> tuple[DirectionOnlyLinear, list[dict[str, object]], np.ndarray, dict[str, float]]:
    x, y = torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)
    head = DirectionOnlyLinear(weight.to(device), bias.to(device)).to(device)
    optimizer = torch.optim.AdamW([head.direction], lr=LR, weight_decay=0.0)
    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["params"] != [head.direction]:
        raise RuntimeError("replacement N optimizer contains parameters besides directions")
    before_bias, before_norms = head.fixed_bias.detach().cpu().clone(), head.fixed_norms.detach().cpu().clone()
    generator = torch.Generator().manual_seed(10_000)
    observed = np.zeros(K, dtype=np.int64); history: list[dict[str, object]] = []
    for epoch in range(1, EPOCHS + 1):
        losses: list[float] = []; steps = 0
        for index in natural_batch_indices(len(y), BATCH, generator):
            observed += np.bincount(y[index].numpy(), minlength=K)
            loss = torch.nn.functional.cross_entropy(head(x[index].to(device)), y[index].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu())); steps += 1
        history.append({"epoch": epoch, "natural_ce": float(np.mean(losses)), "steps": steps})
    expected = expected_draw_counts(labels)
    if not np.array_equal(observed, expected):
        raise RuntimeError(f"natural sampling count failure: observed={observed.tolist()} expected={expected.tolist()}")
    constraints = {"max_norm_error": float(head.max_norm_error().detach().cpu()), "max_bias_error": float((head.fixed_bias.detach().cpu() - before_bias).abs().max()), "stored_norm_error": float((head.fixed_norms.detach().cpu() - before_norms).abs().max())}
    if max(constraints.values()) > TOL:
        raise RuntimeError(f"replacement N fixed parameter failure: {constraints}")
    return head, history, observed, constraints


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite replacement N: {OUT}")
    runtime = runtime_gate()
    manifest = json.loads((FEATURES / "feature_manifest.json").read_text())
    train = load_feature(FEATURES / "train_features.npz", manifest["splits"]["train"]["sha256"])
    val = load_feature(FEATURES / "val_features.npz", manifest["splits"]["val"]["sha256"])
    evaluation = load_feature(FEATURES / "eval_features.npz", manifest["splits"]["eval"]["sha256"])
    if [len(item["labels"]) for item in (train, val, evaluation)] != [45047, 2431, 28006] or int((evaluation["labels"] == 4).sum()) != 921:
        raise RuntimeError("replacement N split-count gate failed")
    if set(train["sample_ids"]) & set(val["sample_ids"]) or set(train["sample_ids"]) & set(evaluation["sample_ids"]) or set(val["sample_ids"]) & set(evaluation["sample_ids"]):
        raise RuntimeError("replacement N split overlap")
    head_a = torch.load(A_ROOT / "A_original_head.pt", map_location="cpu", weights_only=False)
    weight, bias = head_a["weight"].float(), head_a["bias"].float()
    with np.load(C_ROOT / "per_sample_arrays.npz", allow_pickle=False) as source:
        reference = {key: source[key] for key in ("sample_ids", "labels", "a_logits", "c_logits", "a_probabilities", "c_probabilities", "a_mode", "c_mode", "a_l1", "c_l1", "a_predictive_mean", "c_predictive_mean", "nearest_centroid", "endpoint_adj_margin", "generic_centroid_margin")}
    if not np.array_equal(reference["sample_ids"], evaluation["sample_ids"]) or not np.array_equal(reference["labels"], evaluation["labels"]):
        raise RuntimeError("replacement N A/C/evaluation alignment failure")
    head, history, observed, constraints = fit_natural(train["features"], train["labels"], weight, bias, torch.device("cuda:0"))
    with torch.no_grad():
        n_logits = head(torch.as_tensor(evaluation["features"], dtype=torch.float32, device="cuda:0")).cpu().numpy().astype(np.float32)
    n_probability = softmax(n_logits); n_mode = n_probability.argmax(axis=1).astype(np.int64); n_l1 = l1_decisions(n_probability); y = evaluation["labels"]
    endpoint = y == 4
    metrics = {"endpoint_mae_A": float(np.abs(y[endpoint] - reference["a_l1"][endpoint]).mean()), "endpoint_mae_C": float(np.abs(y[endpoint] - reference["c_l1"][endpoint]).mean()), "endpoint_mae_N": float(np.abs(y[endpoint] - n_l1[endpoint]).mean())}
    metrics.update({"delta_NA_endpoint": metrics["endpoint_mae_N"] - metrics["endpoint_mae_A"], "delta_CA_endpoint": metrics["endpoint_mae_C"] - metrics["endpoint_mae_A"], "delta_NC_endpoint": metrics["endpoint_mae_N"] - metrics["endpoint_mae_C"]})
    OUT.mkdir(parents=True)
    torch.save({"condition": "N_replacement_natural_direction_only", "state_dict": head.cpu().state_dict(), "fixed_norms": head.fixed_norms.cpu(), "fixed_bias": head.fixed_bias.cpu()}, OUT / "N_head.pt")
    write_history(OUT / "training_history.csv", history)
    np.savez_compressed(OUT / "per_sample_arrays.npz", **reference, n_logits=n_logits, n_probabilities=n_probability, n_mode=n_mode, n_l1=n_l1, n_predictive_mean=n_probability @ np.arange(K))
    config = {"condition": "N natural empirical-sampling direction-only", "adaptation_objective": "cross_entropy", "optimizer": "AdamW", "learning_rate": LR, "weight_decay": 0.0, "batch_size": BATCH, "epochs": EPOCHS, "initialization": "replacement A direction", "fixed_original_norms": True, "fixed_original_biases": True, "sampling": "one shuffled empirical frozen-training pass per epoch; no class weighting, balanced sampler, over/undersampling, logit adjustment, or bias adjustment", "validation_selection": "none; exact frozen Solar C terminal-epoch protocol"}
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    out_manifest = {"dataset": "solar", "objective": "ce", "logical_seed": 2, "run_type": "replacement", "condition": "N natural direction-only", "feature_manifest_sha256": sha256(FEATURES / "feature_manifest.json"), "source_A_head_sha256": sha256(A_ROOT / "A_original_head.pt"), "source_C_head_sha256": sha256(C_ROOT / "C_head.pt"), "config": config, "train_rows": len(train["labels"]), "validation_rows": len(val["labels"]), "eval_rows": len(evaluation["labels"]), "train_class_counts": np.bincount(train["labels"], minlength=K).tolist(), "observed_draw_counts": observed.tolist(), "constraints": constraints, "head_steps": int(sum(int(row["steps"]) for row in history)), "runtime": runtime, "metrics": metrics, "n_head_sha256": sha256(OUT / "N_head.pt"), "per_sample_arrays_sha256": sha256(OUT / "per_sample_arrays.npz"), "no_backbone_training": True, "no_c_refit": True, "created_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "manifest.json").write_text(json.dumps(out_manifest, indent=2) + "\n")
    print(json.dumps(out_manifest, indent=2))


if __name__ == "__main__":
    main()
