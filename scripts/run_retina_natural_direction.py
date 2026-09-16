#!/usr/bin/env python3
"""Fit Retina-only natural-sampling direction heads on saved frozen features."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import natural_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions


FEATURE_ROOT = Path("outputs/mechanism_replication/features/retina")
AC_ROOT = Path("outputs/mechanism_replication/ac/retina")
OUT_ROOT = Path("outputs/mechanism_replication/natural_direction/retina")
LR, BATCH, EPOCHS, FOLDS, TOL = 1e-3, 64, 100, 5, 1e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    value = np.exp(shifted)
    return value / value.sum(axis=1, keepdims=True)


def l1(probabilities: np.ndarray) -> np.ndarray:
    return bayes_decisions(probabilities)["l1_bayes_decision"].astype(np.int64)


def assert_reference_identity(y: np.ndarray, a: dict[str, np.ndarray], c: dict[str, np.ndarray]) -> None:
    for name, values in (("A", a), ("C", c)):
        probability = values["probabilities"]
        if not np.array_equal(l1(probability), values["l1"]):
            raise RuntimeError(f"archived {name} exact-L1 decisions are inconsistent")
        if not np.allclose(probability @ np.arange(5), values["predictive_mean"], atol=1e-12):
            raise RuntimeError(f"archived {name} predictive means are inconsistent")
        if not np.array_equal(y, values["labels"]):
            raise RuntimeError(f"archived {name} labels are inconsistent")


def fit_natural(features: np.ndarray, labels: np.ndarray, weight: torch.Tensor, bias: torch.Tensor, fold: int, device: torch.device):
    x = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.long)
    head = DirectionOnlyLinear(weight.to(device), bias.to(device)).to(device)
    if any(parameter.requires_grad for parameter in (head.fixed_norms, head.fixed_bias)):
        raise RuntimeError("fixed norms/biases unexpectedly require gradients")
    optimizer = torch.optim.AdamW([head.direction], lr=LR, weight_decay=0.0)
    if len(optimizer.param_groups) != 1 or optimizer.param_groups[0]["params"] != [head.direction]:
        raise RuntimeError("optimizer contains parameters other than direction")
    generator = torch.Generator().manual_seed(10_000 + fold)
    histories = []
    before_bias = head.fixed_bias.detach().cpu().clone()
    before_norms = head.fixed_norms.detach().cpu().clone()
    observed_counts = np.zeros(5, dtype=np.int64)
    for epoch in range(1, EPOCHS + 1):
        losses, steps = [], 0
        for index in natural_batch_indices(len(y), BATCH, generator):
            observed_counts += np.bincount(y[index].numpy(), minlength=5)
            logits = head(x[index].to(device))
            loss = torch.nn.functional.cross_entropy(logits, y[index].to(device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu())); steps += 1
        histories.append({"epoch": epoch, "natural_ce": float(np.mean(losses)), "steps": steps})
    norm_error = float(head.max_norm_error().detach().cpu())
    bias_error = float((head.fixed_bias.detach().cpu() - before_bias).abs().max())
    source_norm_error = float((head.fixed_norms.detach().cpu() - before_norms).abs().max())
    if norm_error > TOL or bias_error > TOL or source_norm_error > TOL:
        raise RuntimeError(f"fixed-parameter violation: norm={norm_error}, bias={bias_error}, stored_norm={source_norm_error}")
    return head, histories, observed_counts, {"max_norm_error": norm_error, "max_bias_error": bias_error, "stored_norm_error": source_norm_error}


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    feature_dir = FEATURE_ROOT / args.objective / f"seed_{args.seed}"
    ac_dir = AC_ROOT / args.objective / f"seed_{args.seed}"
    output = OUT_ROOT / args.objective / f"seed_{args.seed}"
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    feature_manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    feature_path = feature_dir / "features.npz"
    if sha256(feature_path) != feature_manifest["archive_sha256"]: raise RuntimeError("feature archive SHA256 mismatch")
    if feature_manifest["a_output_reproduction"]["status"] != "PASS": raise RuntimeError("feature source lacks A reproduction PASS")
    with np.load(feature_path, allow_pickle=False) as data, np.load(ac_dir / "per_sample_arrays.npz", allow_pickle=False) as archived:
        x, y, ids, folds = data["features"], data["labels"], data["sample_ids"], data["folds"]
        if x.shape != (1080, 512) or x.dtype != np.float32 or not np.array_equal(ids, data["indices"]): raise RuntimeError("invalid frozen feature contract")
        if not np.array_equal(ids, archived["sample_ids"]) or not np.array_equal(y, archived["labels"]) or not np.array_equal(folds, archived["folds"]): raise RuntimeError("feature/A-C alignment mismatch")
        a = {"labels": archived["labels"], "logits": archived["a_logits"], "probabilities": archived["a_probabilities"], "l1": archived["a_l1"], "predictive_mean": archived["a_predictive_mean"]}
        c = {"labels": archived["labels"], "logits": archived["c_logits"], "probabilities": archived["c_probabilities"], "l1": archived["c_l1"], "predictive_mean": archived["c_predictive_mean"]}
    assert_reference_identity(y, a, c)
    head_a = torch.load(ac_dir / "A_original_head.pt", map_location="cpu", weights_only=False)
    weight, bias = head_a["weight"].float(), head_a["bias"].float()
    if tuple(weight.shape) != (5, 512) or tuple(bias.shape) != (5,): raise RuntimeError("invalid original A head")
    n_logits = np.empty_like(a["logits"]); history_rows=[]; constraints=[]; observed={}
    output.mkdir(parents=True); (output / "heads").mkdir(); (output / "histories").mkdir()
    for fold in range(FOLDS):
        fit, held = folds != fold, folds == fold
        head, history, counts, constraint = fit_natural(x[fit], y[fit], weight, bias, fold, args.device)
        with torch.inference_mode(): n_logits[held] = head(torch.as_tensor(x[held], device=args.device)).cpu().numpy()
        torch.save({"condition":"N_natural_direction_only","fold":fold,"state_dict":head.cpu().state_dict(),"fixed_norms":head.fixed_norms.cpu(),"fixed_bias":head.fixed_bias.cpu()}, output / "heads" / f"N_fold_{fold}.pt")
        save_csv(output / "histories" / f"fold_{fold}.csv", history)
        history_rows.extend([{"fold":fold, **row} for row in history]); observed[str(fold)] = counts.tolist(); constraints.append({"fold":fold, **constraint, "fit_rows":int(fit.sum()), "held_rows":int(held.sum())})
    n_probability=softmax(n_logits); n_l1=l1(n_probability); n_mean=n_probability @ np.arange(5)
    np.savez_compressed(output / "per_sample_arrays.npz", sample_ids=ids, labels=y, folds=folds, a_logits=a["logits"], n_logits=n_logits, c_logits=c["logits"], a_probabilities=a["probabilities"], n_probabilities=n_probability, c_probabilities=c["probabilities"], a_l1=a["l1"], n_l1=n_l1, c_l1=c["l1"], a_predictive_mean=a["predictive_mean"], n_predictive_mean=n_mean, c_predictive_mean=c["predictive_mean"])
    endpoint=y==4
    metrics={"endpoint_mae_A":float(np.abs(y[endpoint]-a["l1"][endpoint]).mean()),"endpoint_mae_N":float(np.abs(y[endpoint]-n_l1[endpoint]).mean()),"endpoint_mae_C":float(np.abs(y[endpoint]-c["l1"][endpoint]).mean())}
    metrics.update({"delta_NA_endpoint":metrics["endpoint_mae_N"]-metrics["endpoint_mae_A"],"delta_CA_endpoint":metrics["endpoint_mae_C"]-metrics["endpoint_mae_A"],"delta_NC_endpoint":metrics["endpoint_mae_N"]-metrics["endpoint_mae_C"]})
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2)+"\n")
    manifest={"dataset":"retina","objective":args.objective,"seed":args.seed,"condition":"N natural empirical-sampling direction-only","feature_archive":str(feature_path),"feature_archive_sha256":sha256(feature_path),"source_ac":str(ac_dir),"oof":"same archived five-fold assignment; 100 fixed epochs (C has no validation checkpoint selection)","optimizer":"AdamW","learning_rate":LR,"weight_decay":0.0,"batch_size":BATCH,"epochs":EPOCHS,"sampling":"natural shuffled empirical fold-training distribution; no weighting/rebalancing/adjustment","observed_draw_counts_across_100_epochs":observed,"constraints":constraints,"a_c_identity":"PASS","no_backbone_training":True,"no_feature_regeneration":True,"no_c_refit":True,"metrics":metrics}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--objective",choices=("ce","rps"),required=True); parser.add_argument("--seed",type=int,choices=(1,2,3,4),required=True); parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    args=parser.parse_args()
    if args.device=="cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    print(json.dumps(run(args),indent=2))

if __name__=="__main__": main()
