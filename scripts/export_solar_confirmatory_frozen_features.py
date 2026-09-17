#!/usr/bin/env python3
"""Deterministically export Solar train/validation/readout frozen features.

This is Phase A only: it never constructs an optimizer or changes a checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_7a_solar_3ch import EXPECTED, Solar, manifest, source_channels
from phase3_8_solar_confirmation import verify_stats
from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.metrics.decision import bayes_decisions


FEATURE_ROOT = Path("outputs/mechanism_replication/features/solar")
AC_ROOT = Path("outputs/mechanism_replication/ac/solar")
INTEGRITY = Path("outputs/solar/mechanism_replication/backbone_integrity_audit.json")
TOL_LOGIT, TOL_PROB = 2e-5, 2e-6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_state(path: Path) -> dict[str, torch.Tensor]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved.get("model_state_dict", saved))
    if tuple(state.get("fc.weight", torch.empty(0)).shape) != (5, 512) or tuple(state.get("fc.bias", torch.empty(0)).shape) != (5,):
        raise RuntimeError("checkpoint is not a ResNet18 512->5 state")
    return state


def capture(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    features: list[torch.Tensor] = []
    def hook(_module, inputs):
        features.append(inputs[0].detach().cpu())
    handle = model.fc.register_forward_pre_hook(hook)
    logits, labels, ids = [], [], []
    model.eval()
    with torch.inference_mode():
        for image, label, sample_id in loader:
            logits.append(model(image.to(device, non_blocking=True)).cpu())
            labels.append(label.reshape(-1).cpu())
            ids.append(sample_id.reshape(-1).cpu())
    handle.remove()
    result = {"features": torch.cat(features).numpy().astype(np.float32), "labels": torch.cat(labels).numpy().astype(np.int64), "sample_ids": torch.cat(ids).numpy().astype(np.int64), "model_logits": torch.cat(logits).numpy().astype(np.float32)}
    if result["features"].ndim != 2 or result["features"].shape[1] != 512:
        raise RuntimeError(f"invalid feature shape {result['features'].shape}")
    if len(np.unique(result["sample_ids"])) != len(result["sample_ids"]):
        raise RuntimeError("duplicate sample IDs in deterministic export")
    return result


def compare_a_eval(data: dict[str, np.ndarray], archived: Path, weight: torch.Tensor, bias: torch.Tensor) -> dict[str, object]:
    if not archived.is_file():
        return {"status": "UNAVAILABLE", "reason": f"missing {archived}"}
    with np.load(archived, allow_pickle=False) as z:
        if not np.array_equal(data["sample_ids"], z["sample_ids"]) or not np.array_equal(data["labels"], z["labels"]):
            return {"status": "MISMATCH", "reason": "sample ID or label alignment"}
        logits = F.linear(torch.from_numpy(data["features"]), weight, bias).numpy()
        prob = probabilities(logits)
        expected_l1 = bayes_decisions(prob)["l1_bayes_decision"].astype(np.int64)
        archived_l1 = z["a_l1"]
        logit_error = float(np.abs(logits - z["a_logits"]).max())
        logit_mean_error = float(np.abs(logits - z["a_logits"]).mean())
        logit_median_error = float(np.median(np.abs(logits - z["a_logits"])))
        probability_error = float(np.abs(prob - z["a_probabilities"]).max())
        mean_error = float(np.abs(prob @ np.arange(5) - z["a_predictive_mean"]).max())
        exact_l1 = bool(np.array_equal(expected_l1, archived_l1))
        mode = prob.argmax(axis=1).astype(np.int64)
        archived_mode = z["a_probabilities"].argmax(axis=1).astype(np.int64)
        mode_differences = int((mode != archived_mode).sum())
        l1_differences = int((expected_l1 != archived_l1).sum())
    ok = logit_error <= TOL_LOGIT and probability_error <= TOL_PROB and exact_l1
    return {"status": "PASS" if ok else "MISMATCH", "logit_max_abs_error": logit_error, "logit_mean_abs_error": logit_mean_error, "logit_median_abs_error": logit_median_error, "probability_max_abs_error": probability_error, "predictive_mean_max_abs_error": mean_error, "mode_difference_count": mode_differences, "exact_l1_difference_count": l1_differences, "exact_l1_match": exact_l1, "tolerances": {"logits": TOL_LOGIT, "probabilities": TOL_PROB}}


def compare_c_eval(data: dict[str, np.ndarray], archived: Path, c_head: Path, weight: torch.Tensor, bias: torch.Tensor) -> dict[str, object]:
    if not archived.is_file() or not c_head.is_file():
        return {"status": "UNAVAILABLE"}
    saved = torch.load(c_head, map_location="cpu", weights_only=False)
    head = DirectionOnlyLinear(weight, bias)
    head.load_state_dict(saved["state_dict"], strict=True)
    with np.load(archived, allow_pickle=False) as z, torch.inference_mode():
        logits = head(torch.from_numpy(data["features"])).numpy()
        prob = probabilities(logits)
        l1 = bayes_decisions(prob)["l1_bayes_decision"].astype(np.int64)
        logit_error = float(np.abs(logits - z["c_logits"]).max())
        probability_error = float(np.abs(prob - z["c_probabilities"]).max())
        exact_l1 = bool(np.array_equal(l1, z["c_l1"]))
    ok = logit_error <= TOL_LOGIT and probability_error <= TOL_PROB and exact_l1
    return {"status": "PASS" if ok else "MISMATCH", "logit_max_abs_error": logit_error, "probability_max_abs_error": probability_error, "exact_l1_match": exact_l1}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objective", required=True, choices=("ce", "rps")); p.add_argument("--seed", required=True, type=int, choices=(1,2,3,4))
    p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--stats", type=Path, required=True)
    p.add_argument("--root", default="/scratch/users/jhong36/data/surya-bench-224.zarr"); p.add_argument("--index", default="/scratch/users/jhong36/data")
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--workers", type=int, default=4); p.add_argument("--device", default="cuda:0")
    p.add_argument("--strict-fp32", action="store_true", help="disable TF32 and require float32-compatible inference")
    a = p.parse_args(); out = FEATURE_ROOT / a.objective / f"seed_{a.seed}"
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}")
    if not torch.cuda.is_available() or not str(a.device).startswith("cuda"): raise RuntimeError("Phase A requires an allocated CUDA device")
    audit = json.loads(INTEGRITY.read_text()); record = next((x for x in audit["records"] if x["objective"] == a.objective and x["seed"] == a.seed), None)
    if not record or record["status"] != "READY" or Path(record["checkpoint_path"]) != a.checkpoint: raise RuntimeError("checkpoint integrity identity is ambiguous")
    if not a.checkpoint.is_file(): raise FileNotFoundError(a.checkpoint)
    if a.strict_fp32:
        if a.objective != "ce": raise RuntimeError("strict-FP32 regeneration is authorized only for CE")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    stats = verify_stats(a.stats); source_channels(a.root)
    frames = [manifest(Path(a.index) / f"{name}.csv", expected) for name, expected in zip(("train", "validation", "test"), EXPECTED)]
    datasets = [Solar(frame, a.root, (stats["mean"], stats["std"]), augment=False) for frame in frames]
    loaders = [DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True) for ds in datasets]
    state = load_state(a.checkpoint); model = resnet18(weights=None); model.fc = torch.nn.Linear(512, 5); model.load_state_dict(state, strict=True); model.to(a.device).eval()
    outputs = [capture(model, loader, torch.device(a.device)) for loader in loaders]
    ids = [set(x["sample_ids"].tolist()) for x in outputs]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]: raise RuntimeError("split sample-ID overlap")
    ac = AC_ROOT / a.objective / f"seed_{a.seed}"; arrays = ac / "per_sample_arrays.npz"; a_head = ac / "A_original_head.pt"
    saved_a = torch.load(a_head, map_location="cpu", weights_only=False)
    weight, bias = saved_a["weight"].float(), saved_a["bias"].float()
    if not torch.equal(weight, state["fc.weight"].cpu()) or not torch.equal(bias, state["fc.bias"].cpu()): raise RuntimeError("A head and checkpoint head differ")
    a_check = compare_a_eval(outputs[2], arrays, weight, bias)
    c_check = compare_c_eval(outputs[2], arrays, ac / "C_head.pt", weight, bias)
    if a_check["status"] != "PASS": raise RuntimeError(f"mandatory A identity gate failed: {a_check}")
    out.mkdir(parents=True)
    file_info = {}
    for name, data in zip(("train", "val", "eval"), outputs):
        path = out / f"{name}_features.npz"; np.savez_compressed(path, features=data["features"], labels=data["labels"], sample_ids=data["sample_ids"])
        file_info[name] = {"path": str(path), "sha256": sha256(path), "rows": int(len(data["labels"])), "class_counts": np.bincount(data["labels"], minlength=5).tolist()}
    manifest_out = {"dataset":"solar", "objective":a.objective, "seed":a.seed, "checkpoint_path":str(a.checkpoint), "checkpoint_sha256":sha256(a.checkpoint), "selected_epoch":record["selected_epoch"], "checkpoint_integrity":"READY", "gpu_model":torch.cuda.get_device_name(0), "architecture":"torchvision ResNet18 weights=None; fc Linear(512,5)", "feature_definition":"float32 512-D input to model.fc after average pooling/flatten", "class_count":5, "preprocessing":{"channels":stats["channels"],"channel_indices":stats["channel_indices"],"transform":stats["transform"],"normalization_artifact":str(a.stats),"normalization_sha256":sha256(a.stats)}, "splits":file_info, "extraction":{"script":"scripts/export_solar_confirmatory_frozen_features.py","device":str(a.device),"cluster_job_id":os.getenv("SLURM_JOB_ID","unset"),"batch_size":a.batch_size,"strict_fp32":a.strict_fp32,"tf32_matmul_enabled":torch.backends.cuda.matmul.allow_tf32,"tf32_cudnn_enabled":torch.backends.cudnn.allow_tf32,"float32_matmul_precision":torch.get_float32_matmul_precision(),"autocast":False,"model_dtype":str(next(model.parameters()).dtype),"input_dtype":"torch.float32","deterministic_inference":"model.eval(); torch.inference_mode(); augment=False; shuffle=False; no optimizer; no training"}, "a_output_reproduction":a_check, "c_output_replay":c_check, "no_training":True, "created_utc":datetime.now(timezone.utc).isoformat()}
    (out / "feature_manifest.json").write_text(json.dumps(manifest_out, indent=2) + "\n")
    print(json.dumps(manifest_out, indent=2))

if __name__ == "__main__": main()
