#!/usr/bin/env python3
"""Export replacement seed-2 Solar CE features and define replacement A only."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
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


ROOT = "/scratch/users/jhong36/data/surya-bench-224.zarr"
INDEX = "/scratch/users/jhong36/data"
STATS = Path("outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json")
BASE = Path("outputs/solar/mechanism_replication/replacement_seed2/ce")
BACKBONE = BASE / "backbone"
FEATURES = BASE / "features"
A_ROOT = BASE / "a"
CHECKPOINT = BACKBONE / "selected_checkpoint.pt"
EXPECTED_ENV = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "cuda": "12.4", "cudnn": 90100}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def l1_decisions(probability: np.ndarray) -> np.ndarray:
    classes = np.arange(probability.shape[1])
    risks = (probability[:, None, :] * np.abs(classes[None, None, :] - classes[None, :, None])).sum(axis=2)
    return risks.argmin(axis=1).astype(np.int64)


def a_logits_from_features(features: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        return F.linear(torch.from_numpy(features), weight.cpu().float(), bias.cpu().float()).numpy().astype(np.float32)


def runtime_gate() -> dict[str, object]:
    import torchvision
    if not torch.cuda.is_available() or "V100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("replacement feature export requires a V100 allocation")
    actual = {"torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if actual != EXPECTED_ENV:
        raise RuntimeError(f"replacement environment mismatch: {actual} != {EXPECTED_ENV}")
    query = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"], text=True).strip()
    return {"hostname": subprocess.check_output(["hostname"], text=True).strip(), "job_id": os.getenv("SLURM_JOB_ID", "unset"), "partition": os.getenv("SLURM_JOB_PARTITION", "unset"), "account": os.getenv("SLURM_JOB_ACCOUNT", "unset"), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "unset"), "nvidia_smi_l": subprocess.check_output(["nvidia-smi", "-L"], text=True).strip(), "nvidia_smi_query": query, "gpu": torch.cuda.get_device_name(0), "device_properties": str(torch.cuda.get_device_properties(0)), "python": platform.python_version(), **actual, "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32, "matmul_precision": torch.get_float32_matmul_precision(), "autocast": False, "model_dtype": "torch.float32", "input_dtype": "torch.float32"}


def capture(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []
    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        features.append(inputs[0].detach().cpu())
    handle = model.fc.register_forward_pre_hook(hook)
    model.eval()
    with torch.no_grad():
        for image, label, sample_id in loader:
            logits.append(model(image.to(device, non_blocking=True)).detach().cpu())
            labels.append(label.reshape(-1).cpu())
            ids.append(sample_id.reshape(-1).cpu())
    handle.remove()
    result = {"features": torch.cat(features).numpy().astype(np.float32), "model_logits": torch.cat(logits).numpy().astype(np.float32), "labels": torch.cat(labels).numpy().astype(np.int64), "sample_ids": torch.cat(ids).numpy().astype(np.int64)}
    if result["features"].shape != (len(loader.dataset), 512):
        raise RuntimeError(f"invalid 512-D feature archive: {result['features'].shape}")
    if len(np.unique(result["sample_ids"])) != len(result["sample_ids"]):
        raise RuntimeError("duplicate sample IDs")
    return result


def save_feature(role: str, data: dict[str, np.ndarray]) -> dict[str, object]:
    path = FEATURES / f"{role}_features.npz"
    np.savez_compressed(path, features=data["features"], labels=data["labels"], sample_ids=data["sample_ids"])
    return {"path": str(path), "sha256": sha256(path), "rows": int(len(data["labels"])), "class_counts": np.bincount(data["labels"], minlength=5).tolist()}


def main() -> None:
    if FEATURES.exists() or A_ROOT.exists() or not CHECKPOINT.is_file():
        raise RuntimeError("replacement feature/A namespace is occupied or backbone checkpoint is missing")
    runtime = runtime_gate()
    source_channels(ROOT)
    stats = verify_stats(STATS)
    frames = [manifest(Path(INDEX) / f"{name}.csv", expected) for name, expected in zip(("train", "validation", "test"), EXPECTED)]
    datasets = [Solar(frame, ROOT, (stats["mean"], stats["std"]), augment=False) for frame in frames]
    loaders = [DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, drop_last=False) for dataset in datasets]
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    model = resnet18(weights=None); model.fc = torch.nn.Linear(512, 5)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {loaded}")
    model = model.to("cuda:0").eval()
    data = [capture(model, loader, torch.device("cuda:0")) for loader in loaders]
    ids = [set(item["sample_ids"].tolist()) for item in data]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise RuntimeError("replacement feature split overlap")
    expected_rows = [45047, 2431, 28006]
    if [len(item["labels"]) for item in data] != expected_rows or int((data[2]["labels"] == 4).sum()) != 921:
        raise RuntimeError("replacement split-count gate failed")
    FEATURES.mkdir(parents=True); A_ROOT.mkdir(parents=True)
    split_info = {role: save_feature(role, values) for role, values in zip(("train", "val", "eval"), data)}
    weight, bias = state["fc.weight"].detach().cpu().float(), state["fc.bias"].detach().cpu().float()
    a_logits = a_logits_from_features(data[2]["features"], weight, bias)
    replay_error = float(np.abs(a_logits - data[2]["model_logits"]).max())
    if replay_error > 2e-5:
        raise RuntimeError(f"replacement A feature/head replay failed: {replay_error}")
    a_probability = softmax(a_logits)
    a_l1 = l1_decisions(a_probability)
    a_mode = a_probability.argmax(axis=1).astype(np.int64)
    torch.save({"condition": "A_replacement_original", "weight": weight, "bias": bias, "norms": weight.norm(dim=1), "directions": weight / weight.norm(dim=1, keepdim=True), "checkpoint_sha256": sha256(CHECKPOINT)}, A_ROOT / "A_original_head.pt")
    np.savez_compressed(A_ROOT / "per_sample_arrays.npz", sample_ids=data[2]["sample_ids"], labels=data[2]["labels"], a_logits=a_logits, a_probabilities=a_probability, a_mode=a_mode, a_l1=a_l1, a_predictive_mean=a_probability @ np.arange(5))
    summary = {"dataset": "solar", "objective": "ce", "logical_seed": 2, "run_type": "replacement", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "selected_epoch": saved["selected_epoch"], "validation_loss": saved["validation_loss"], "feature_dimension": 512, "splits": split_info, "preprocessing": {"channels": stats["channels"], "channel_indices": stats["channel_indices"], "transform": stats["transform"], "normalization_artifact": str(STATS), "normalization_sha256": sha256(STATS)}, "runtime": runtime, "a_definition": "replacement checkpoint original fc head", "a_feature_head_replay_max_abs_error": replay_error, "no_training": True, "no_head_fitting": True, "created_utc": datetime.now(timezone.utc).isoformat()}
    (FEATURES / "feature_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (A_ROOT / "manifest.json").write_text(json.dumps({**summary, "a_arrays_sha256": sha256(A_ROOT / "per_sample_arrays.npz"), "a_head_sha256": sha256(A_ROOT / "A_original_head.pt")}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
