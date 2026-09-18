#!/usr/bin/env python3
"""One non-retrying historical-path Solar CE seed-2 feature replay.

This script is intentionally locked to the recovered historical CE evaluation
recipe.  It refuses a non-V100 allocation before constructing data loaders,
captures 512-D pre-fc features for train/validation/readout into a temporary
tree, and promotes that tree only after the frozen A-output identity gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
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
from run_ac_mechanism_replication import seed_everything


OBJECTIVE = "ce"
SEED = 2
HISTORICAL_BATCH = 128
HISTORICAL_WORKERS = 4
ROOT = "/scratch/users/jhong36/data/surya-bench-224.zarr"
INDEX = "/scratch/users/jhong36/data"
STATS = Path("outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json")
CHECKPOINT = Path("outputs/solar/mechanism_replication/backbones/ce/seed_2/selected_checkpoint.pt")
ARCHIVED = Path("outputs/mechanism_replication/ac/solar/ce/seed_2/per_sample_arrays.npz")
A_HEAD = Path("outputs/mechanism_replication/ac/solar/ce/seed_2/A_original_head.pt")
INTEGRITY = Path("outputs/solar/mechanism_replication/backbone_integrity_audit.json")
TMP = Path("outputs/mechanism_replication/features/solar/ce/seed_2_historical_replay_tmp")
CANONICAL = Path("outputs/mechanism_replication/features/solar/ce/seed_2")
TOL_LOGIT = 2e-5
TOL_PROB = 2e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probabilities(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def l1_decisions(probability: np.ndarray) -> np.ndarray:
    classes = np.arange(probability.shape[1])
    risks = (probability[:, None, :] * np.abs(classes[None, None, :] - classes[None, :, None])).sum(axis=2)
    return risks.argmin(axis=1).astype(np.int64)


def full_identity_gate(
    *, sample_ids: np.ndarray, labels: np.ndarray, logits: np.ndarray,
    archived_ids: np.ndarray, archived_labels: np.ndarray,
    archived_logits: np.ndarray, archived_probabilities: np.ndarray,
    archived_l1: np.ndarray,
) -> dict[str, object]:
    ids_match = bool(np.array_equal(sample_ids, archived_ids))
    labels_match = bool(np.array_equal(labels, archived_labels))
    if not ids_match or not labels_match:
        return {"status": "MISMATCH", "sample_ids_match": ids_match, "labels_match": labels_match}
    probability = probabilities(logits)
    logit_error = np.abs(logits.astype(np.float64) - archived_logits.astype(np.float64))
    probability_error = np.abs(probability - archived_probabilities.astype(np.float64))
    mode_difference_count = int((probability.argmax(axis=1) != archived_probabilities.argmax(axis=1)).sum())
    exact_l1_difference_count = int((l1_decisions(probability) != archived_l1).sum())
    metrics: dict[str, object] = {
        "sample_ids_match": ids_match,
        "labels_match": labels_match,
        "logit_max_abs_error": float(logit_error.max()),
        "logit_mean_abs_error": float(logit_error.mean()),
        "probability_max_abs_error": float(probability_error.max()),
        "probability_mean_abs_error": float(probability_error.mean()),
        "mode_difference_count": mode_difference_count,
        "exact_l1_difference_count": exact_l1_difference_count,
        "tolerances": {"logits": TOL_LOGIT, "probabilities": TOL_PROB},
    }
    metrics["status"] = "PASS" if (
        metrics["logit_max_abs_error"] <= TOL_LOGIT
        and metrics["probability_max_abs_error"] <= TOL_PROB
        and mode_difference_count == 0
        and exact_l1_difference_count == 0
    ) else "MISMATCH"
    return metrics


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        return f"UNAVAILABLE: {error}"


def require_v100() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("V100 allocation required; CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    if "V100" not in name.upper():
        raise RuntimeError(f"V100 allocation required; refusing inference on {name}")
    properties = torch.cuda.get_device_properties(0)
    return {
        "hostname": command_output(["hostname"]),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", "unset"),
        "partition": os.getenv("SLURM_JOB_PARTITION", "unset"),
        "account": os.getenv("SLURM_JOB_ACCOUNT", "unset"),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", "unset"),
        "nvidia_smi_l": command_output(["nvidia-smi", "-L"]),
        "nvidia_smi_query": command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv"]),
        "torch_device_name": name,
        "torch_device_properties": {"name": properties.name, "major": properties.major, "minor": properties.minor, "total_memory": properties.total_memory},
        "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast": False, "model_dtype": "torch.float32", "input_dtype": "torch.float32",
    }


def load_checkpoint() -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    integrity = json.loads(INTEGRITY.read_text())
    record = next((item for item in integrity["records"] if item["objective"] == OBJECTIVE and item["seed"] == SEED), None)
    if record is None or record["status"] != "READY" or Path(record["checkpoint_path"]) != CHECKPOINT:
        raise RuntimeError("seed-2 checkpoint identity is not READY")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved.get("state_dict", saved.get("model_state_dict", saved))
    if tuple(state.get("fc.weight", torch.empty(0)).shape) != (5, 512) or tuple(state.get("fc.bias", torch.empty(0)).shape) != (5,):
        raise RuntimeError("checkpoint is not ResNet18 Linear(512,5)")
    head = torch.load(A_HEAD, map_location="cpu", weights_only=False)
    if not torch.equal(head["weight"].float(), state["fc.weight"].float()) or not torch.equal(head["bias"].float(), state["fc.bias"].float()):
        raise RuntimeError("archived A head differs from original checkpoint classifier")
    return state, record


def capture(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        features.append(inputs[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(hook)
    model.eval()
    with torch.no_grad():
        for image, label, identifier in loader:
            logits.append(model(image.to(device, non_blocking=True)).detach().cpu())
            labels.append(label.reshape(-1).cpu())
            sample_ids.append(identifier.reshape(-1).cpu())
    handle.remove()
    result = {
        "features": torch.cat(features).numpy().astype(np.float32),
        "model_logits": torch.cat(logits).numpy().astype(np.float32),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "sample_ids": torch.cat(sample_ids).numpy().astype(np.int64),
    }
    if result["features"].shape != (len(loader.dataset), 512):
        raise RuntimeError(f"feature shape is not (rows,512): {result['features'].shape}")
    if len(np.unique(result["sample_ids"])) != len(result["sample_ids"]):
        raise RuntimeError("duplicate sample IDs in replay")
    return result


def save_features(out: Path, role: str, data: dict[str, np.ndarray]) -> dict[str, object]:
    path = out / f"{role}_features.npz"
    np.savez_compressed(path, features=data["features"], labels=data["labels"], sample_ids=data["sample_ids"])
    return {"path": str(path), "sha256": sha256(path), "rows": int(len(data["labels"])), "class_counts": np.bincount(data["labels"], minlength=5).tolist()}


def main() -> None:
    if TMP.exists() or CANONICAL.exists():
        raise FileExistsError("refusing to overwrite temporary or canonical seed-2 feature archive")
    runtime = require_v100()
    state, integrity = load_checkpoint()
    stats = verify_stats(STATS)
    source_channels(ROOT)
    frames = [manifest(Path(INDEX) / f"{name}.csv", expected) for name, expected in zip(("train", "validation", "test"), EXPECTED)]
    datasets = [Solar(frame, ROOT, (stats["mean"], stats["std"]), augment=False) for frame in frames]
    loaders = [DataLoader(dataset, batch_size=HISTORICAL_BATCH, shuffle=False, num_workers=HISTORICAL_WORKERS, pin_memory=True, drop_last=False) for dataset in datasets]
    seed_everything(SEED)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 5)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict state loading failed: {loaded}")
    model = model.to("cuda:0").eval()
    data = [capture(model, loader, torch.device("cuda:0")) for loader in loaders]
    split_ids = [set(item["sample_ids"].tolist()) for item in data]
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        raise RuntimeError("replay split sample IDs overlap")
    with np.load(ARCHIVED, allow_pickle=False) as archived:
        a_head = torch.load(A_HEAD, map_location="cpu", weights_only=False)
        feature_logits = F.linear(torch.from_numpy(data[2]["features"]), a_head["weight"].float(), a_head["bias"].float()).numpy()
        identity = full_identity_gate(
            sample_ids=data[2]["sample_ids"], labels=data[2]["labels"], logits=feature_logits,
            archived_ids=archived["sample_ids"], archived_labels=archived["labels"],
            archived_logits=archived["a_logits"], archived_probabilities=archived["a_probabilities"], archived_l1=archived["a_l1"],
        )
    TMP.mkdir(parents=True)
    splits = {role: save_features(TMP, role, values) for role, values in zip(("train", "val", "eval"), data)}
    summary = {
        "dataset": "solar", "objective": OBJECTIVE, "seed": SEED,
        "scope": "single bounded seed-2 historical replay; no training, head fitting, A/C fitting, or tolerance change",
        "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_objective": "ce", "selected_epoch": integrity["selected_epoch"], "strict_state_load": True,
        "architecture": "torchvision ResNet18 weights=None; fc Linear(512,5)", "feature_definition": "float32 512-D input to model.fc after average pooling/flatten",
        "preprocessing": {"channels": stats["channels"], "channel_indices": stats["channel_indices"], "transform": stats["transform"], "normalization_artifact": str(STATS), "normalization_sha256": sha256(STATS)},
        "historical_recipe": {"evaluation_batch_size": HISTORICAL_BATCH, "num_workers": HISTORICAL_WORKERS, "pin_memory": True, "persistent_workers": False, "prefetch_factor": 2, "shuffle": False, "drop_last": False, "sampler": "SequentialSampler", "eval_mode": True, "context": "torch.no_grad", "historical_backend_flags": "UNKNOWN; no retained direct record", "historical_source_revision": "strongly supported 1d80e095d7a552f0a89369fa708d67c1463511f7; exact historical checkout unknown"},
        "runtime": runtime, "splits": splits, "a_output_identity": identity,
        "promotion": "PROMOTED" if identity["status"] == "PASS" else "NOT_PROMOTED", "no_training": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (TMP / "feature_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    if identity["status"] == "PASS":
        shutil.move(str(TMP), str(CANONICAL))
        print(json.dumps({**summary, "promoted_to": str(CANONICAL)}, indent=2))
        return
    print(json.dumps(summary, indent=2))
    raise RuntimeError(f"frozen A-output identity gate failed; temporary archive retained at {TMP}: {identity}")


if __name__ == "__main__":
    main()
