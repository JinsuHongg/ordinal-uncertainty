#!/usr/bin/env python3
"""Evidence-led, non-promoting Solar CE seed-2 recipe audit.

The primary candidate mirrors the archived A/C runner: seed_everything(2),
the original 128-sample readout batch, four pinned DataLoader workers, eval
mode, and ``torch.no_grad``.  The sole control applies strict FP32 to that
same recorded path.  No model, head, or canonical feature archive is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3_7a_solar_3ch import EXPECTED, Solar, manifest, source_channels
from phase3_8_solar_confirmation import verify_stats
from run_ac_mechanism_replication import seed_everything


N_COMPARE = 64
HISTORICAL_BATCH = 128
HISTORICAL_WORKERS = 4
ROOT = "/scratch/users/jhong36/data/surya-bench-224.zarr"
INDEX = "/scratch/users/jhong36/data"
STATS = Path("outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json")
CHECKPOINT = Path("outputs/solar/mechanism_replication/backbones/ce/seed_2/selected_checkpoint.pt")
ARCHIVED = Path("outputs/mechanism_replication/ac/solar/ce/seed_2/per_sample_arrays.npz")
OUT = Path("outputs/mechanism_replication/audits/solar_ce_historical_recipe")
TOL_LOGIT, TOL_PROB = 2e-5, 2e-6


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


def l1_decisions(probabilities: np.ndarray) -> np.ndarray:
    classes = np.arange(probabilities.shape[1])
    loss = (probabilities[:, None, :] * np.abs(classes[None, None, :] - classes[None, :, None])).sum(axis=2)
    return loss.argmin(axis=1).astype(np.int64)


def identity_metrics(logits: np.ndarray, archived_logits: np.ndarray, archived_probabilities: np.ndarray, archived_l1: np.ndarray) -> dict[str, object]:
    probabilities = softmax(logits)
    l1 = l1_decisions(probabilities)
    logit_delta = np.abs(logits.astype(np.float64) - archived_logits.astype(np.float64))
    probability_delta = np.abs(probabilities - archived_probabilities)
    result = {
        "logit_max_abs_error": float(logit_delta.max()),
        "logit_mean_abs_error": float(logit_delta.mean()),
        "probability_max_abs_error": float(probability_delta.max()),
        "probability_mean_abs_error": float(probability_delta.mean()),
        "mode_difference_count": int((probabilities.argmax(axis=1) != archived_probabilities.argmax(axis=1)).sum()),
        "exact_l1_difference_count": int((l1 != archived_l1).sum()),
    }
    result["status"] = "PASS" if result["logit_max_abs_error"] <= TOL_LOGIT and result["probability_max_abs_error"] <= TOL_PROB and result["exact_l1_difference_count"] == 0 else "MISMATCH"
    return result


def full_population_identity(*, sample_ids: np.ndarray, labels: np.ndarray, logits: np.ndarray, archived_ids: np.ndarray, archived_labels: np.ndarray, archived_logits: np.ndarray, archived_probabilities: np.ndarray, archived_l1: np.ndarray) -> dict[str, object]:
    ids_match = bool(np.array_equal(sample_ids, archived_ids))
    labels_match = bool(np.array_equal(labels, archived_labels))
    if not ids_match or not labels_match:
        return {"status": "MISMATCH", "sample_ids_match": ids_match, "labels_match": labels_match}
    result = identity_metrics(logits, archived_logits, archived_probabilities, archived_l1)
    result.update({"sample_ids_match": ids_match, "labels_match": labels_match})
    return result


def error_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {"max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean())}


def capture_first_historical_batch(model: torch.nn.Module, dataset: Solar, device: torch.device) -> dict[str, np.ndarray]:
    loader = DataLoader(
        Subset(dataset, range(HISTORICAL_BATCH)), batch_size=HISTORICAL_BATCH,
        shuffle=False, num_workers=HISTORICAL_WORKERS, pin_memory=True,
        drop_last=False,
    )
    features: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        features.append(values[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(hook)
    model.eval()
    with torch.no_grad():
        image, labels, sample_ids = next(iter(loader))
        logits = model(image.to(device, non_blocking=True)).detach().cpu()
    handle.remove()
    return {
        "features": torch.cat(features).numpy().astype(np.float32),
        "logits": logits.numpy().astype(np.float32),
        "labels": labels.numpy().astype(np.int64),
        "sample_ids": sample_ids.numpy().astype(np.int64),
    }


def capture_full_historical_readout(model: torch.nn.Module, dataset: Solar, device: torch.device) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=HISTORICAL_BATCH, shuffle=False, num_workers=HISTORICAL_WORKERS, pin_memory=True, drop_last=False)
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    sample_ids: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        features.append(values[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(hook)
    model.eval()
    with torch.no_grad():
        for image, label, identifier in loader:
            logits.append(model(image.to(device, non_blocking=True)).detach().cpu())
            labels.append(label.reshape(-1).detach().cpu())
            sample_ids.append(identifier.reshape(-1).detach().cpu())
    handle.remove()
    captured_features = torch.cat(features).numpy().astype(np.float32)
    if captured_features.shape != (len(dataset), 512):
        raise RuntimeError(f"historical full feature shape mismatch: {captured_features.shape}")
    return {"logits": torch.cat(logits).numpy().astype(np.float32), "labels": torch.cat(labels).numpy().astype(np.int64), "sample_ids": torch.cat(sample_ids).numpy().astype(np.int64)}


def gpu_driver() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_settings() -> dict[str, object]:
    return {
        "gpu": torch.cuda.get_device_name(0), "driver": gpu_driver(),
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast": False, "model_dtype": "torch.float32", "input_dtype": "torch.float32",
    }


def configure(candidate: str) -> None:
    if candidate == "historical_recorded_defaults":
        return
    if candidate == "historical_recorded_strict_fp32_control":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        return
    raise ValueError(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--full-readout", action="store_true", help="evaluate only the full archived seed-2 readout after subset compatibility passes")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit output {args.out}")
    if not torch.cuda.is_available():
        raise RuntimeError("historical recipe audit requires a V100 CUDA allocation")
    source_channels(ROOT)
    stats = verify_stats(STATS)
    frame = manifest(Path(INDEX) / "test.csv", EXPECTED[2])
    dataset = Solar(frame, ROOT, (stats["mean"], stats["std"]), augment=False)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    with np.load(ARCHIVED, allow_pickle=False) as archive:
        archived_ids = archive["sample_ids"][:N_COMPARE]
        archived_labels = archive["labels"][:N_COMPARE]
        archived_logits = archive["a_logits"][:N_COMPARE]
        archived_probabilities = archive["a_probabilities"][:N_COMPARE]
        archived_l1 = archive["a_l1"][:N_COMPARE]
        full_archived = {key: archive[key] for key in ("sample_ids", "labels", "a_logits", "a_probabilities", "a_l1")}
    device = torch.device("cuda:0")
    if args.full_readout:
        seed_everything(2)
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 5)
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"strict state load failed: {loaded}")
        run = capture_full_historical_readout(model.to(device).eval(), dataset, device)
        result = full_population_identity(sample_ids=run["sample_ids"], labels=run["labels"], logits=run["logits"], archived_ids=full_archived["sample_ids"], archived_labels=full_archived["labels"], archived_logits=full_archived["a_logits"], archived_probabilities=full_archived["a_probabilities"], archived_l1=full_archived["a_l1"])
        args.out.mkdir(parents=True)
        summary = {"scope": "full seed-2 archived readout only; exact recorded historical recipe; no train/validation capture, feature promotion, fitting, training, or tolerance change", "historical_recipe": {"extract_batch": HISTORICAL_BATCH, "workers": HISTORICAL_WORKERS, "pin_memory": True, "shuffle": False, "drop_last": False, "persistent_workers": False, "prefetch_factor": 2, "eval_mode": True, "context": "torch.no_grad", "seed_everything": 2}, "runtime": runtime_settings(), "checkpoint_sha256": sha256(CHECKPOINT), "normalization_sha256": sha256(STATS), "identity": result, "frozen_tolerances": {"logits": TOL_LOGIT, "probabilities": TOL_PROB}, "created_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.getenv("SLURM_JOB_ID", "unset")}
        (args.out / "full_seed2_identity.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        return
    candidates: dict[str, dict[str, object]] = {}
    for candidate in ("historical_recorded_defaults", "historical_recorded_strict_fp32_control"):
        configure(candidate)
        seed_everything(2)
        model = resnet18(weights=None)
        model.fc = torch.nn.Linear(512, 5)
        loaded = model.load_state_dict(state, strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise RuntimeError(f"strict state load failed: {loaded}")
        model = model.to(device).eval()
        first = capture_first_historical_batch(model, dataset, device)
        seed_everything(2)
        second = capture_first_historical_batch(model, dataset, device)
        if not np.array_equal(first["sample_ids"][:N_COMPARE], archived_ids) or not np.array_equal(first["labels"][:N_COMPARE], archived_labels):
            raise RuntimeError("first 64 rows of historical batch do not align to archived output")
        candidates[candidate] = {
            "recipe": {"extract_batch": HISTORICAL_BATCH, "workers": HISTORICAL_WORKERS, "pin_memory": True, "shuffle": False, "drop_last": False, "persistent_workers": False, "prefetch_factor": 2, "eval_mode": not model.training, "context": "torch.no_grad", "seed_everything": 2},
            "runtime": runtime_settings(),
            "identity": identity_metrics(first["logits"][:N_COMPARE], archived_logits, archived_probabilities, archived_l1),
            "repeatability": {"features": error_summary(first["features"], second["features"]), "logits": error_summary(first["logits"], second["logits"])},
        }
        del model
    args.out.mkdir(parents=True)
    summary = {
        "scope": "seed-2 fixed 64 archived IDs embedded in the first historical 128-row batch; no feature promotion, fitting, training, or tolerance change",
        "historical_recipe_evidence": {"slurm_job": "4399479_1", "script": "scripts/slurm_solar_ac_mechanism_replication.sbatch", "source_commit": "1d80e095d7a552f0a89369fa708d67c1463511f7", "exact_command": "--extract-batch 128 --workers 4 --device cuda:0"},
        "checkpoint_sha256": sha256(CHECKPOINT), "normalization_sha256": sha256(STATS),
        "candidates": candidates,
        "frozen_tolerances": {"logits": TOL_LOGIT, "probabilities": TOL_PROB},
        "created_utc": datetime.now(timezone.utc).isoformat(), "slurm_job_id": os.getenv("SLURM_JOB_ID", "unset"),
    }
    (args.out / "subset_recipe_matrix.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
