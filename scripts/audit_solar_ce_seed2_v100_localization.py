#!/usr/bin/env python3
"""Non-promoting V100-only localization audit for Solar CE seed 2.

It compares a fixed 64-row readout subset at batch sizes 32 and 64. Archived
Solar A arrays contain logits/probabilities but not transformed inputs or
penultimate features, so those earlier stages are audited for deterministic
batch-size agreement; logits are the first stage directly comparable with the
archived historical path.
"""
from __future__ import annotations

import hashlib
import json
import os
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


N = 64
SEED = 2
ROOT = "/scratch/users/jhong36/data/surya-bench-224.zarr"
INDEX = "/scratch/users/jhong36/data"
STATS = Path("outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json")
CHECKPOINT = Path("outputs/solar/mechanism_replication/backbones/ce/seed_2/selected_checkpoint.pt")
ARCHIVED = Path("outputs/mechanism_replication/ac/solar/ce/seed_2/per_sample_arrays.npz")
OUT = Path("outputs/mechanism_replication/audits/solar_ce_seed2_v100_localization")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def error_summary(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    return {"max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean())}


def summarize_stages(reference: np.ndarray, stages: dict[str, np.ndarray]) -> dict[str, object]:
    results = {name: error_summary(value, reference) for name, value in stages.items()}
    earliest = next((name for name, value in results.items() if value["max_abs_error"] != 0.0), None)
    return {"earliest_nonzero_stage": earliest, "stages": results}


def capture(model: torch.nn.Module, dataset: Solar, batch_size: int, device: torch.device) -> dict[str, np.ndarray]:
    loader = DataLoader(Subset(dataset, range(N)), batch_size=batch_size, shuffle=False, num_workers=0)
    inputs: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[torch.Tensor] = []

    def hook(_module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        features.append(values[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(hook)
    model.eval()
    with torch.inference_mode():
        for image, label, sample_id in loader:
            inputs.append(image.cpu())
            logits.append(model(image.to(device)).cpu())
            labels.append(label.cpu())
            ids.append(sample_id.cpu())
    handle.remove()
    return {
        "normalized_inputs": torch.cat(inputs).numpy().astype(np.float32),
        "features": torch.cat(features).numpy().astype(np.float32),
        "logits": torch.cat(logits).numpy().astype(np.float32),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "sample_ids": torch.cat(ids).numpy().astype(np.int64),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic audit {OUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("V100-only diagnostic requires an allocated CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source_channels(ROOT)
    stats = verify_stats(STATS)
    frame = manifest(Path(INDEX) / "test.csv", EXPECTED[2])
    dataset = Solar(frame, ROOT, (stats["mean"], stats["std"]), augment=False)
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved["state_dict"]
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 5)
    loaded = model.load_state_dict(state, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise RuntimeError(f"strict checkpoint load failed: {loaded}")
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    run64 = capture(model, dataset, 64, device)
    run32 = capture(model, dataset, 32, device)
    with np.load(ARCHIVED, allow_pickle=False) as archived:
        archived_ids = archived["sample_ids"][:N]
        archived_labels = archived["labels"][:N]
        archived_logits = archived["a_logits"][:N]
    for run in (run64, run32):
        if not np.array_equal(run["sample_ids"], archived_ids) or not np.array_equal(run["labels"], archived_labels):
            raise RuntimeError("fixed 64-row archived sample alignment failed")
    if run64["features"].shape != (N, 512):
        raise RuntimeError(f"unexpected penultimate feature shape {run64['features'].shape}")
    batch_summary = {
        "normalized_inputs": error_summary(run64["normalized_inputs"], run32["normalized_inputs"]),
        "features": error_summary(run64["features"], run32["features"]),
        "logits": error_summary(run64["logits"], run32["logits"]),
    }
    archived_logit_summary = error_summary(run64["logits"], archived_logits)
    OUT.mkdir(parents=True)
    np.savez_compressed(
        OUT / "seed2_fixed64_diagnostic.npz",
        sample_ids=run64["sample_ids"], labels=run64["labels"],
        normalized_inputs_batch64=run64["normalized_inputs"], normalized_inputs_batch32=run32["normalized_inputs"],
        features_batch64=run64["features"], features_batch32=run32["features"],
        logits_batch64=run64["logits"], logits_batch32=run32["logits"], archived_logits=archived_logits,
    )
    summary = {
        "scope": "fixed first 64 archived CE seed-2 readout rows; diagnostic only; no feature promotion, head fitting, checkpoint change, or training",
        "runtime": {
            "slurm_job_id": os.getenv("SLURM_JOB_ID", "unset"),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "tf32_matmul_enabled": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn_enabled": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "autocast": False, "model_dtype": str(next(model.parameters()).dtype), "input_dtype": "torch.float32",
        },
        "identity": {"checkpoint": str(CHECKPOINT), "checkpoint_sha256": sha256(CHECKPOINT), "normalization": str(STATS), "normalization_sha256": sha256(STATS), "strict_state_load": True, "sample_alignment": True},
        "archived_availability": {"normalized_inputs": False, "features": False, "logits": True},
        "batch32_vs_64": batch_summary,
        "archived_logit_comparison_batch64": archived_logit_summary,
        "earliest_archived_comparable_divergence": "logits" if archived_logit_summary["max_abs_error"] != 0.0 else None,
        "interpretation": "Archived CE A arrays do not preserve transformed inputs or penultimate features. Batch-size agreement tests deterministic regenerated input/feature stages; logits are the first directly archived-comparable stage.",
        "artifact": {"path": str(OUT / "seed2_fixed64_diagnostic.npz"), "sha256": None},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary["artifact"]["sha256"] = sha256(OUT / "seed2_fixed64_diagnostic.npz")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
