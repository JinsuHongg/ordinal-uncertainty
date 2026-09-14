#!/usr/bin/env python3
"""Verify native-28 RetinaMNIST RPS checkpoint identity by frozen-logit replay.

This is an integrity/provenance utility.  It only reads historical checkpoints,
saved predictions, and the local official RetinaMNIST NPZ; it never trains or
calculates scientific performance metrics.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np
import torch

from ordinal_uncertainty.models.resnet import make_resnet18


ROOT = Path("outputs/retinamnist/native28/phase2_model_comparison/rps")
DATASET = Path("data/medmnist/retinamnist.npz")
OUTPUT = Path("outputs/retinamnist/mechanism_replication_provenance/rps_replay_audit.json")
SEEDS = (1, 2, 3, 4)
TOLERANCE = 1e-4
TRANSFORMS = ("to_tensor", "normalize_half")


def replay_status(replays: dict[str, dict[str, object]]) -> str:
    """Return a conservative status from exact identity and logit checks."""
    passing = [
        name
        for name, result in replays.items()
        if result["ids_match"] and result["labels_match"] and result["max_abs_logit_error"] <= TOLERANCE
    ]
    if len(passing) == 1:
        return "COMPATIBLE"
    if len(passing) > 1:
        return "AMBIGUOUS"
    return "INCOMPATIBLE"


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"empty saved-prediction file: {path}")
    ids = np.asarray([int(row["sample_id"]) for row in rows], dtype=np.int64)
    labels = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    logits = np.asarray([ast.literal_eval(row["logits"]) for row in rows], dtype=np.float32)
    return ids, labels, logits


def images_for_transform(images: np.ndarray, name: str) -> torch.Tensor:
    values = torch.from_numpy(images).permute(0, 3, 1, 2).to(torch.float32).div_(255.0)
    if name == "normalize_half":
        return values.sub_(0.5).div_(0.5)
    if name == "to_tensor":
        return values
    raise ValueError(f"unsupported transform candidate: {name}")


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved.get("state_dict"))
    if state is None:
        raise ValueError(f"checkpoint lacks state dictionary: {checkpoint}")
    model = make_resnet18(5)
    model.load_state_dict(state, strict=True)
    if model.conv1.kernel_size != (3, 3) or model.conv1.stride != (1, 1):
        raise ValueError("checkpoint reconstruction does not have a 3x3 stride-1 stem")
    if not isinstance(model.maxpool, torch.nn.Identity):
        raise ValueError("checkpoint reconstruction does not remove maxpool")
    if model.fc.in_features != 512 or tuple(model.fc.weight.shape) != (5, 512):
        raise ValueError("checkpoint reconstruction does not have a 5x512 final head")
    return model.to(device).eval()


def infer_logits(model: torch.nn.Module, images: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            outputs.append(model(images[start : start + batch_size].to(device)).cpu())
    return torch.cat(outputs).numpy().astype(np.float32)


def audit_seed(seed: int, images: np.ndarray, labels: np.ndarray, device: torch.device) -> dict[str, object]:
    artifact = ROOT / f"seed_{seed}_artifact_complete"
    checkpoint = artifact / "best_checkpoint.pt"
    prediction_path = artifact / "evaluation/predictions.csv"
    config_path = artifact / "config.json"
    smoke_path = artifact / "smoke.json"
    required = (checkpoint, prediction_path, config_path, smoke_path, artifact / "training_history.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"seed {seed} missing required artifacts: {missing}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    ids, saved_labels, reference_logits = load_reference(prediction_path)
    expected_ids = np.arange(labels.size, dtype=np.int64)
    id_match = bool(np.array_equal(ids, expected_ids))
    label_match = bool(np.array_equal(saved_labels, labels))
    if reference_logits.shape != (labels.size, 5):
        raise ValueError(f"seed {seed} reference logits shape is {reference_logits.shape}, expected {(labels.size, 5)}")

    model = load_model(checkpoint, device)
    replays: dict[str, dict[str, object]] = {}
    for transform in TRANSFORMS:
        logits = infer_logits(model, images_for_transform(images, transform), device, int(config["batch_size"]))
        replays[transform] = {
            "ids_match": id_match,
            "labels_match": label_match,
            "max_abs_logit_error": float(np.max(np.abs(logits - reference_logits))),
        }
    status = replay_status(replays)
    verified = [name for name, result in replays.items() if result["ids_match"] and result["labels_match"] and result["max_abs_logit_error"] <= TOLERANCE]
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "prediction_path": str(prediction_path),
        "config_path": str(config_path),
        "training_history_path": str(artifact / "training_history.csv"),
        "split_metadata_path": None,
        "preprocessing_metadata_path": None,
        "selected_epoch": smoke["best_epoch"],
        "validation_selection_rule": smoke["selection_metric"],
        "training_provenance": config,
        "architecture": {
            "name": "unpretrained small-image ResNet18",
            "input": "native 28x28 RGB",
            "conv1": "3x3 stride 1",
            "maxpool": "Identity",
            "representation_dimension": 512,
            "final_head_shape": [5, 512],
        },
        "official_test_split_identity": {
            "dataset_path": str(DATASET),
            "sample_ids_match": id_match,
            "labels_match": label_match,
            "test_sample_count": int(labels.size),
        },
        "replays": replays,
        "verified_preprocessing": verified[0] if len(verified) == 1 else None,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite provenance report: {args.output}")
    if not DATASET.is_file():
        raise FileNotFoundError(f"missing official RetinaMNIST NPZ: {DATASET}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    with np.load(DATASET) as dataset:
        images = dataset["test_images"]
        labels = dataset["test_labels"].reshape(-1).astype(np.int64)
    if images.shape != (400, 28, 28, 3) or labels.shape != (400,):
        raise ValueError(f"unexpected official test split shapes: images={images.shape}, labels={labels.shape}")
    report = {
        "purpose": "frozen checkpoint/saved-logit provenance replay only; no training or performance evaluation",
        "dataset": {"path": str(DATASET), "split": "official RetinaMNIST test", "image_shape": list(images.shape)},
        "tolerance": TOLERANCE,
        "transform_candidates": list(TRANSFORMS),
        "seeds": [audit_seed(seed, images, labels, torch.device(args.device)) for seed in SEEDS],
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
