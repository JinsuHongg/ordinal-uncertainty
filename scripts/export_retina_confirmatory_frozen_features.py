#!/usr/bin/env python3
"""Export RetinaMNIST confirmatory frozen features without fitting a head.

This utility is intentionally Retina-only.  It regenerates the exact 512-D
input to ``model.fc`` for a confirmed A/C backbone and validates the archived
original-head outputs before publishing an immutable feature archive.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ordinal_uncertainty.evaluation.oof import stratified_five_fold_assignments
from ordinal_uncertainty.metrics.decision import bayes_decisions

from run_ac_mechanism_replication import build_model, capture_features, load_state, make_retina_data, softmax


ROOT = Path("outputs/mechanism_replication")
DATASET = Path("data/medmnist/retinamnist.npz")
EXPECTED_COUNTS = np.asarray([486, 128, 206, 194, 66], dtype=np.int64)
REPLAY_TOL = 2e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_epoch(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("epoch", "best_epoch", "selected_epoch"):
        if key in payload:
            return int(payload[key])
    return None


def selected_epoch(path: Path) -> int | None:
    """Recover the selected epoch from checkpoint metadata or its frozen history."""
    epoch = checkpoint_epoch(path)
    if epoch is not None:
        return epoch
    history = path.with_name("training_history.csv")
    if not history.is_file():
        return None
    rows = list(csv.DictReader(history.open(encoding="utf-8")))
    if not rows:
        return None
    metric = "val_nll" if "val_nll" in rows[0] else "val_loss" if "val_loss" in rows[0] else None
    if metric is None:
        return None
    return int(min(rows, key=lambda row: float(row[metric]))["epoch"])


def require_retina_architecture(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    if model.training:
        raise RuntimeError("feature extraction model must be in eval mode")
    if model.conv1.kernel_size != (3, 3) or model.conv1.stride != (1, 1):
        raise RuntimeError("expected native Retina 3x3 stride-1 stem")
    if not isinstance(model.maxpool, torch.nn.Identity):
        raise RuntimeError("expected native Retina maxpool=Identity")
    if model.fc.in_features != 512 or tuple(state["fc.weight"].shape) != (5, 512):
        raise RuntimeError("expected a 5x512 Retina classifier head")


def exact_l1(probabilities: np.ndarray) -> np.ndarray:
    return bayes_decisions(probabilities)["l1_bayes_decision"].astype(np.int64)


def compare_a_outputs(
    features: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    archived: dict[str, np.ndarray],
    head_state: dict[str, object],
) -> dict[str, object]:
    weight = head_state["weight"].detach().cpu().float()
    bias = head_state["bias"].detach().cpu().float()
    with torch.inference_mode():
        logits = torch.nn.functional.linear(torch.from_numpy(features), weight, bias).numpy()
    probabilities = softmax(logits)
    l1 = exact_l1(probabilities)
    ids_match = bool(np.array_equal(sample_ids, archived["sample_ids"]))
    labels_match = bool(np.array_equal(labels, archived["labels"]))
    logit_error = float(np.max(np.abs(logits - archived["a_logits"])))
    probability_error = float(np.max(np.abs(probabilities - archived["a_probabilities"])))
    mean_error = float(np.max(np.abs(probabilities @ np.arange(probabilities.shape[1]) - archived["a_predictive_mean"])))
    l1_match = bool(np.array_equal(l1, archived["a_l1"]))
    status = "PASS" if ids_match and labels_match and logit_error <= REPLAY_TOL and probability_error <= REPLAY_TOL and l1_match else "MISMATCH"
    return {
        "status": status,
        "sample_ids_match": ids_match,
        "labels_match": labels_match,
        "max_abs_logit_error": logit_error,
        "max_abs_probability_error": probability_error,
        "max_abs_predictive_mean_error": mean_error,
        "exact_l1_match": l1_match,
    }


def h1_identity_check(labels: np.ndarray, archived: dict[str, np.ndarray], source_manifest: dict[str, object]) -> dict[str, object]:
    mask = labels == 4
    summary = source_manifest["rare_end_h1_summary"]
    observed = {
        "support": int(mask.sum()),
        "a_mae": float(np.abs(4 - archived["a_l1"][mask]).mean()),
        "a_exact": int((archived["a_l1"][mask] == 4).sum()),
        "a_mean_shrinkage": float(archived["a_inward_shrinkage"][mask].mean()),
    }
    match = all(np.isclose(observed[name], summary[name], atol=1e-12) for name in observed)
    return {"status": "PASS" if match else "MISMATCH", "observed": observed}


def deterministic_subset_check(model: torch.nn.Module, loader: DataLoader, device: torch.device, batch: int, workers: int) -> dict[str, object]:
    subset = Subset(loader.dataset, list(range(min(64, len(loader.dataset)))))
    subset_loader = DataLoader(subset, batch_size=batch, shuffle=False, num_workers=workers)
    first = capture_features(model, subset_loader, device)
    second = capture_features(model, subset_loader, device)
    return {
        "status": "PASS" if np.array_equal(first["sample_ids"], second["sample_ids"]) and np.array_equal(first["labels"], second["labels"]) and np.array_equal(first["features"], second["features"]) else "MISMATCH",
        "rows": int(len(first["labels"])),
        "max_abs_feature_error": float(np.max(np.abs(first["features"] - second["features"]))),
    }


def export(args: argparse.Namespace) -> dict[str, object]:
    source = ROOT / "ac" / "retina" / args.objective / f"seed_{args.seed}"
    source_manifest_path = source / "manifest.json"
    source_arrays_path = source / "per_sample_arrays.npz"
    source_head_path = source / "A_original_head.pt"
    for required in (source_manifest_path, source_arrays_path, source_head_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing required confirmatory artifact: {required}")
    source_manifest = read_json(source_manifest_path)
    if source_manifest["dataset"] != "retina" or source_manifest["backbone_objective"] != args.objective or source_manifest["backbone_seed"] != args.seed:
        raise RuntimeError("source manifest does not identify the requested Retina confirmatory setting")
    checkpoint = Path(str(source_manifest["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing manifest checkpoint: {checkpoint}")
    destination = ROOT / "features" / "retina" / args.objective / f"seed_{args.seed}"
    if destination.exists() and not args.refresh_manifest:
        raise FileExistsError(f"refusing to overwrite feature archive: {destination}")
    if destination.exists() and not (destination / "features.npz").is_file():
        raise RuntimeError(f"existing feature destination is incomplete: {destination}")

    state = load_state(checkpoint)
    model = build_model("retina", state, args.device)
    require_retina_architecture(model, state)
    loader = make_retina_data(args.retina_root, "normalize_half", args.batch_size, args.workers)
    deterministic = deterministic_subset_check(model, loader, args.device, args.batch_size, args.workers)
    if deterministic["status"] != "PASS":
        raise RuntimeError(f"non-deterministic frozen feature extraction: {deterministic}")
    started = time.perf_counter()
    data = capture_features(model, loader, args.device)
    runtime = time.perf_counter() - started
    if data["features"].shape != (1080, 512) or data["features"].dtype != np.float32:
        raise RuntimeError(f"unexpected feature archive shape/dtype: {data['features'].shape}, {data['features'].dtype}")
    if not np.array_equal(data["sample_ids"], np.arange(1080, dtype=np.int64)):
        raise RuntimeError("unexpected natural Retina sample-ID order")
    if not np.array_equal(np.bincount(data["labels"], minlength=5), EXPECTED_COUNTS):
        raise RuntimeError("Retina train class counts mismatch")
    folds = stratified_five_fold_assignments(data["labels"], seed=0)
    with np.load(source_arrays_path, allow_pickle=False) as archive:
        archived = {name: archive[name] for name in archive.files}
    if not np.array_equal(folds, archived["folds"]):
        raise RuntimeError("reconstructed OOF folds do not match archived confirmatory folds")
    head = torch.load(source_head_path, map_location="cpu", weights_only=False)
    if not torch.equal(head["weight"], state["fc.weight"].cpu()) or not torch.equal(head["bias"], state["fc.bias"].cpu()):
        raise RuntimeError("archived A head does not exactly match the manifest checkpoint")
    a_reproduction = compare_a_outputs(data["features"], data["labels"], data["sample_ids"], archived, head)
    if a_reproduction["status"] != "PASS":
        raise RuntimeError(f"A-output reproduction failed: {a_reproduction}")
    h1 = h1_identity_check(data["labels"], archived, source_manifest)
    if h1["status"] != "PASS":
        raise RuntimeError(f"H1 A identity check failed: {h1}")

    destination.mkdir(parents=True, exist_ok=True)
    feature_path = destination / "features.npz"
    if args.refresh_manifest:
        with np.load(feature_path, allow_pickle=False) as existing:
            if not np.array_equal(existing["features"], data["features"]) or not np.array_equal(existing["labels"], data["labels"]) or not np.array_equal(existing["sample_ids"], data["sample_ids"]) or not np.array_equal(existing["folds"], folds):
                raise RuntimeError("existing feature archive differs from the verified regeneration")
    else:
        np.savez_compressed(
            feature_path,
            features=data["features"], labels=data["labels"], sample_ids=data["sample_ids"],
            indices=data["sample_ids"].copy(), folds=folds.astype(np.int64),
            split_roles=np.full(len(folds), "training_oof_member", dtype="U20"),
        )
    manifest = {
        "purpose": "deterministic RetinaMNIST confirmatory frozen feature regeneration only; no training or head fitting",
        "dataset": "RetinaMNIST",
        "objective": args.objective,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "selected_epoch": selected_epoch(checkpoint),
        "architecture": "unpretrained small-image ResNet18; 3x3 stride-1 stem; no maxpool",
        "feature_layer": "512-D input to model.fc after average pooling/flatten",
        "feature_dimension": 512,
        "num_classes": 5,
        "preprocessing": "official local RetinaMNIST train split; native 28x28 RGB; ToTensor; Normalize(mean=(0.5,)*3, std=(0.5,)*3); no augmentation",
        "source_dataset": {"path": str(DATASET), "split": "official train", "rows": 1080},
        "oof_fold_provenance": "stratified_five_fold_assignments(labels, seed=0), exact match to archived A/C folds; one shared frozen backbone per seed",
        "counts": {"total": 1080, "class_counts": EXPECTED_COUNTS.tolist(), "fold_counts": np.bincount(folds, minlength=5).tolist()},
        "archive_path": str(feature_path),
        "archive_sha256": sha256(feature_path),
        "archive_fields": ["features", "labels", "sample_ids", "indices", "folds", "split_roles"],
        "extraction_script": "scripts/export_retina_confirmatory_frozen_features.py",
        "extraction_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(args.device),
        "runtime_seconds": runtime,
        "deterministic_subset_check": deterministic,
        "no_training": True,
        "no_head_fitting": True,
        "a_output_reproduction": a_reproduction,
        "h1_a_identity_check": h1,
        "c_output_verification": "UNAVAILABLE (not performed; C replay is outside the required A identity gate)",
        "source_ac_manifest": str(source_manifest_path),
        "source_per_sample_arrays": str(source_arrays_path),
        "source_a_head": str(source_head_path),
    }
    (destination / "feature_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", required=True, choices=("ce", "rps"))
    parser.add_argument("--seed", required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--retina-root", default="data/medmnist")
    parser.add_argument("--refresh-manifest", action="store_true", help="reverify an existing archive and refresh only its manifest")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    print(json.dumps(export(args), indent=2))


if __name__ == "__main__":
    main()
