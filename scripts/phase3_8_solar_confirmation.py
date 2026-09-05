#!/usr/bin/env python3
"""Phase 3.8 frozen solar CE/RPS rare-upper-extreme confirmation protocol.

This deliberately reuses the audited Phase 3.7A data loader and only adds the
all-split alignment audit, smoke gate, explicit artifact contracts, and richer
provenance required for the resumed confirmation experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18

from phase3_7a_solar_3ch import (
    CHANNELS, CHANNEL_INDICES, EXPECTED, Solar, manifest, report, rps,
    source_channels,
)


DATASET = "/scratch/users/jhong36/data/surya-bench-224.zarr"
INDEX = "/scratch/users/jhong36/data"
SPLIT_NAMES = ("train", "validation", "test")
SEED = 0
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 300
PATIENCE = 3


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load_splits(index: str):
    return [manifest(Path(index) / f"{name}.csv", expected)
            for name, expected in zip(SPLIT_NAMES, EXPECTED)]


def alignment_summary(original, aligned):
    aligned_ids = set(int(x) for x in aligned.id.tolist())
    missing = original.loc[~original.id.isin(aligned_ids)].copy()
    def counts(frame):
        return [int((frame.y == klass).sum()) for klass in range(5)]
    by_year = []
    for year, group in original.groupby(original.timestamp.dt.year):
        retained = int(group.id.isin(aligned_ids).sum())
        by_year.append({"year": int(year), "original": int(len(group)),
                        "aligned": retained, "missing": int(len(group) - retained),
                        "missing_rate": float(1 - retained / len(group))})
    original_counts, aligned_counts, missing_counts = map(counts, (original, aligned, missing))
    missing_rates = [float(m / n) if n else None for n, m in zip(original_counts, missing_counts)]
    global_missing = float(len(missing) / len(original))
    return {
        "original_count": int(len(original)), "aligned_count": int(len(aligned)),
        "missing_count": int(len(missing)), "missing_rate": global_missing,
        "original_class_counts": original_counts, "aligned_class_counts": aligned_counts,
        "missing_class_counts": missing_counts, "missing_rate_by_class": missing_rates,
        "x_support": int(aligned_counts[4]), "by_year": by_year,
    }


def data_audit(splits, root: str, stats):
    datasets = [Solar(split, root, (stats["mean"], stats["std"]), augment=False)
                for split in splits]
    summaries = {name: alignment_summary(original, dataset.d)
                 for name, original, dataset in zip(SPLIT_NAMES, splits, datasets)}
    # These are pre-specified integrity—not performance—gates.  An absent X
    # class cannot answer the endpoint question; a >25-point class-specific
    # missing-rate excess would make the retained subset materially distorted.
    issues = []
    for name, summary in summaries.items():
        if summary["x_support"] == 0:
            issues.append(f"{name} has no aligned X-class support")
        for klass, rate in enumerate(summary["missing_rate_by_class"]):
            if rate is not None and rate - summary["missing_rate"] > 0.25:
                issues.append(f"{name} class {klass} missing rate exceeds global by >0.25")
    return datasets, {"dataset": root, "timestamp_mapping": "Zarr time ns - 8 hours equals manifest timestamp ns", "splits": summaries, "integrity_pass": not issues, "integrity_issues": issues}


def verify_stats(stats_path: Path):
    stats = json.loads(stats_path.read_text())
    required = {"channels", "channel_indices", "shape", "mean", "std", "fit_split", "transform"}
    if not required.issubset(stats):
        raise ValueError(f"incomplete normalization artifact: missing {sorted(required - set(stats))}")
    if tuple(stats["channels"]) != CHANNELS or tuple(stats["channel_indices"]) != CHANNEL_INDICES:
        raise ValueError("normalization artifact violates frozen channel contract")
    if stats["shape"] != [3, 224, 224] or stats["fit_split"] != "train":
        raise ValueError("normalization artifact shape or fit split is invalid")
    mean, std = np.asarray(stats["mean"], dtype=float), np.asarray(stats["std"], dtype=float)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("normalization statistics are invalid")
    return stats


def loaders(datasets, batch, workers, train_augment=True):
    return [DataLoader(dataset, batch_size=batch, shuffle=index == 0,
                       num_workers=workers, pin_memory=True)
            for index, dataset in enumerate(datasets)]


def step(model, loader, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    values = []
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(optimizer is not None):
            value = criterion(model(images), labels)
            if not torch.isfinite(value):
                raise ValueError("non-finite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        values.append(float(value.detach().cpu()))
    return float(np.mean(values))


def make_model(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 5)
    return model.to(device)


def smoke(datasets, method, batch, workers, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 3.8 smoke must run on its requested GPU allocation")
    model = make_model(device)
    criterion = torch.nn.CrossEntropyLoss() if method == "ce" else rps
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = DataLoader(Subset(datasets[0], range(min(batch, len(datasets[0])))),
                        batch_size=min(batch, len(datasets[0])), num_workers=workers, pin_memory=True)
    loss = step(model, loader, criterion, device, optimizer)
    sample, label, identifier = datasets[0][0]
    payload = {"method": method, "loss": loss, "finite_loss": bool(np.isfinite(loss)),
               "device": str(device), "shape": list(sample.shape), "label": int(label),
               "sample_id": int(identifier), "forward_backward": True,
               "gpu_available": bool(torch.cuda.is_available())}
    atomic_json(out / "smoke.json", payload)


def train(datasets, method, batch, workers, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Phase 3.8 training must run on its requested GPU allocation")
    model = make_model(device)
    criterion = torch.nn.CrossEntropyLoss() if method == "ce" else rps
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Frozen project preprocessing: training-only independent spatial flips.
    datasets[0].augment = True
    all_loaders = loaders(datasets, batch, workers)
    history, best, selected_epoch, selected_state, waits = [], math.inf, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = step(model, all_loaders[0], criterion, device, optimizer)
        validation_loss = step(model, all_loaders[1], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if validation_loss < best:
            best, selected_epoch, waits = validation_loss, epoch, 0
            selected_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            waits += 1
        if waits >= PATIENCE:
            break
    model.load_state_dict(selected_state)
    torch.save({"state_dict": selected_state, "selected_epoch": selected_epoch,
                "validation_loss": best, "method": method, "seed": SEED}, out / "selected_checkpoint.pt")
    logits, labels, identifiers = [], [], []
    model.eval()
    with torch.no_grad():
        for images, target, identifier in all_loaders[2]:
            logits.append(model(images.to(device, non_blocking=True)).cpu())
            labels.append(target)
            identifiers.append(identifier)
    result = report(torch.cat(labels).numpy(), torch.cat(logits).numpy(), torch.cat(identifiers).numpy(), out / "evaluation")
    with (out / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader(); writer.writerows(history)
    return {"selected_epoch": selected_epoch, "validation_loss": best, "result": result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "smoke", "train"))
    parser.add_argument("--method", choices=("ce", "rps"))
    parser.add_argument("--root", default=DATASET)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source_channels(args.root)
    stats_path = Path(args.stats)
    stats = verify_stats(stats_path)
    shutil.copy2(stats_path, out / "normalization_used.json")
    splits = load_splits(args.index)
    datasets, audit = data_audit(splits, args.root, stats)
    atomic_json(out / "alignment_audit.json", audit)
    if not audit["integrity_pass"]:
        raise RuntimeError("data-integrity gate failed: " + "; ".join(audit["integrity_issues"]))
    if args.mode == "audit":
        return
    if not args.method:
        parser.error("--method is required for smoke/train")
    if args.mode == "smoke":
        smoke(datasets, args.method, args.batch, args.workers, out)
        return
    payload = train(datasets, args.method, args.batch, args.workers, out)
    payload.update({"method": args.method, "seed": SEED, "dataset": args.root,
                    "channels": list(CHANNELS), "channel_indices": list(CHANNEL_INDICES),
                    "shape": [3, 224, 224], "optimizer": "AdamW",
                    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                    "batch_size": args.batch, "max_epochs": MAX_EPOCHS,
                    "early_stopping_patience": PATIENCE,
                    "checkpoint_selection": "minimum validation CE" if args.method == "ce" else "minimum validation RPS"})
    atomic_json(out / "config.json", payload)


if __name__ == "__main__":
    main()
