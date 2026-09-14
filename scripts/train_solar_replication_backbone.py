#!/usr/bin/env python3
"""Frozen Solar CE/RPS backbone replication runner (confirmatory seeds 1--4).

This intentionally trains/selects using only aligned train/validation data.
The archived readout is touched solely by the metadata alignment audit; no
readout tensor is passed through the model and no scientific test metrics are
generated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from phase3_7a_solar_3ch import (
    CHANNELS,
    CHANNEL_INDICES,
    EXPECTED,
    Solar,
    manifest,
    rps,
    source_channels,
)
from phase3_8_solar_confirmation import alignment_summary


DATASET = "/scratch/users/jhong36/data/surya-bench-224.zarr"
INDEX = "/scratch/users/jhong36/data"
NORMALIZATION = Path(
    "outputs/solar/phase3_7a_3ch/normalization/train_only_retry_1/normalization.json"
)
MAX_EPOCHS, PATIENCE = 300, 3
LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE = 5e-5, 0.01, 16


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_normalization(path: Path) -> dict:
    stats = json.loads(path.read_text())
    required = {
        "channels",
        "channel_indices",
        "shape",
        "mean",
        "std",
        "fit_split",
        "transform",
    }
    if required - set(stats):
        raise ValueError(
            f"incomplete normalization artifact: {sorted(required - set(stats))}"
        )
    if (
        tuple(stats["channels"]) != CHANNELS
        or tuple(stats["channel_indices"]) != CHANNEL_INDICES
    ):
        raise ValueError("normalization artifact violates frozen channel contract")
    if stats["shape"] != [3, 224, 224] or stats["fit_split"] != "train":
        raise ValueError("normalization artifact shape or fit split is invalid")
    if stats["transform"] != "sign(x)*log1p(abs(x))":
        raise ValueError(
            "normalization artifact transform is not the frozen signed-log transform"
        )
    mean, std = (
        np.asarray(stats["mean"], dtype=float),
        np.asarray(stats["std"], dtype=float),
    )
    if (
        mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or not (std > 0).all()
    ):
        raise ValueError("normalization statistics are invalid")
    return stats


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def audit_alignment(splits, datasets, root: str) -> dict:
    summaries = {
        name: alignment_summary(original, dataset.d)
        for name, original, dataset in zip(
            ("train", "validation", "test"), splits, datasets
        )
    }
    issues = []
    for name, summary in summaries.items():
        if summary["x_support"] == 0:
            issues.append(f"{name} has no aligned X-class support")
        for klass, rate in enumerate(summary["missing_rate_by_class"]):
            if rate is not None and rate - summary["missing_rate"] > 0.25:
                issues.append(
                    f"{name} class {klass} missing rate exceeds global by >0.25"
                )
    return {
        "dataset": root,
        "timestamp_mapping": "Zarr time ns - 8 hours equals manifest timestamp ns",
        "splits": summaries,
        "integrity_pass": not issues,
        "integrity_issues": issues,
        "archived_readout_role": "alignment/provenance only; never loaded into a model in backbone training",
    }


def step(model, loader, criterion, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    values = []
    for images, labels, _ in loader:
        images, labels = (
            images.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
        )
        with torch.set_grad_enabled(optimizer is not None):
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise ValueError("non-finite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        values.append(float(loss.detach().cpu()))
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("ce", "rps"))
    parser.add_argument("--seed", required=True, type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--root", default=DATASET)
    parser.add_argument("--index", default=INDEX)
    parser.add_argument("--stats", type=Path, default=NORMALIZATION)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing condition: {args.out}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Solar backbone replication requires CUDA; refusing CPU fallback"
        )

    seed_everything(args.seed)
    source_channels(args.root)
    stats = verify_normalization(args.stats)
    splits = [
        manifest(Path(args.index) / f"{name}.csv", expected)
        for name, expected in zip(("train", "validation", "test"), EXPECTED)
    ]
    datasets = [
        Solar(split, args.root, (stats["mean"], stats["std"]), augment=False)
        for split in splits
    ]
    audit = audit_alignment(splits, datasets, args.root)
    if not audit["integrity_pass"]:
        raise RuntimeError(
            "data-integrity gate failed: " + "; ".join(audit["integrity_issues"])
        )

    args.out.mkdir(parents=True, exist_ok=False)
    atomic_json(args.out / "alignment_audit.json", audit)
    atomic_json(
        args.out / "config.json",
        {
            "protocol": "frozen Solar Phase-3.8-compatible backbone replication",
            "dataset": args.root,
            "method": args.method,
            "seed": args.seed,
            "channels": list(CHANNELS),
            "channel_indices": list(CHANNEL_INDICES),
            "shape": [3, 224, 224],
            "normalization_artifact": str(args.stats),
            "normalization": stats,
            "training_augmentation": "independent horizontal and vertical flips",
            "architecture": "torchvision ResNet18 weights=None; fc Linear(512,5)",
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "checkpoint_selection": "minimum validation CE"
            if args.method == "ce"
            else "minimum validation RPS",
            "archived_readout_model_role": "prohibited; no model forward pass or scientific test metrics in this stage",
        },
    )

    datasets[0].augment = True
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        datasets[0],
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    validation_loader = DataLoader(
        datasets[1],
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    device = torch.device("cuda:0")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 5)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss() if args.method == "ce" else rps
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    history, best, selected_epoch, selected_state, waits = [], math.inf, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = step(model, train_loader, criterion, device, optimizer)
        validation_loss = step(model, validation_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best:
            best, selected_epoch, waits = validation_loss, epoch, 0
            selected_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            waits += 1
        if waits >= PATIENCE:
            break
    if selected_state is None:
        raise RuntimeError("no validation-selected checkpoint was produced")
    torch.save(
        {
            "state_dict": selected_state,
            "selected_epoch": selected_epoch,
            "validation_loss": best,
            "method": args.method,
            "seed": args.seed,
        },
        args.out / "selected_checkpoint.pt",
    )
    with (args.out / "training_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("epoch", "train_loss", "validation_loss")
        )
        writer.writeheader()
        writer.writerows(history)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "method": args.method,
                "seed": args.seed,
                "selected_epoch": selected_epoch,
                "validation_loss": best,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
