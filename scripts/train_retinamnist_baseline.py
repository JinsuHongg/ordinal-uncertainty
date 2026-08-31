#!/usr/bin/env python3
"""Train one CE/ResNet18 RetinaMNIST baseline and write Experiment 0 outputs."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from torch import nn

from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.evaluation.ordinal_uncertainty import evaluate_predictions
from ordinal_uncertainty.models.resnet import make_resnet18
from ordinal_uncertainty.utils.reproducibility import set_seed


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    data_root: str = "data/medmnist"
    output_root: str = "outputs"
    output_directory: str = ""
    image_size: int = 28
    device: str = "auto"
    download: bool = False


def run_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    criterion = nn.CrossEntropyLoss()
    total_loss = total_correct = total = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.reshape(-1).long().to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            total_correct += int((logits.argmax(1) == labels).sum())
            total += len(labels)
    return total_loss / total, total_correct / total


def collect_predictions(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); all_labels = []; all_logits = []
    with torch.no_grad():
        for images, labels in loader:
            all_labels.append(labels.reshape(-1).numpy())
            all_logits.append(model(images.to(device)).cpu().numpy())
    logits = np.concatenate(all_logits)
    return np.concatenate(all_labels), logits, torch.softmax(torch.from_numpy(logits), dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    for field in TrainConfig.__dataclass_fields__.values():
        flag = "--" + field.name.replace("_", "-")
        if isinstance(field.default, bool):
            parser.add_argument(flag, action="store_true", default=field.default)
        else:
            parser.add_argument(flag, type=type(field.default), default=field.default)
    args = TrainConfig(**vars(parser.parse_args()))
    if args.seed not in range(5): raise ValueError("Experiment 0 seeds are restricted to 0, 1, 2, 3, 4")
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    output = Path(args.output_directory) if args.output_directory else Path(args.output_root) / "retinamnist" / "single_model_baseline" / f"seed_{args.seed}"
    if output.exists(): raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    loaders, dataset_metadata = retinamnist_loaders(Path(args.data_root), args.batch_size, args.num_workers, args.download, args.image_size)
    model = make_resnet18(dataset_metadata["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []; best_nll = float("inf"); best_state = None
    for epoch in range(1, args.epochs + 1):
        train_nll, train_accuracy = run_epoch(model, loaders["train"], device, optimizer)
        val_nll, val_accuracy = run_epoch(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_nll": train_nll, "train_accuracy": train_accuracy, "val_nll": val_nll, "val_accuracy": val_accuracy})
        if val_nll < best_nll:
            best_nll, best_state = val_nll, {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    labels, logits, probabilities = collect_predictions(model, loaders["test"], device)
    summary = evaluate_predictions(labels, logits, probabilities, output)
    torch.save({"model_state_dict": best_state, "num_classes": dataset_metadata["num_classes"]}, output / "best_checkpoint.pt")
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    configuration = asdict(args) | {"model": "ResNet18 (unpretrained; 3x3 stride-1 stem, no maxpool)", "loss": "cross_entropy", "optimizer": "AdamW", "checkpoint_selection": "minimum validation NLL", "device_resolved": str(device), "deterministic_algorithms_requested": True, "dataset": dataset_metadata}
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["prediction_metrics"], indent=2))


if __name__ == "__main__": main()
