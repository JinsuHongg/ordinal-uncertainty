#!/usr/bin/env python3
"""Frozen-manifest UTKFace CE/RPS seed-0 ordinal-failure replication."""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from torch.utils.data import DataLoader

from ordinal_uncertainty.data.utkface import (
    NUM_CLASSES, UTKFaceDataset, audit_corpus, class_counts, load_manifest, records_for_split, utkface_transform,
)
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics
from ordinal_uncertainty.models.ordinal import rps_loss
from ordinal_uncertainty.models.resnet import make_resnet18
from ordinal_uncertainty.utils.reproducibility import set_seed


ARCHIVED_MANIFEST = Path("/home/jhong90/github_proj/ordinal-cqr/data/manifests/conference_v0_3/utkface/manifest.jsonl")
ARCHIVED_METADATA = ARCHIVED_MANIFEST.with_name("manifest_metadata.json")
DATA_ROOT = Path("/mnt/storage/data/utkface/UTKFace")
COVERAGES = np.round(np.arange(1.0, 0.099, -0.05), 2)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def manifest_audit(manifest: Path, metadata: Path, data_root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    corpus = audit_corpus(data_root)
    records = load_manifest(manifest, data_root)
    source_names = {path.name for path in data_root.iterdir() if path.is_file()}
    manifest_names = {str(record["sample_id"]).removeprefix("utkface:") for record in records}
    if source_names != manifest_names:
        raise ValueError(f"manifest/corpus mismatch: manifest={len(manifest_names)}, corpus={len(source_names)}")
    counts = {split: class_counts(records_for_split(records, split)) for split in ("train", "validation", "calibration", "test")}
    summary = {
        "corpus": corpus, "manifest_path": str(manifest), "manifest_metadata_path": str(metadata),
        "manifest_metadata": json.loads(metadata.read_text(encoding="utf-8")), "split_class_counts": counts,
        "replication_splits": {key: counts[key] for key in ("train", "validation", "test")},
        "calibration_split": "preserved from the historical manifest but unused in this CE/RPS replication",
        "preprocessing": "RGB; Resize(128,128); train RandomHorizontalFlip; ImageNet normalization; no extra crop/alignment",
    }
    (output / "dataset_audit.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(manifest, output / "frozen_manifest.jsonl")
    shutil.copy2(metadata, output / "frozen_manifest_metadata.json")
    rows = []
    for split, values in counts.items():
        total = sum(values); largest = max(values)
        for label, count in enumerate(values):
            rows.append({"split": split, "class": label, "count": count, "proportion": count / total, "majority_to_class_ratio": largest / count})
    write_csv(output / "split_class_counts.csv", rows)


def loaders(manifest: Path, data_root: Path, batch_size: int, workers: int) -> tuple[dict[str, DataLoader], dict[str, list[dict[str, object]]]]:
    records = load_manifest(manifest, data_root)
    splits = {name: records_for_split(records, name) for name in ("train", "validation", "test")}
    datasets = {name: UTKFaceDataset(value, data_root, utkface_transform(name == "train")) for name, value in splits.items()}
    return ({
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=torch.cuda.is_available()),
        "validation": DataLoader(datasets["validation"], batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available()),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available()),
    }, splits)


def loss_for(method: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, labels) if method == "ce" else rps_loss(logits, labels)


def run_epoch(model: torch.nn.Module, loader: DataLoader, method: str, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> float:
    model.train(optimizer is not None)
    total = 0.0; count = 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels, _ in loader:
            labels = labels.to(device, non_blocking=True)
            logits = model(images.to(device, non_blocking=True))
            loss = loss_for(method, logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training/validation loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach()) * len(labels); count += len(labels)
    return total / count


def infer(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval(); all_logits = []; all_labels = []; all_ids: list[str] = []
    with torch.no_grad():
        for images, labels, sample_ids in loader:
            all_logits.append(model(images.to(device, non_blocking=True)).cpu())
            all_labels.append(labels.cpu())
            all_ids.extend(str(value) for value in sample_ids)
    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy().astype(int), np.asarray(all_ids, dtype=str)


def decision_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    error = np.abs(labels - prediction)
    return {"accuracy": float((labels == prediction).mean()), "mae": float(error.mean()), "qwk": float(cohen_kappa_score(labels, prediction, weights="quadratic")), "severe_count": int((error >= 2).sum()), "severe_prevalence": float((error >= 2).mean())}


def evaluate(method: str, output: Path, logits: np.ndarray, labels: np.ndarray, sample_ids: np.ndarray) -> dict[str, object]:
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).double().numpy()
    decisions = bayes_decisions(probabilities); classes = np.arange(NUM_CLASSES); predictive_mean = probabilities @ classes
    shrinkage = inward_shrinkage(labels, probabilities)
    ece, _ = expected_calibration_error(labels, probabilities)
    probability = prediction_metrics(labels, probabilities) | {"ece": float(ece)}
    decision_rows = []; risk_rows = []; coverage_rows = []; classwise = []; endpoint_rows = []; routing = []
    for rule, key, risk_key in (("mode", "mode_decision", "mode_l1_risk"), ("l1", "l1_bayes_decision", "l1_bayes_risk"), ("l2", "l2_bayes_decision", "l2_bayes_risk")):
        prediction = decisions[key]; error = np.abs(labels - prediction); severe = error >= 2
        decision_rows.append({"method": method, "decision": rule, **decision_metrics(labels, prediction)})
        risk = decisions[risk_key]; order = np.argsort(risk, kind="stable"); curve = []
        for coverage in COVERAGES:
            retained = order[:max(1, int(np.ceil(coverage * len(labels))))]
            row = {"method": method, "decision": rule, "coverage": float(coverage), "retained_count": int(len(retained)), "ordinal_mae": float(error[retained].mean())}
            coverage_rows.append(row); curve.append(row)
        risk_rows.append({"method": method, "decision": rule, "spearman": float(spearmanr(risk, error).statistic), "severe_auroc": float(roc_auc_score(severe, risk)), "severe_auprc": float(average_precision_score(severe, risk)), "mean_selective_mae": float(np.mean([row["ordinal_mae"] for row in curve])), "mode_to_l1_changed_fraction": float((decisions["mode_decision"] != decisions["l1_bayes_decision"]).mean()) if rule == "l1" else None})
        for true_class in range(NUM_CLASSES):
            mask = labels == true_class
            classwise.append({"method": method, "decision": rule, "true_class": true_class, "count": int(mask.sum()), **decision_metrics(labels[mask], prediction[mask]), "mean_true_probability": float(probabilities[mask, true_class].mean()), "mean_predictive_mean": float(predictive_mean[mask].mean()), "mean_l1_bayes_risk": float(decisions["l1_bayes_risk"][mask].mean())})
        if rule in {"mode", "l1", "l2"}:
            for true_class in (0, 4):
                mask = labels == true_class
                for predicted_class in range(NUM_CLASSES):
                    routing.append({"method": method, "decision": rule, "true_class": true_class, "predicted_class": predicted_class, "count": int((prediction[mask] == predicted_class).sum()), "fraction": float((prediction[mask] == predicted_class).mean())})
    for true_class in (0, 4):
        mask = labels == true_class; l1 = decisions["l1_bayes_decision"]; error = np.abs(l1[mask] - true_class); adjacent = 1 if true_class == 0 else 3
        endpoint_rows.append({"method": method, "true_class": true_class, "count": int(mask.sum()), "accuracy_l1": float((l1[mask] == true_class).mean()), "mae_l1": float(error.mean()), "severe_prevalence_l1": float((error >= 2).mean()), "mean_p_true": float(probabilities[mask, true_class].mean()), "median_p_true": float(np.median(probabilities[mask, true_class])), "mean_p_adjacent": float(probabilities[mask, adjacent].mean()), "median_p_adjacent": float(np.median(probabilities[mask, adjacent])), "mean_top_two_mass": float((probabilities[mask, true_class] + probabilities[mask, adjacent]).mean()), "median_top_two_mass": float(np.median(probabilities[mask, true_class] + probabilities[mask, adjacent])), "mean_predictive_mean": float(predictive_mean[mask].mean()), "inward_shrinkage": float(shrinkage[mask].mean()), "mean_l1_bayes_risk": float(decisions["l1_bayes_risk"][mask].mean()), "median_l1_bayes_risk": float(np.median(decisions["l1_bayes_risk"][mask]))})
    prediction_rows = [{
        "sample_id": str(sample_ids[i]),
        "true_label": int(labels[i]),
        "logits": json.dumps(logits[i].tolist()),
        "probabilities": json.dumps(probabilities[i].tolist()),
        **{key: int(decisions[key][i]) for key in ("mode_decision", "l1_bayes_decision", "l2_bayes_decision")},
        "mode_severe_error": bool(abs(int(labels[i]) - int(decisions["mode_decision"][i])) >= 2),
        "l1_severe_error": bool(abs(int(labels[i]) - int(decisions["l1_bayes_decision"][i])) >= 2),
        "l2_severe_error": bool(abs(int(labels[i]) - int(decisions["l2_bayes_decision"][i])) >= 2),
        "l1_bayes_risk": float(decisions["l1_bayes_risk"][i]),
    } for i in range(len(labels))]
    write_csv(output / "predictions.csv", prediction_rows); write_csv(output / "decision_metrics.csv", decision_rows); write_csv(output / "risk_metrics.csv", risk_rows); write_csv(output / "risk_coverage.csv", coverage_rows); write_csv(output / "classwise_metrics.csv", classwise); write_csv(output / "endpoint_metrics.csv", endpoint_rows); write_csv(output / "endpoint_routing.csv", routing)
    np.savez_compressed(output / "test_arrays.npz", logits=logits, probabilities=probabilities, labels=labels, sample_ids=sample_ids, **decisions)
    result = {"method": method, "probability": probability, "decisions": decision_rows, "risk_quality": risk_rows, "endpoints": endpoint_rows}
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def smoke(method: str, manifest: Path, data_root: Path, output: Path, batch_size: int, workers: int, device: torch.device) -> None:
    output.mkdir(parents=True, exist_ok=False); set_seed(0)
    loader, splits = loaders(manifest, data_root, batch_size, workers)
    model = make_resnet18(NUM_CLASSES).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    images, labels, ids = next(iter(loader["train"])); logits = model(images.to(device)); loss = loss_for(method, logits, labels.to(device)); optimizer.zero_grad(); loss.backward(); optimizer.step()
    if not torch.isfinite(loss) or logits.shape[1] != NUM_CLASSES:
        raise ValueError("smoke forward/backward failed")
    (output / "smoke.json").write_text(json.dumps({"method": method, "loss": float(loss.detach()), "tensor_shape": list(images.shape), "class_count": NUM_CLASSES, "sample_id": str(ids[0]), "finite": True, "train_class_counts": class_counts(splits["train"])}, indent=2) + "\n", encoding="utf-8")


def train(method: str, manifest: Path, data_root: Path, output: Path, batch_size: int, workers: int, epochs: int, device: torch.device) -> None:
    output.mkdir(parents=True, exist_ok=False); set_seed(0)
    loader, splits = loaders(manifest, data_root, batch_size, workers)
    model = make_resnet18(NUM_CLASSES).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    best, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, loader["train"], method, device, optimizer)
        validation_loss = run_epoch(model, loader["validation"], method, device)
        history.append({"epoch": epoch, "training_loss": train_loss, "validation_selection_loss": validation_loss})
        print(json.dumps({"method": method, "epoch": epoch, "training_loss": train_loss, "validation_selection_loss": validation_loss}), flush=True)
        if validation_loss < best:
            best, best_epoch = validation_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state); logits, labels, ids = infer(model, loader["test"], device)
    result = evaluate(method, output, logits, labels, ids)
    torch.save({"model_state_dict": best_state, "method": method, "seed": 0, "best_epoch": best_epoch, "best_validation_score": best, "selection_metric": "minimum validation CE" if method == "ce" else "minimum validation RPS"}, output / "best_checkpoint.pt")
    write_csv(output / "training_history.csv", history)
    write_csv(output / "validation_history.csv", [{"epoch": row["epoch"], "validation_selection_loss": row["validation_selection_loss"]} for row in history])
    config = {"dataset": "UTKFace filename corpus", "seed": 0, "method": method, "image_size": 128, "preprocessing": "RGB, ImageNet normalization, RandomHorizontalFlip train only", "backbone": "unpretrained small-stem ResNet18 (3x3 stride 1, no maxpool)", "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": 0.01, "batch_size": batch_size, "max_epochs": epochs, "checkpoint_selection": "minimum validation CE" if method == "ce" else "minimum validation RPS", "best_epoch": best_epoch, "best_validation_score": best, "split_counts": {name: class_counts(value) for name, value in splits.items()}, "device": str(device), "result": result}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def summarize(root: Path) -> None:
    """Create deterministic CE/RPS seed-0 comparison artifacts from saved runs."""
    methods = ("ce", "rps")
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=False)
    results: dict[str, dict[str, object]] = {}
    global_rows: list[dict[str, object]] = []
    classwise_rows: list[dict[str, object]] = []
    routing_rows: list[dict[str, object]] = []
    for method in methods:
        run = root / method / "seed_0"
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        results[method] = {"config": config, "metrics": metrics}
        probability = dict(metrics["probability"])
        for decision in metrics["decisions"]:
            row = {"method": method, "decision": decision["decision"], **probability, **decision}
            risk = next(item for item in metrics["risk_quality"] if item["decision"] == decision["decision"])
            row.update({key: value for key, value in risk.items() if key not in {"method", "decision"}})
            global_rows.append(row)
        with (run / "classwise_metrics.csv").open(encoding="utf-8", newline="") as handle:
            classwise_rows.extend(dict(row) for row in csv.DictReader(handle))
        with (run / "endpoint_routing.csv").open(encoding="utf-8", newline="") as handle:
            routing_rows.extend(dict(row) for row in csv.DictReader(handle))
    replication_rows = [
        {"finding": "RPS improves L1 risk/error association", "retinamnist": "REPLICATED", "utkface": "NOT REPLICATED", "evidence": "CE 0.4312 vs RPS 0.4265 Spearman"},
        {"finding": "RPS improves severe-error detection", "retinamnist": "REPLICATED", "utkface": "PARTIALLY REPLICATED", "evidence": "AUROC CE 0.8467 vs RPS 0.8804; AUPRC CE 0.1473 vs RPS 0.1242"},
        {"finding": "RPS improves ordinal selective prediction", "retinamnist": "REPLICATED", "utkface": "NOT REPLICATED", "evidence": "mean selective MAE CE 0.1092 vs RPS 0.1301 (lower is better)"},
        {"finding": "Upper extreme has elevated decision risk", "retinamnist": "REPLICATED", "utkface": "REPLICATED", "evidence": "class-4 mean L1 risk exceeds class-0 risk for CE and RPS"},
        {"finding": "Upper extreme is poorly localized", "retinamnist": "REPLICATED", "utkface": "PARTIALLY REPLICATED", "evidence": "inward predictive means remain below 4, but RPS L1 class-4 accuracy is 52.2%"},
        {"finding": "L1/L2 correction is insufficient", "retinamnist": "REPLICATED", "utkface": "PARTIALLY REPLICATED", "evidence": "decision changes are small and residual class-4 inward routing remains"},
        {"finding": "Lower endpoint is easier than upper endpoint", "retinamnist": "REPLICATED", "utkface": "REPLICATED", "evidence": "class-0 L1 MAE is lower than class-4 for CE and RPS"},
    ]
    write_csv(output / "global_comparison.csv", global_rows)
    write_csv(output / "classwise_comparison.csv", classwise_rows)
    write_csv(output / "upper_extreme_routing.csv", [row for row in routing_rows if row["true_class"] == "4"])
    write_csv(output / "replication_table.csv", replication_rows)
    summary = {
        "decision": "PARTIAL REPLICATION",
        "scope": "UTKFace seed-0 CE/RPS only; frozen bins and manifest",
        "rationale": "Upper-extreme inward displacement and elevated risk reproduce, but the RetinaMNIST RPS L1 risk-quality advantage does not reproduce uniformly: RPS has higher severe AUROC but lower Spearman, AUPRC, and selective MAE quality.",
        "methods": results,
    }
    (output / "phase3_7a_utkface_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("audit", "smoke", "train", "summary")); parser.add_argument("--method", choices=("ce", "rps"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT); parser.add_argument("--manifest", type=Path, default=ARCHIVED_MANIFEST); parser.add_argument("--metadata", type=Path, default=ARCHIVED_METADATA)
    # Historical OCQR used 128, but this fixed 32 is required by the available
    # 5.6 GiB GPU. It is shared by CE/RPS and is not a tuned parameter.
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--workers", type=int, default=0); parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--device", default="auto")
    args = parser.parse_args(); device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if args.mode == "audit": manifest_audit(args.manifest, args.metadata, args.data_root, args.output); return
    if args.mode == "summary": summarize(args.output); return
    if args.method is None: parser.error("--method is required for smoke/train")
    if args.mode == "smoke": smoke(args.method, args.manifest, args.data_root, args.output, args.batch_size, args.workers, device)
    else: train(args.method, args.manifest, args.data_root, args.output, args.batch_size, args.workers, args.epochs, device)


if __name__ == "__main__": main()
