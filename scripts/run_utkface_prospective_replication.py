#!/usr/bin/env python3
"""Run the frozen prospective UTKFace backbone and A/C/N replication protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ordinal_uncertainty.data.utkface import (
    NUM_CLASSES,
    UTKFaceDataset,
    class_counts,
    load_manifest,
    parse_age,
    records_for_split,
    utkface_transform,
)
from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices, natural_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics
from ordinal_uncertainty.models.ordinal import rps_loss
from ordinal_uncertainty.models.resnet import make_resnet18


BACKBONE_BATCH, BACKBONE_EPOCHS = 32, 10
HEAD_BATCH, HEAD_EPOCHS, HEAD_LR = 64, 100, 1e-3
REPLAY_TOL, CONSTRAINT_TOL = 1e-4, 1e-6
K = NUM_CLASSES


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def assert_disjoint_ids(train_ids: list[str] | np.ndarray, validation_ids: list[str] | np.ndarray) -> None:
    if set(map(str, train_ids)) & set(map(str, validation_ids)):
        raise RuntimeError("training and validation sample IDs overlap")


def assert_feature_split_integrity(split_ids: dict[str, np.ndarray]) -> None:
    names = tuple(split_ids)
    for name, ids in split_ids.items():
        values = list(map(str, ids))
        if len(values) != len(set(values)):
            raise RuntimeError(f"duplicate sample IDs in {name} feature archive")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if set(map(str, split_ids[left])) & set(map(str, split_ids[right])):
                raise RuntimeError(f"sample ID overlap between {left} and {right} feature archives")


def replay_original_head(features: np.ndarray, weight: np.ndarray, bias: np.ndarray, direct_logits: np.ndarray) -> dict[str, float | int]:
    replayed_logits = features @ weight.T + bias
    direct_probability = probabilities(direct_logits)
    replayed_probability = probabilities(replayed_logits)
    direct_decisions = bayes_decisions(direct_probability)
    replayed_decisions = bayes_decisions(replayed_probability)
    return {
        "max_logit_diff": float(np.abs(replayed_logits - direct_logits).max()),
        "max_probability_diff": float(np.abs(replayed_probability - direct_probability).max()),
        "mode_differences": int(np.count_nonzero(replayed_decisions["mode_decision"] != direct_decisions["mode_decision"])),
        "l1_differences": int(np.count_nonzero(replayed_decisions["l1_bayes_decision"] != direct_decisions["l1_bayes_decision"])),
    }


def records(manifest: Path, data_root: Path) -> dict[str, list[dict[str, object]]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    frozen = load_manifest(manifest, data_root)
    splits = {name: records_for_split(frozen, name) for name in ("train", "validation")}
    if class_counts(splits["train"]) != [2756, 7128, 2726, 1210, 404]:
        raise RuntimeError("frozen train counts mismatch")
    if class_counts(splits["validation"]) != [459, 1188, 455, 202, 67]:
        raise RuntimeError("frozen validation counts mismatch")
    assert_disjoint_ids(
        [str(row["sample_id"]) for row in splits["train"]],
        [str(row["sample_id"]) for row in splits["validation"]],
    )
    return splits


def export_records(manifest: Path, data_root: Path) -> dict[str, list[dict[str, object]]]:
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    frozen = load_manifest(manifest, data_root)
    splits = {name: records_for_split(frozen, name) for name in ("train", "validation", "test")}
    expected = {
        "train": [2756, 7128, 2726, 1210, 404],
        "validation": [459, 1188, 455, 202, 67],
        "test": [459, 1189, 454, 202, 67],
    }
    for name, counts in expected.items():
        if class_counts(splits[name]) != counts:
            raise RuntimeError(f"frozen {name} counts mismatch")
    assert_feature_split_integrity({name: np.asarray([str(row["sample_id"]) for row in rows]) for name, rows in splits.items()})
    return splits


def loader(items: list[dict[str, object]], data_root: Path, *, train: bool, batch: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        UTKFaceDataset(items, data_root, utkface_transform(train)),
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def loss(objective: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits, labels) if objective == "ce" else rps_loss(logits, labels)


def run_epoch(model: torch.nn.Module, batches: DataLoader, objective: str, device: torch.device, optimizer: torch.optim.Optimizer | None) -> float:
    model.train(optimizer is not None)
    total, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for images, labels, _ in batches:
            logits = model(images.to(device, non_blocking=True))
            value = loss(objective, logits, labels.to(device, non_blocking=True))
            if not torch.isfinite(value):
                raise FloatingPointError("non-finite backbone loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                value.backward()
                optimizer.step()
            total += float(value.detach().cpu()) * len(labels)
            count += len(labels)
    return total / count


def capture(model: torch.nn.Module, batches: DataLoader, device: torch.device) -> dict[str, np.ndarray]:
    features: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    ids: list[str] = []

    def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        features.append(inputs[0].detach().cpu())

    handle = model.fc.register_forward_pre_hook(hook)
    model.eval()
    with torch.inference_mode():
        for images, target, sample_ids in batches:
            logits.append(model(images.to(device, non_blocking=True)).cpu())
            labels.append(target.cpu())
            ids.extend(map(str, sample_ids))
    handle.remove()
    result = {
        "features": torch.cat(features).numpy().astype(np.float32),
        "logits": torch.cat(logits).numpy().astype(np.float32),
        "labels": torch.cat(labels).numpy().astype(np.int64),
        "sample_ids": np.asarray(ids, dtype=str),
    }
    if result["features"].shape[1:] != (512,) or len(np.unique(result["sample_ids"])) != len(ids):
        raise RuntimeError("invalid feature export")
    return result


def run_backbone(args: argparse.Namespace) -> None:
    out = args.out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    splits = records(args.manifest, args.data_root)
    train_loader = loader(splits["train"], args.data_root, train=True, batch=BACKBONE_BATCH, shuffle=True)
    validation_loader = loader(splits["validation"], args.data_root, train=False, batch=BACKBONE_BATCH)
    model = make_resnet18(K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    best, best_epoch, best_state, history = float("inf"), 0, None, []
    for epoch in range(1, BACKBONE_EPOCHS + 1):
        train_value = run_epoch(model, train_loader, args.objective, device, optimizer)
        validation_value = run_epoch(model, validation_loader, args.objective, device, None)
        history.append({"epoch": epoch, "training_loss": train_value, "validation_selection_loss": validation_value})
        if validation_value < best:
            best, best_epoch = validation_value, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no selected checkpoint")
    model.load_state_dict(best_state)
    validation = capture(model, validation_loader, device)
    probability = torch.softmax(torch.from_numpy(validation["logits"]), dim=1).numpy()
    if not np.isfinite(probability).all() or float(np.abs(probability.sum(1) - 1).max()) > 1e-6:
        raise RuntimeError("invalid validation probabilities")
    torch.save({"model_state_dict": best_state, "objective": args.objective, "seed": args.seed, "best_epoch": best_epoch, "best_validation_score": best}, out / "best_checkpoint.pt")
    np.savez_compressed(out / "validation_backbone_arrays.npz", **validation, probabilities=probability)
    write_csv(out / "training_history.csv", history)
    write_json(out / "config.json", {
        "protocol": "UTKFace prospective replication frozen protocol",
        "objective": args.objective,
        "seed": args.seed,
        "manifest_sha256": sha256(args.manifest),
        "evaluation_split": "validation",
        "architecture": "unpretrained small-stem ResNet18, 3x3 stride-1 stem, no max-pool, 5-way head",
        "optimizer": "AdamW", "learning_rate": 1e-4, "weight_decay": .01,
        "batch_size": BACKBONE_BATCH, "epochs": BACKBONE_EPOCHS,
        "selection": f"minimum validation {'CE' if args.objective == 'ce' else 'RPS'}",
        "best_epoch": best_epoch, "best_validation_score": best,
        "device": str(device), "validation_rows": len(validation["labels"]),
    })


def fit_direction_head(features: np.ndarray, labels: np.ndarray, weight: torch.Tensor, bias: torch.Tensor, *, sampling: str, seed: int, device: torch.device) -> tuple[DirectionOnlyLinear, list[dict[str, object]], np.ndarray, dict[str, float]]:
    x, y = torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long)
    head = DirectionOnlyLinear(weight.to(device), bias.to(device)).to(device)
    optimizer = torch.optim.AdamW([head.direction], lr=HEAD_LR, weight_decay=0.0)
    generator = torch.Generator().manual_seed(10_000 + seed)
    source_bias, source_norms = head.fixed_bias.detach().cpu().clone(), head.fixed_norms.detach().cpu().clone()
    observed = np.zeros(K, dtype=np.int64)
    history: list[dict[str, object]] = []
    for epoch in range(1, HEAD_EPOCHS + 1):
        indices = natural_batch_indices(len(y), HEAD_BATCH, generator) if sampling == "natural" else [balanced_batch_indices(y, HEAD_BATCH, generator) for _ in range(int(np.ceil(len(y) / HEAD_BATCH)))]
        losses = []
        for index in indices:
            observed += np.bincount(y[index].numpy(), minlength=K)
            value = torch.nn.functional.cross_entropy(head(x[index].to(device)), y[index].to(device))
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            optimizer.step()
            losses.append(float(value.detach().cpu()))
        history.append({"epoch": epoch, f"{sampling}_ce": float(np.mean(losses)), "steps": len(indices)})
    if sampling == "natural" and not np.array_equal(observed, np.bincount(labels, minlength=K) * HEAD_EPOCHS):
        raise RuntimeError("natural sampler failed empirical draw-count contract")
    constraints = {
        "max_norm_error": float(head.max_norm_error().detach().cpu()),
        "max_bias_error": float((head.fixed_bias.detach().cpu() - source_bias).abs().max()),
        "stored_norm_error": float((head.fixed_norms.detach().cpu() - source_norms).abs().max()),
    }
    if max(constraints.values()) > CONSTRAINT_TOL:
        raise RuntimeError(f"fixed parameter constraint failed: {constraints}")
    return head, history, observed, constraints


def probabilities(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits), dim=1).double().numpy()


def evaluate(labels: np.ndarray, logits: np.ndarray, sample_ids: np.ndarray, output: Path) -> dict[str, object]:
    output.mkdir(parents=True)
    prob = probabilities(logits)
    decisions = bayes_decisions(prob)
    prediction = decisions["l1_bayes_decision"]
    error = np.abs(labels - prediction)
    severe = error >= 2
    mean = prob @ np.arange(K)
    shrinkage = inward_shrinkage(labels, prob)
    ece, _ = expected_calibration_error(labels, prob)
    order = np.argsort(decisions["l1_bayes_risk"], kind="stable")
    coverage = []
    for value in np.round(np.arange(1.0, .099, -.05), 2):
        retained = order[:max(1, int(np.ceil(value * len(labels))))]
        coverage.append({"coverage": float(value), "retained_count": int(len(retained)), "ordinal_mae": float(error[retained].mean())})
    endpoint = {}
    for klass in (0, 4):
        mask = labels == klass
        endpoint[str(klass)] = {
            "support": int(mask.sum()), "routing": [int((prediction[mask] == j).sum()) for j in range(K)],
            "mae_l1": float(error[mask].mean()), "exact_l1": int((prediction[mask] == klass).sum()),
            "severe_prevalence_l1": float(severe[mask].mean()), "mean_predictive_mean": float(mean[mask].mean()),
            "inward_shrinkage": float(shrinkage[mask].mean()), "mean_l1_risk": float(decisions["l1_bayes_risk"][mask].mean()),
            "mean_p_true": float(prob[mask, klass].mean()), "mean_p_adjacent": float(prob[mask, 1 if klass == 0 else 3].mean()),
        }
    values = {
        "global": {**prediction_metrics(labels, prob), "ece": float(ece), "accuracy_l1": float((labels == prediction).mean()), "mae_l1": float(error.mean()), "qwk_l1": float(cohen_kappa_score(labels, prediction, weights="quadratic")), "severe_prevalence_l1": float(severe.mean())},
        "risk_quality": {"spearman": float(spearmanr(decisions["l1_bayes_risk"], error).statistic), "severe_auroc": float(roc_auc_score(severe, decisions["l1_bayes_risk"])), "severe_auprc": float(average_precision_score(severe, decisions["l1_bayes_risk"])), "mean_selective_mae": float(np.mean([row["ordinal_mae"] for row in coverage]))},
        "endpoints": endpoint,
    }
    np.savez_compressed(output / "validation_arrays.npz", sample_ids=sample_ids, labels=labels, logits=logits, probabilities=prob, predictive_mean=mean, inward_shrinkage=shrinkage, **decisions)
    write_csv(output / "risk_coverage.csv", coverage)
    write_json(output / "metrics.json", values)
    return values


def geometry(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray) -> dict[str, np.ndarray]:
    centroids = np.stack([train_x[train_y == k].mean(0) for k in range(K)])
    distances = np.sqrt(((val_x[:, None, :] - centroids[None, :, :]) ** 2).sum(2))
    return {"nearest_centroid": distances.argmin(1).astype(np.int64), "c4_vs_c3_margin": (distances[:, 3] - distances[:, 4]).astype(np.float64)}


def export_archive_arrays(captured: dict[str, np.ndarray], items: list[dict[str, object]], split: str) -> dict[str, np.ndarray]:
    sample_ids = captured["sample_ids"]
    expected_ids = np.asarray([str(row["sample_id"]) for row in items], dtype=str)
    if not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError(f"{split} capture IDs do not match frozen manifest order")
    labels = np.asarray([int(row["Y_ord"]) for row in items], dtype=np.int64)
    if not np.array_equal(captured["labels"], labels):
        raise RuntimeError(f"{split} capture labels do not match frozen manifest")
    filenames = np.asarray([sample_id.removeprefix("utkface:") for sample_id in sample_ids], dtype=str)
    ages = np.asarray([parse_age(filename) for filename in filenames], dtype=np.float64)
    probability = probabilities(captured["logits"])
    decisions = bayes_decisions(probability)
    return {
        "sample_ids": sample_ids, "filenames": filenames, "ages": ages,
        "split": np.full(len(sample_ids), split), "labels": labels,
        "features": captured["features"], "logits": captured["logits"],
        "probabilities": probability, **decisions,
    }


def a_baseline_metrics(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    labels, probability = arrays["labels"], arrays["probabilities"]
    prediction = arrays["l1_bayes_decision"]
    error = np.abs(labels - prediction)
    endpoint = labels == 4
    mean = probability @ np.arange(K)
    shrinkage = inward_shrinkage(labels, probability)
    return {
        "global_l1_mae": float(error.mean()),
        "macro_l1_mae": float(np.mean([error[labels == klass].mean() for klass in range(K)])),
        "severe_error_rate_l1": float((error >= 2).mean()),
        "endpoint_4": {
            "support": int(endpoint.sum()),
            "l1_mae": float(error[endpoint].mean()),
            "exact_match_rate": float((prediction[endpoint] == 4).mean()),
            "mean_predictive_mean": float(mean[endpoint].mean()),
            "mean_inward_shrinkage": float(shrinkage[endpoint].mean()),
            "mean_p4": float(probability[endpoint, 4].mean()),
            "routing": [int((prediction[endpoint] == klass).sum()) for klass in range(K)],
        },
    }


def run_export(args: argparse.Namespace) -> None:
    out = args.out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("objective") != args.objective or checkpoint.get("seed") != args.seed:
        raise RuntimeError("checkpoint objective or seed does not match requested export unit")
    state = checkpoint["model_state_dict"]
    weight, bias = state["fc.weight"].float(), state["fc.bias"].float()
    if tuple(weight.shape) != (K, 512) or tuple(bias.shape) != (K,):
        raise RuntimeError("checkpoint head shape does not match frozen contract")
    device = torch.device(args.device)
    splits = export_records(args.manifest, args.data_root)
    model = make_resnet18(K)
    model.load_state_dict(state)
    model.to(device).eval()
    archives: dict[str, dict[str, np.ndarray]] = {}
    replay: dict[str, dict[str, float | int]] = {}
    for split, items in splits.items():
        captured = capture(model, loader(items, args.data_root, train=False, batch=BACKBONE_BATCH), device)
        archive = export_archive_arrays(captured, items, split)
        if not np.isfinite(archive["features"]).all() or not np.isfinite(archive["logits"]).all() or not np.isfinite(archive["probabilities"]).all():
            raise FloatingPointError(f"non-finite {split} feature archive")
        archives[split] = archive
        replay[split] = replay_original_head(archive["features"], weight.numpy(), bias.numpy(), archive["logits"])
        if replay[split]["max_logit_diff"] > REPLAY_TOL or replay[split]["max_probability_diff"] > REPLAY_TOL:
            raise RuntimeError(f"{split} A feature replay failed: {replay[split]}")
        if replay[split]["mode_differences"] or replay[split]["l1_differences"]:
            raise RuntimeError(f"{split} A decision replay failed: {replay[split]}")
    accepted = np.load(args.checkpoint.parent / "validation_backbone_arrays.npz")
    exported_validation = archives["validation"]
    if not np.array_equal(accepted["sample_ids"], exported_validation["sample_ids"]) or not np.array_equal(accepted["labels"], exported_validation["labels"]):
        raise RuntimeError("exported validation IDs or labels differ from accepted backbone arrays")
    accepted_validation = {
        "max_logit_diff": float(np.abs(accepted["logits"] - exported_validation["logits"]).max()),
        "max_probability_diff": float(np.abs(accepted["probabilities"] - exported_validation["probabilities"]).max()),
    }
    if max(accepted_validation.values()) > REPLAY_TOL:
        raise RuntimeError(f"exported validation outputs differ from accepted backbone arrays: {accepted_validation}")
    out.mkdir(parents=True)
    for split, archive in archives.items():
        np.savez_compressed(out / f"{split}_features.npz", **archive)
    torch.save({"condition": "A_original", "weight": weight, "bias": bias}, out / "A_original_head.pt")
    write_json(out / "feature_export_metadata.json", {
        "protocol": "UTKFace prospective A/C/N frozen protocol",
        "objective": args.objective, "seed": args.seed, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint), "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest), "feature_definition": "penultimate feature immediately before model.fc",
        "feature_dimension": int(weight.shape[1]), "model_mode": "eval",
        "transform": "Resize(128,128), ToTensor, ImageNet normalization; no augmentation",
        "replay_tolerance_max_abs": REPLAY_TOL, "replay": replay,
        "accepted_validation_comparison": accepted_validation,
        "split_counts": {name: int(len(archive["labels"])) for name, archive in archives.items()},
        "class_counts": {name: np.bincount(archive["labels"], minlength=K).tolist() for name, archive in archives.items()},
        "test_class_4_support": int((archives["test"]["labels"] == 4).sum()),
        "test_a_metrics": a_baseline_metrics(archives["test"]),
    })


def run_heads(args: argparse.Namespace) -> None:
    out = args.out
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    weight, bias = state["fc.weight"].float(), state["fc.bias"].float()
    if tuple(weight.shape) != (K, 512) or tuple(bias.shape) != (K,):
        raise RuntimeError("checkpoint head shape does not match frozen contract")
    device = torch.device(args.device)
    splits = records(args.manifest, args.data_root)
    model = make_resnet18(K); model.load_state_dict(state); model.to(device).eval()
    train = capture(model, loader(splits["train"], args.data_root, train=False, batch=HEAD_BATCH), device)
    validation = capture(model, loader(splits["validation"], args.data_root, train=False, batch=HEAD_BATCH), device)
    assert_disjoint_ids(train["sample_ids"], validation["sample_ids"])
    replay = float(np.abs(validation["features"] @ weight.numpy().T + bias.numpy() - validation["logits"]).max())
    if replay > REPLAY_TOL:
        raise RuntimeError(f"A feature replay failed: {replay}")
    out.mkdir(parents=True)
    np.savez_compressed(out / "train_features.npz", **train)
    np.savez_compressed(out / "validation_features.npz", **validation)
    torch.save({"condition": "A_original", "weight": weight, "bias": bias}, out / "A_original_head.pt")
    all_logits = {"A": validation["logits"]}
    constraints: dict[str, object] = {}
    for condition, sampling in (("C", "balanced"), ("N", "natural")):
        head, history, draws, check = fit_direction_head(train["features"], train["labels"], weight, bias, sampling=sampling, seed=args.seed, device=device)
        with torch.inference_mode():
            all_logits[condition] = head(torch.as_tensor(validation["features"], device=device)).cpu().numpy()
        torch.save({"condition": condition, "sampling": sampling, "state_dict": head.cpu().state_dict(), "fixed_norms": head.fixed_norms.cpu(), "fixed_bias": head.fixed_bias.cpu()}, out / f"{condition}_head.pt")
        write_csv(out / f"{condition}_history.csv", history)
        constraints[condition] = {**check, "observed_draw_counts": draws.tolist()}
    metrics = {name: evaluate(validation["labels"], logits, validation["sample_ids"], out / name) for name, logits in all_logits.items()}
    geom = geometry(train["features"], train["labels"], validation["features"])
    np.savez_compressed(out / "validation_geometry.npz", sample_ids=validation["sample_ids"], labels=validation["labels"], **geom)
    deltas = {"N_minus_A_c4_mae": metrics["N"]["endpoints"]["4"]["mae_l1"] - metrics["A"]["endpoints"]["4"]["mae_l1"], "C_minus_A_c4_mae": metrics["C"]["endpoints"]["4"]["mae_l1"] - metrics["A"]["endpoints"]["4"]["mae_l1"], "C_minus_N_c4_mae": metrics["C"]["endpoints"]["4"]["mae_l1"] - metrics["N"]["endpoints"]["4"]["mae_l1"]}
    write_json(out / "manifest.json", {"protocol": "UTKFace prospective A/C/N frozen protocol", "objective": args.objective, "seed": args.seed, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint), "manifest_sha256": sha256(args.manifest), "evaluation_split": "validation", "a_replay_max_abs_error": replay, "constraints": constraints, "deltas": deltas, "metrics": metrics})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("backbone", "heads", "export"):
        child = sub.add_parser(name)
        child.add_argument("--objective", choices=("ce", "rps"), required=True)
        child.add_argument("--seed", choices=(1, 2, 3, 4), type=int, required=True)
        child.add_argument("--out", type=Path, required=True)
        child.add_argument("--data-root", type=Path, required=True, help="UTKFace image directory matching the frozen manifest")
        child.add_argument("--manifest", type=Path, required=True, help="immutable UTKFace frozen split manifest")
        child.add_argument("--device", default="cuda:0")
        if name in {"heads", "export"}:
            child.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for this prospective protocol")
    if args.command == "backbone":
        run_backbone(args)
    elif args.command == "export":
        run_export(args)
    else:
        run_heads(args)


if __name__ == "__main__":
    main()
