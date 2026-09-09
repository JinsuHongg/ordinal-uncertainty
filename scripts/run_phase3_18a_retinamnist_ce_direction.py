#!/usr/bin/env python3
"""Frozen CE A/B/C OOF robustness gate for Phase 3.18A.

This script deliberately loads only the training members of the established CE
feature archive. It never constructs a validation or test dataset.
"""
from __future__ import annotations

import csv
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics


ROOT = Path("outputs/retinamnist")
FEATURES = ROOT / "native28/phase3_3_representation_audit_replay_verified/ce/seed_0/features.npz"
CHECKPOINT = ROOT / "resolution_sanity_check/seed_0/size_28/best_checkpoint.pt"
FOLDS = ROOT / "phase3_10a_rop_objective_falsification/fold_assignments/assignments.csv"
RPS_C = ROOT / "phase3_10c_direction_only_head/summary/summary.json"
OUT = ROOT / "phase3_18a_ce_direction_robustness"
EPOCHS = 100
BATCH_SIZE = 64
SEED = 0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def endpoint_metrics(labels: np.ndarray, probabilities: np.ndarray, l1: np.ndarray, class_index: int) -> dict:
    mask = labels == class_index
    error = np.abs(l1[mask] - class_index)
    expected = probabilities[mask] @ np.arange(5)
    risks = bayes_decisions(probabilities)["l1_bayes_risk"][mask]
    result = {
        "support": int(mask.sum()),
        "routing_l1": [int((l1[mask] == item).sum()) for item in range(5)],
        "exact": int((l1[mask] == class_index).sum()),
        "mae": float(error.mean()),
        "severe_prevalence": float((error >= 2).mean()),
        "predictive_mean": float(expected.mean()),
        "mean_l1_risk": float(risks.mean()),
        "median_l1_risk": float(np.median(risks)),
    }
    if class_index == 4:
        result.update(
            {
                "mean_p4": float(probabilities[mask, 4].mean()),
                "median_p4": float(np.median(probabilities[mask, 4])),
                "mean_p3": float(probabilities[mask, 3].mean()),
                "median_p3": float(np.median(probabilities[mask, 3])),
                "mean_p3_p4": float(probabilities[mask, 3:].sum(axis=1).mean()),
                "median_p3_p4": float(np.median(probabilities[mask, 3:].sum(axis=1))),
                "inward_shrinkage": float(4 - expected.mean()),
            }
        )
    else:
        result.update(
            {
                "mean_p0": float(probabilities[mask, 0].mean()),
                "median_p0": float(np.median(probabilities[mask, 0])),
                "mean_p1": float(probabilities[mask, 1].mean()),
                "median_p1": float(np.median(probabilities[mask, 1])),
                "mean_p0_p1": float(probabilities[mask, :2].sum(axis=1).mean()),
                "median_p0_p1": float(np.median(probabilities[mask, :2].sum(axis=1))),
            }
        )
    return result


def condition_report(name: str, labels: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    probabilities = softmax(logits)
    decisions = bayes_decisions(probabilities)
    l1 = decisions["l1_bayes_decision"]
    absolute_error = np.abs(labels - l1)
    severe = absolute_error >= 2
    ordered = np.argsort(decisions["l1_bayes_risk"])
    coverages = np.arange(1.0, 0.09, -0.05)
    selective = [absolute_error[ordered[: max(1, int(np.ceil(value * len(labels))))]].mean() for value in coverages]
    global_metrics = []
    for decision_name, key in (("mode", "mode_decision"), ("l1", "l1_bayes_decision"), ("l2", "l2_bayes_decision")):
        decision = decisions[key]
        error = np.abs(labels - decision)
        global_metrics.append(
            {
                "condition": name,
                "decision": decision_name,
                "accuracy": float((decision == labels).mean()),
                "mae": float(error.mean()),
                "qwk": float(cohen_kappa_score(labels, decision, weights="quadratic")),
                "severe_prevalence": float((error >= 2).mean()),
            }
        )
    probability = prediction_metrics(labels, probabilities)
    probability["ece"] = expected_calibration_error(labels, probabilities)[0]
    risk = {
        "spearman": float(spearmanr(decisions["l1_bayes_risk"], absolute_error).statistic),
        "severe_auroc": float(roc_auc_score(severe, decisions["l1_bayes_risk"])),
        "severe_auprc": float(average_precision_score(severe, decisions["l1_bayes_risk"])),
        "mean_selective_mae": float(np.mean(selective)),
        "ordinal_mae_risk_coverage": [float(value) for value in selective],
    }
    return probabilities, {
        "condition": name,
        "global": global_metrics,
        "probability": probability,
        "risk": risk,
        "class4": endpoint_metrics(labels, probabilities, l1, 4),
        "class0": endpoint_metrics(labels, probabilities, l1, 0),
    }, decisions


def margins(labels: np.ndarray, logits: np.ndarray, condition: str) -> list[dict]:
    rows = []
    for target, left, right in ((4, 4, 3), (4, 4, 2), (0, 0, 1)):
        values = logits[labels == target, left] - logits[labels == target, right]
        rows.append(
            {
                "condition": condition,
                "target": target,
                "margin": f"z{left}-z{right}",
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q10": float(np.quantile(values, 0.1)),
                "q90": float(np.quantile(values, 0.9)),
                "fraction_positive": float((values > 0).mean()),
            }
        )
    return rows


def centroid_groups(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroids = np.stack([features[labels == item].mean(axis=0) for item in range(5)])
    nearest = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return nearest, centroids


def subgroup_rows(labels: np.ndarray, nearest: np.ndarray, a_l1: np.ndarray, c_l1: np.ndarray) -> list[dict]:
    rows = []
    for name, mask in (
        ("feature_nearest_4", (labels == 4) & (nearest == 4)),
        ("representation_inward", (labels == 4) & (nearest != 4)),
    ):
        rows.append(
            {
                "subgroup": name,
                "support": int(mask.sum()),
                "nearest_routing": [int((nearest[mask] == item).sum()) for item in range(5)],
                "a_routing_l1": [int((a_l1[mask] == item).sum()) for item in range(5)],
                "c_routing_l1": [int((c_l1[mask] == item).sum()) for item in range(5)],
                "a_exact": int((a_l1[mask] == 4).sum()),
                "c_exact": int((c_l1[mask] == 4).sum()),
                "a_to_c_inward_improved": int((np.abs(4 - c_l1[mask]) < np.abs(4 - a_l1[mask])).sum()),
            }
        )
    return rows


def head_parameters(weight_a: np.ndarray, bias_a: np.ndarray, weight_b: np.ndarray, bias_b: np.ndarray, weight_c: np.ndarray, bias_c: np.ndarray, fold: int) -> list[dict]:
    rows = []
    for item in range(5):
        norm_a, norm_b, norm_c = (float(np.linalg.norm(value[item])) for value in (weight_a, weight_b, weight_c))
        cosine_b = float(np.dot(weight_a[item], weight_b[item]) / (norm_a * norm_b))
        cosine_c = float(np.dot(weight_a[item], weight_c[item]) / (norm_a * norm_c))
        rows.append(
            {
                "fold": fold,
                "class": item,
                "original_norm": norm_a,
                "balanced_norm": norm_b,
                "direction_only_norm": norm_c,
                "balanced_delta_norm": norm_b - norm_a,
                "direction_only_norm_error": abs(norm_c - norm_a),
                "original_bias": float(bias_a[item]),
                "balanced_bias": float(bias_b[item]),
                "direction_only_bias": float(bias_c[item]),
                "balanced_delta_bias": float(bias_b[item] - bias_a[item]),
                "direction_only_bias_error": float(abs(bias_c[item] - bias_a[item])),
                "balanced_cosine_vs_original": cosine_b,
                "direction_only_cosine_vs_original": cosine_c,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(5), help="run one deterministic OOF fold only")
    parser.add_argument("--finalize", action="store_true", help="assemble OOF reports from five saved fold states")
    args = parser.parse_args()
    if args.fold is not None and args.finalize:
        raise ValueError("--fold and --finalize are mutually exclusive")
    if OUT.exists() and args.fold is None and not args.finalize:
        raise FileExistsError(f"refusing to overwrite existing artifact directory: {OUT}")
    for directory in ("metadata", "fold_states", "histories", "predictions", "metrics", "parameters", "subgroups", "summary"):
        (OUT / directory).mkdir(parents=True, exist_ok=True)

    # Intentionally read only canonical training arrays from the established archive.
    archive = np.load(FEATURES)
    ids = archive["train_sample_id"].astype(np.int64)
    labels = archive["train_labels"].astype(np.int64)
    features = archive["train_features"].astype(np.float32)
    cached_logits = archive["train_logits"].astype(np.float32)
    if features.shape != (1080, 512) or labels.shape != (1080,) or not np.isfinite(features).all():
        raise RuntimeError("invalid CE training feature provenance")
    fold_rows = list(csv.DictReader(FOLDS.open()))
    if len(fold_rows) != len(ids) or any(int(row["sample_id"]) != int(ids[index]) or int(row["label"]) != int(labels[index]) for index, row in enumerate(fold_rows)):
        raise RuntimeError("Phase 3.10A OOF folds do not align with CE training features")
    folds = np.asarray([int(row["fold"]) for row in fold_rows], dtype=np.int64)
    if set(folds) != set(range(5)):
        raise RuntimeError("invalid OOF folds")

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    original_weight = state["fc.weight"].detach().float()
    original_bias = state["fc.bias"].detach().float()
    with torch.no_grad():
        a_logits = (torch.from_numpy(features) @ original_weight.T + original_bias).numpy()
    replay_error = float(np.abs(a_logits - cached_logits).max())
    if replay_error > 2e-5:
        raise RuntimeError(f"CE cached-logit replay failed: {replay_error}")

    torch.set_num_threads(1)
    b_logits = np.empty_like(a_logits)
    c_logits = np.empty_like(a_logits)
    parameter_rows: list[dict] = []
    history_rows: list[dict] = []
    maximum_c_initial_error = 0.0
    maximum_c_norm_error = 0.0
    maximum_c_bias_error = 0.0
    if args.finalize:
        for fold in range(5):
            b_payload = torch.load(OUT / "fold_states" / f"B_fold_{fold}.pt", map_location="cpu", weights_only=False)
            c_payload = torch.load(OUT / "fold_states" / f"C_fold_{fold}.pt", map_location="cpu", weights_only=False)
            balanced = torch.nn.Linear(512, 5)
            balanced.load_state_dict(b_payload["head_state_dict"])
            direction_only = DirectionOnlyLinear(original_weight, original_bias)
            direction_only.load_state_dict(c_payload["head_state_dict"])
            held = folds == fold
            with torch.no_grad():
                b_logits[held] = balanced(torch.from_numpy(features[held])).numpy()
                c_logits[held] = direction_only(torch.from_numpy(features[held])).numpy()
                b_weight, b_bias = balanced.weight.numpy(), balanced.bias.numpy()
                c_weight, c_bias = direction_only.effective_weight().numpy(), direction_only.fixed_bias.numpy()
            initial_error = float(c_payload["max_initial_logit_error"])
            norm_error = float(direction_only.max_norm_error().detach())
            bias_error = float((direction_only.fixed_bias - original_bias).abs().max())
            maximum_c_initial_error = max(maximum_c_initial_error, initial_error)
            maximum_c_norm_error = max(maximum_c_norm_error, norm_error)
            maximum_c_bias_error = max(maximum_c_bias_error, bias_error)
            parameter_rows.extend(head_parameters(original_weight.numpy(), original_bias.numpy(), b_weight, b_bias, c_weight, c_bias, fold))
            history_rows.extend(list(csv.DictReader((OUT / "histories" / f"fold_{fold}.csv").open())))
    folds_to_fit = [] if args.finalize else ([args.fold] if args.fold is not None else range(5))
    for fold in folds_to_fit:
        fitting, held = folds != fold, folds == fold
        random.seed(SEED + fold)
        np.random.seed(SEED + fold)
        torch.manual_seed(SEED + fold)
        batch_generator = torch.Generator().manual_seed(10000 + fold)
        fitting_features = torch.from_numpy(features[fitting])
        fitting_labels = torch.from_numpy(labels[fitting])

        balanced = torch.nn.Linear(512, 5)
        with torch.no_grad():
            balanced.weight.copy_(original_weight)
            balanced.bias.copy_(original_bias)
        direction_only = DirectionOnlyLinear(original_weight, original_bias)
        with torch.no_grad():
            initial_error = (direction_only(torch.from_numpy(features[held])) - a_logits[held]).abs().max().item()
        maximum_c_initial_error = max(maximum_c_initial_error, initial_error)
        if initial_error > 2e-5:
            raise RuntimeError(f"direction-only initialization did not replay A in fold {fold}: {initial_error}")
        balanced_optimizer = torch.optim.AdamW(balanced.parameters(), lr=0.001, weight_decay=1e-4)
        direction_optimizer = torch.optim.AdamW([direction_only.direction], lr=0.001, weight_decay=0.0)
        # Separate but identically seeded generators guarantee the same B/C balanced batches.
        direction_generator = torch.Generator().manual_seed(10000 + fold)
        for epoch in range(1, EPOCHS + 1):
            b_losses, c_losses = [], []
            for _ in range(int(np.ceil(len(fitting_labels) / BATCH_SIZE))):
                b_indices = balanced_batch_indices(fitting_labels, BATCH_SIZE, batch_generator)
                c_indices = balanced_batch_indices(fitting_labels, BATCH_SIZE, direction_generator)
                if not torch.equal(b_indices, c_indices):
                    raise RuntimeError("B/C balanced batches diverged")
                b_loss = torch.nn.functional.cross_entropy(balanced(fitting_features[b_indices]), fitting_labels[b_indices])
                balanced_optimizer.zero_grad(set_to_none=True)
                b_loss.backward()
                balanced_optimizer.step()
                c_loss = torch.nn.functional.cross_entropy(direction_only(fitting_features[c_indices]), fitting_labels[c_indices])
                direction_optimizer.zero_grad(set_to_none=True)
                c_loss.backward()
                direction_optimizer.step()
                b_losses.append(float(b_loss.detach()))
                c_losses.append(float(c_loss.detach()))
            norm_error = float(direction_only.max_norm_error().detach())
            bias_error = float((direction_only.fixed_bias - original_bias).abs().max())
            maximum_c_norm_error = max(maximum_c_norm_error, norm_error)
            maximum_c_bias_error = max(maximum_c_bias_error, bias_error)
            if norm_error > 1e-6 or bias_error > 1e-7:
                raise RuntimeError(f"C constraint violation in fold {fold}: norm={norm_error}, bias={bias_error}")
            history_rows.extend(
                [
                    {"condition": "B_CE_balanced", "fold": fold, "epoch": epoch, "balanced_ce": float(np.mean(b_losses))},
                    {"condition": "C_CE_direction_only", "fold": fold, "epoch": epoch, "balanced_ce": float(np.mean(c_losses)), "max_norm_error": norm_error},
                ]
            )
        with torch.no_grad():
            b_logits[held] = balanced(torch.from_numpy(features[held])).numpy()
            c_logits[held] = direction_only(torch.from_numpy(features[held])).numpy()
            b_weight, b_bias = balanced.weight.numpy(), balanced.bias.numpy()
            c_weight, c_bias = direction_only.effective_weight().numpy(), direction_only.fixed_bias.numpy()
        torch.save({"fold": fold, "head_state_dict": balanced.state_dict(), "initial_weight": original_weight, "initial_bias": original_bias}, OUT / "fold_states" / f"B_fold_{fold}.pt")
        torch.save({"fold": fold, "head_state_dict": direction_only.state_dict(), "fixed_original_norms": original_weight.norm(dim=1), "fixed_original_bias": original_bias, "max_initial_logit_error": initial_error, "max_norm_error": norm_error}, OUT / "fold_states" / f"C_fold_{fold}.pt")
        parameter_rows.extend(head_parameters(original_weight.numpy(), original_bias.numpy(), b_weight, b_bias, c_weight, c_bias, fold))

    if args.fold is not None:
        write_csv(OUT / "histories" / f"fold_{args.fold}.csv", history_rows)
        return

    a_probabilities, a_summary, a_decisions = condition_report("A_CE_original", labels, a_logits)
    b_probabilities, b_summary, b_decisions = condition_report("B_CE_balanced", labels, b_logits)
    c_probabilities, c_summary, c_decisions = condition_report("C_CE_direction_only", labels, c_logits)
    for name, logits, probabilities, decisions in (
        ("A_CE_original", a_logits, a_probabilities, a_decisions),
        ("B_CE_balanced", b_logits, b_probabilities, b_decisions),
        ("C_CE_direction_only", c_logits, c_probabilities, c_decisions),
    ):
        write_csv(
            OUT / "predictions" / f"{name}.csv",
            [
                {
                    "sample_id": int(ids[index]), "fold": int(folds[index]), "label": int(labels[index]),
                    "logits": json.dumps(logits[index].tolist()), "probabilities": json.dumps(probabilities[index].tolist()),
                    "mode_decision": int(decisions["mode_decision"][index]), "l1_decision": int(decisions["l1_bayes_decision"][index]),
                    "l2_decision": int(decisions["l2_bayes_decision"][index]), "l1_bayes_risk": float(decisions["l1_bayes_risk"][index]),
                }
                for index in range(len(ids))
            ],
        )
    write_csv(OUT / "histories" / "training.csv", history_rows)
    write_csv(OUT / "parameters" / "per_fold.csv", parameter_rows)
    write_csv(OUT / "metrics" / "global.csv", a_summary["global"] + b_summary["global"] + c_summary["global"])
    write_csv(OUT / "metrics" / "margins.csv", margins(labels, a_logits, "A_CE_original") + margins(labels, b_logits, "B_CE_balanced") + margins(labels, c_logits, "C_CE_direction_only"))
    per_fold_class4 = []
    for name, probabilities, decisions in (
        ("A_CE_original", a_probabilities, a_decisions),
        ("B_CE_balanced", b_probabilities, b_decisions),
        ("C_CE_direction_only", c_probabilities, c_decisions),
    ):
        for fold in range(5):
            held = folds == fold
            values = endpoint_metrics(labels[held], probabilities[held], decisions["l1_bayes_decision"][held], 4)
            per_fold_class4.append({"condition": name, "fold": fold, **values})
    write_csv(OUT / "metrics" / "per_fold_class4.csv", per_fold_class4)
    nearest, centroids = centroid_groups(features, labels)
    write_csv(OUT / "subgroups" / "class4_centroid_groups.csv", subgroup_rows(labels, nearest, a_decisions["l1_bayes_decision"], c_decisions["l1_bayes_decision"]))
    write_csv(OUT / "metadata" / "fold_class_counts.csv", [{"fold": fold, "class": item, "held_out_count": int(((folds == fold) & (labels == item)).sum()), "fitting_count": int(((folds != fold) & (labels == item)).sum())} for fold in range(5) for item in range(5)])
    rps = json.loads(RPS_C.read_text())
    rps_a, rps_c = rps["A"]["class_4"], rps["C"]["class4"]
    cross_objective = [
        {"representation": "RPS_saved", "a_c4_mae": rps_a["mae"], "c_c4_mae": rps_c["mae"], "a_exact": rps_a["accuracy"] * rps_a["support"], "c_exact": rps_c["exact"], "a_shrinkage": rps_a["inward_displacement"], "c_shrinkage": rps_c["inward_shrinkage"], "interpretation": "saved direction effect"},
        {"representation": "CE_new", "a_c4_mae": a_summary["class4"]["mae"], "c_c4_mae": c_summary["class4"]["mae"], "a_exact": a_summary["class4"]["exact"], "c_exact": c_summary["class4"]["exact"], "a_shrinkage": a_summary["class4"]["inward_shrinkage"], "c_shrinkage": c_summary["class4"]["inward_shrinkage"], "interpretation": "Phase 3.18A robustness result"},
    ]
    write_csv(OUT / "summary" / "rps_vs_ce_direction.csv", cross_objective)
    manifest = {
        "phase": "3.18A", "training_performed": "linear heads only on frozen CE features; CE backbone never loaded for training", "test_or_validation_loaded": False,
        "feature_cache": str(FEATURES), "checkpoint": str(CHECKPOINT), "feature_shape": list(features.shape), "feature_dtype": str(features.dtype),
        "fold_source": str(FOLDS), "folds": 5, "seed": SEED, "optimizer": "AdamW", "learning_rate": 0.001, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "balanced_sampling": "balanced_batch_indices on fitting-fold labels only", "B_weight_decay": 1e-4, "C_weight_decay": 0.0,
        "a_cached_logit_replay_max_error": replay_error, "c_initial_logit_replay_max_error": maximum_c_initial_error,
        "c_max_norm_error": maximum_c_norm_error, "c_max_bias_error": maximum_c_bias_error, "centroid_definition": "raw Euclidean training centroids derived from CE training features only",
    }
    (OUT / "metadata" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (OUT / "summary" / "results.json").write_text(json.dumps({"A": a_summary, "B": b_summary, "C": c_summary, "integrity": manifest, "cross_objective": cross_objective}, indent=2) + "\n")


if __name__ == "__main__":
    main()
