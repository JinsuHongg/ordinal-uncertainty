#!/usr/bin/env python3
"""Seed-0 frozen-feature linear-head audit; never updates a CE/RPS backbone."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.evaluation.frozen_head import (
    class_priors,
    feature_nearest_strata,
    head_loss,
    inverse_frequency_weights,
)
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics, ranked_probability_score
from ordinal_uncertainty.models.ordinal import rps_loss


FEATURE_ROOT = Path("outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified")
HEADS = ("ce", "rps", "balanced_ce", "logit_adjusted")
COVERAGES = np.round(np.arange(1.0, 0.099, -0.05), 2)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("head produced invalid probabilities")
    return probabilities


def decision_rows(method: str, labels: np.ndarray, probabilities: np.ndarray) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    decisions = bayes_decisions(probabilities)
    rows = []
    for decision, key in (("mode", "mode_decision"), ("l1", "l1_bayes_decision"), ("l2", "l2_bayes_decision")):
        prediction = decisions[key]
        error = np.abs(labels - prediction)
        rows.append({
            "condition": method, "decision": decision, "accuracy": float((prediction == labels).mean()),
            "mae": float(error.mean()), "qwk": float(cohen_kappa_score(labels, prediction, weights="quadratic")),
            "severe_count": int((error >= 2).sum()), "severe_prevalence": float((error >= 2).mean()),
        })
    return rows, decisions


def risk_rows(method: str, labels: np.ndarray, decisions: dict[str, np.ndarray]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for decision, key, risk_key in (
        ("mode", "mode_decision", "mode_l1_risk"),
        ("l1", "l1_bayes_decision", "l1_bayes_risk"),
        ("l2", "l2_bayes_decision", "l2_bayes_risk"),
    ):
        prediction, risk = decisions[key], decisions[risk_key]
        error = np.abs(labels - prediction)
        severe = error >= 2
        order = np.argsort(risk, kind="stable")
        condition_coverage = []
        for coverage in COVERAGES:
            retained = order[: max(1, int(np.ceil(coverage * len(labels))))]
            entry = {
                "condition": method, "decision": decision, "coverage": float(coverage),
                "retained_count": int(len(retained)), "ordinal_risk_mae": float(error[retained].mean()),
            }
            condition_coverage.append(entry)
            coverage_rows.append(entry)
        rows.append({
            "condition": method, "decision": decision,
            "spearman": float(spearmanr(risk, error).statistic),
            "severe_auroc": float(roc_auc_score(severe, risk)),
            "severe_auprc": float(average_precision_score(severe, risk)),
            "severe_count": int(severe.sum()), "severe_prevalence": float(severe.mean()),
            "mean_mae_selective_risk": float(np.mean([entry["ordinal_risk_mae"] for entry in condition_coverage])),
            "mode_to_l1_changed_fraction": float((decisions["mode_decision"] != decisions["l1_bayes_decision"]).mean()) if decision == "l1" else None,
        })
    return rows, coverage_rows


def classwise_rows(method: str, labels: np.ndarray, probabilities: np.ndarray, decisions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for decision, key in (("mode", "mode_decision"), ("l1", "l1_bayes_decision"), ("l2", "l2_bayes_decision")):
        prediction = decisions[key]
        for true_class in range(probabilities.shape[1]):
            mask = labels == true_class
            error = np.abs(labels[mask] - prediction[mask])
            rows.append({
                "condition": method, "decision": decision, "true_class": true_class, "count": int(mask.sum()),
                "accuracy": float((prediction[mask] == true_class).mean()), "mae": float(error.mean()),
                "severe_count": int((error >= 2).sum()), "severe_prevalence": float((error >= 2).mean()),
                "mean_l1_bayes_risk": float(decisions["l1_bayes_risk"][mask].mean()),
            })
    return rows


def extreme_rows(method: str, labels: np.ndarray, probabilities: np.ndarray, decisions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    classes = np.arange(probabilities.shape[1])
    predictive_mean = probabilities @ classes
    shrinkage = inward_shrinkage(labels, probabilities)
    l1 = decisions["l1_bayes_decision"]
    rows = []
    for true_class in (0, probabilities.shape[1] - 1):
        mask = labels == true_class
        error = np.abs(l1[mask] - true_class)
        near = probabilities[mask, :2].sum(1) if true_class == 0 else probabilities[mask, -2:].sum(1)
        row = {
            "condition": method, "true_class": true_class, "count": int(mask.sum()),
            "accuracy_l1": float((l1[mask] == true_class).mean()), "mae_l1": float(error.mean()),
            "severe_count_l1": int((error >= 2).sum()), "severe_prevalence_l1": float((error >= 2).mean()),
            "mean_p_true": float(probabilities[mask, true_class].mean()), "median_p_true": float(np.median(probabilities[mask, true_class])),
            "mean_near_mass": float(near.mean()), "median_near_mass": float(np.median(near)),
            "predictive_mean": float(predictive_mean[mask].mean()), "inward_shrinkage": float(shrinkage[mask].mean()),
            "l1_bayes_risk": float(decisions["l1_bayes_risk"][mask].mean()),
        }
        if true_class == probabilities.shape[1] - 1:
            row.update({"mean_p3": float(probabilities[mask, -2].mean()), "median_p3": float(np.median(probabilities[mask, -2]))})
        rows.append(row)
    return rows


def routing_rows(method: str, labels: np.ndarray, decisions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    prediction = decisions["l1_bayes_decision"]
    for true_class in (0, 4):
        mask = labels == true_class
        for predicted_class in range(5):
            count = int((prediction[mask] == predicted_class).sum())
            rows.append({"condition": method, "true_class": true_class, "predicted_class": predicted_class, "count": count, "fraction": count / int(mask.sum())})
    return rows


def stratified_class4_rows(method: str, labels: np.ndarray, probabilities: np.ndarray, decisions: dict[str, np.ndarray], nearest: np.ndarray) -> list[dict[str, object]]:
    strata = feature_nearest_strata(labels, nearest, true_class=4)
    l1 = decisions["l1_bayes_decision"]
    rows = []
    for stratum in ("feature_nearest_true", "feature_nearest_other"):
        mask = strata == stratum
        error = np.abs(l1[mask] - 4)
        rows.append({
            "condition": method, "stratum": stratum, "count": int(mask.sum()),
            "exact_recovery_count": int((l1[mask] == 4).sum()), "adjacent_recovery_count": int((l1[mask] == 3).sum()),
            "exact_or_adjacent_count": int((l1[mask] >= 3).sum()), "severe_count": int((error >= 2).sum()),
            "severe_prevalence": float((error >= 2).mean()), "mean_p4": float(probabilities[mask, 4].mean()),
        })
    return rows


def prediction_rows(sample_ids: np.ndarray, labels: np.ndarray, logits: np.ndarray, probabilities: np.ndarray, decisions: dict[str, np.ndarray]) -> list[dict[str, object]]:
    rows = []
    for index in range(len(labels)):
        rows.append({
            "sample_id": int(sample_ids[index]), "true_label": int(labels[index]),
            "logits": json.dumps(logits[index].tolist()), "probabilities": json.dumps(probabilities[index].tolist()),
            **{key: int(decisions[key][index]) for key in ("mode_decision", "l1_bayes_decision", "l2_bayes_decision")},
            **{key: float(decisions[key][index]) for key in ("mode_l1_risk", "l1_bayes_risk", "l2_bayes_risk")},
        })
    return rows


def evaluate_condition(output: Path, condition: str, sample_ids: np.ndarray, labels: np.ndarray, logits: np.ndarray, nearest: np.ndarray) -> dict[str, list[dict[str, object]]]:
    probabilities = softmax(logits)
    decisions_rows, decisions = decision_rows(condition, labels, probabilities)
    risk, coverage = risk_rows(condition, labels, decisions)
    ece, _ = expected_calibration_error(labels, probabilities)
    predictive = prediction_metrics(labels, probabilities)
    predictive.update({"condition": condition, "ece": float(ece), "probability_sum_error": float(np.abs(probabilities.sum(1) - 1).max())})
    rows = {
        "predictive": [predictive], "decision": decisions_rows, "risk": risk, "coverage": coverage,
        "classwise": classwise_rows(condition, labels, probabilities, decisions),
        "extreme": extreme_rows(condition, labels, probabilities, decisions),
        "routing": routing_rows(condition, labels, decisions),
        "stratified": stratified_class4_rows(condition, labels, probabilities, decisions, nearest),
    }
    write_csv(output / "predictions.csv", prediction_rows(sample_ids, labels, logits, probabilities, decisions))
    for name, values in rows.items():
        write_csv(output / f"{name}_metrics.csv", values)
    (output / "metrics.json").write_text(json.dumps({"condition": condition, "predictive": predictive}, indent=2) + "\n", encoding="utf-8")
    return rows


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as values:
        return {name: values[name] for name in values.files}


def train_head(
    features: dict[str, np.ndarray], objective: str, device: torch.device, epochs: int, batch_size: int, seed: int
) -> tuple[torch.nn.Linear, list[dict[str, float]], int, float, np.ndarray, np.ndarray]:
    train_x = torch.as_tensor(features["train_features"], dtype=torch.float32)
    train_y = torch.as_tensor(features["train_labels"], dtype=torch.long)
    val_x = torch.as_tensor(features["val_features"], dtype=torch.float32, device=device)
    val_y = torch.as_tensor(features["val_labels"], dtype=torch.long, device=device)
    priors_np = class_priors(features["train_labels"], 5)
    weights_np = inverse_frequency_weights(features["train_labels"], 5)
    priors, weights = torch.tensor(priors_np, device=device), torch.tensor(weights_np, device=device)
    torch.manual_seed(seed)
    head = torch.nn.Linear(train_x.shape[1], 5).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train_y), generator=generator)
        total = 0.0
        for indices in order.split(batch_size):
            logits = head(train_x[indices].to(device))
            loss = head_loss(objective, logits, train_y[indices].to(device), weights, priors)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        head.eval()
        with torch.no_grad():
            val_logits = head(val_x)
            selection = rps_loss(val_logits, val_y) if objective == "rps" else torch.nn.functional.cross_entropy(val_logits, val_y)
            val_rps = rps_loss(val_logits, val_y)
        score = float(selection)
        history.append({"epoch": epoch, "training_objective": total / len(train_y), "validation_selection_loss": score, "validation_rps": float(val_rps)})
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    assert best_state is not None
    head.load_state_dict(best_state)
    return head.eval(), history, best_epoch, best_score, priors_np, weights_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/retinamnist/native28/phase3_4_frozen_head_audit"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    args.output.mkdir(parents=True)
    all_rows = {name: [] for name in ("predictive", "decision", "risk", "coverage", "classwise", "extreme", "routing", "stratified")}
    condition_records = []
    head_predictions: dict[str, dict[str, np.ndarray]] = {}
    for source in ("ce", "rps"):
        features = arrays(args.feature_root / source / "seed_0" / "features.npz")
        if not (np.array_equal(features["train_sample_id"], np.arange(1080)) and np.array_equal(features["val_sample_id"], np.arange(120)) and np.array_equal(features["test_sample_id"], np.arange(400))):
            raise ValueError(f"{source} feature IDs are not canonical split indices")
        nearest_table = list(csv.DictReader((args.feature_root / source / "seed_0" / "test_raw_euclidean_per_sample.csv").open(encoding="utf-8")))
        nearest = np.asarray([int(row["nearest_centroid"]) for row in nearest_table])
        labels, sample_ids = features["test_labels"], features["test_sample_id"]
        baseline_dir = args.output / f"{source}_features" / "original_frozen_head"
        baseline_dir.mkdir(parents=True)
        original = evaluate_condition(baseline_dir, f"{source}_features/original_frozen_head", sample_ids, labels, features["test_logits"], nearest)
        for name, rows in original.items(): all_rows[name].extend(rows)
        for head_name in HEADS:
            condition = f"{source}_features/{head_name}_head"
            output = args.output / f"{source}_features" / f"{head_name}_head"
            output.mkdir(parents=True)
            head, history, best_epoch, best_score, priors, weights = train_head(features, head_name, device, args.epochs, args.batch_size, args.seed)
            with torch.no_grad():
                logits = head(torch.as_tensor(features["test_features"], dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
            rows = evaluate_condition(output, condition, sample_ids, labels, logits, nearest)
            for name, values in rows.items(): all_rows[name].extend(values)
            torch.save({"head_state_dict": head.state_dict(), "feature_source": source, "objective": head_name, "frozen_backbone": True, "feature_dimension": int(features["train_features"].shape[1]), "best_epoch": best_epoch, "best_validation_score": best_score}, output / "best_head_checkpoint.pt")
            write_csv(output / "training_history.csv", history)
            config = {"feature_source": source, "head": "linear_512_to_5", "objective": head_name, "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "training_class_priors": priors.tolist(), "balanced_ce_weights": weights.tolist(), "logit_adjustment": "CE(z + log(pi_train), y), tau=1; probabilities use unadjusted z" if head_name == "logit_adjusted" else None, "selection_metric": "validation RPS" if head_name == "rps" else "validation ordinary CE", "best_epoch": best_epoch, "best_validation_score": best_score, "frozen_backbone": True, "backbone_updates": False}
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            head_predictions[condition] = bayes_decisions(softmax(logits))
            condition_records.append({"condition": condition, "best_epoch": best_epoch, "best_validation_score": best_score})
    summary = args.output / "summary"; summary.mkdir()
    for name, rows in all_rows.items(): write_csv(summary / f"{name}.csv", rows)
    write_csv(summary / "checkpoint_selection.csv", condition_records)
    representation_limited = []
    for source in ("ce", "rps"):
        features = arrays(args.feature_root / source / "seed_0" / "features.npz")
        nearest = np.asarray([int(row["nearest_centroid"]) for row in csv.DictReader((args.feature_root / source / "seed_0" / "test_raw_euclidean_per_sample.csv").open(encoding="utf-8"))])
        target = (features["test_labels"] == 4) & (nearest != 4)
        predictions = [head_predictions[f"{source}_features/{head}_head"]["l1_bayes_decision"] for head in HEADS]
        any_near_recovery = np.logical_or.reduce([prediction[target] >= 3 for prediction in predictions])
        representation_limited.append({"feature_source": source, "feature_nearest_other_count": int(target.sum()), "none_of_four_heads_adjacent_or_exact_count": int((~any_near_recovery).sum()), "none_of_four_heads_adjacent_or_exact_fraction": float((~any_near_recovery).mean())})
    write_csv(summary / "representation_limited_subset.csv", representation_limited)
    metadata = {"training_performed": "linear heads only; frozen CE/RPS backbones were not loaded or updated", "feature_root": str(args.feature_root), "feature_dimension": 512, "split_counts": {"train": 1080, "val": 120, "test": 400}, "prior_adjustment_source": "Menon et al., ICLR 2021: training-time CE(z + tau log(pi_train), y), tau=1", "backbone_updates": False, "seeds_launched": [0]}
    (summary / "phase3_4_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
