#!/usr/bin/env python3
"""Phase 3.19 training-signal factorial on frozen RetinaMNIST RPS features.

Only the established training archive and saved five-fold assignments are read.
The C cell is replayed from its valid Phase 3.10C artifact; E/F/G are the three
previously unrun factorial cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices, natural_batch_indices
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics
from ordinal_uncertainty.models.ordinal import rps_loss


ROOT = Path("outputs/retinamnist")
P10A = ROOT / "phase3_10a_rop_objective_falsification"
P10C = ROOT / "phase3_10c_direction_only_head"
P33 = ROOT / "native28/phase3_3_representation_audit_replay_verified/rps/seed_0/features.npz"
CHECKPOINT = ROOT / "native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt"
OUT = ROOT / "phase3_19_sampling_objective_direction_disentanglement"
EPOCHS, BATCH_SIZE = 100, 64
CELLS = {
    "C_balanced_ce": ("balanced", "ce"),
    "E_natural_ce": ("natural", "ce"),
    "F_balanced_rps": ("balanced", "rps"),
    "G_natural_rps": ("natural", "rps"),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def endpoint_metrics(labels: np.ndarray, probabilities: np.ndarray, l1: np.ndarray, class_index: int) -> dict:
    mask = labels == class_index
    expected = probabilities[mask] @ np.arange(5)
    error = np.abs(l1[mask] - class_index)
    risk = bayes_decisions(probabilities)["l1_bayes_risk"][mask]
    result = {
        "support": int(mask.sum()), "routing_l1": [int((l1[mask] == value).sum()) for value in range(5)],
        "exact": int((l1[mask] == class_index).sum()), "mae": float(error.mean()),
        "severe_prevalence": float((error >= 2).mean()), "predictive_mean": float(expected.mean()),
        "mean_l1_risk": float(risk.mean()), "median_l1_risk": float(np.median(risk)),
    }
    if class_index == 4:
        result.update({
            "mean_p4": float(probabilities[mask, 4].mean()), "median_p4": float(np.median(probabilities[mask, 4])),
            "mean_p3": float(probabilities[mask, 3].mean()), "median_p3": float(np.median(probabilities[mask, 3])),
            "mean_p3_p4": float(probabilities[mask, 3:].sum(1).mean()),
            "median_p3_p4": float(np.median(probabilities[mask, 3:].sum(1))),
            "inward_shrinkage": float(4 - expected.mean()),
        })
    else:
        result.update({
            "mean_p0": float(probabilities[mask, 0].mean()), "median_p0": float(np.median(probabilities[mask, 0])),
            "mean_p1": float(probabilities[mask, 1].mean()), "median_p1": float(np.median(probabilities[mask, 1])),
            "mean_p0_p1": float(probabilities[mask, :2].sum(1).mean()),
            "median_p0_p1": float(np.median(probabilities[mask, :2].sum(1))),
        })
    return result


def report(name: str, labels: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, dict, dict[str, np.ndarray]]:
    probabilities = softmax(logits)
    decisions = bayes_decisions(probabilities)
    l1 = decisions["l1_bayes_decision"]
    error = np.abs(labels - l1)
    severe = error >= 2
    ordered = np.argsort(decisions["l1_bayes_risk"])
    selective = [error[ordered[:max(1, int(np.ceil(coverage * len(labels))))]].mean() for coverage in np.arange(1, .09, -.05)]
    global_rows = []
    for label, key in (("mode", "mode_decision"), ("l1", "l1_bayes_decision"), ("l2", "l2_bayes_decision")):
        decision = decisions[key]
        decision_error = np.abs(labels - decision)
        global_rows.append({"condition": name, "decision": label, "accuracy": float((decision == labels).mean()),
                            "mae": float(decision_error.mean()), "qwk": float(cohen_kappa_score(labels, decision, weights="quadratic")),
                            "severe_prevalence": float((decision_error >= 2).mean())})
    probability = prediction_metrics(labels, probabilities)
    probability["ece"] = expected_calibration_error(labels, probabilities)[0]
    return probabilities, {"condition": name, "global": global_rows, "probability": probability,
                            "risk": {"spearman": float(spearmanr(decisions["l1_bayes_risk"], error).statistic),
                                     "severe_auroc": float(roc_auc_score(severe, decisions["l1_bayes_risk"])),
                                     "severe_auprc": float(average_precision_score(severe, decisions["l1_bayes_risk"])),
                                     "mean_selective_mae": float(np.mean(selective)), "ordinal_mae_risk_coverage": [float(x) for x in selective]},
                            "class4": endpoint_metrics(labels, probabilities, l1, 4),
                            "class0": endpoint_metrics(labels, probabilities, l1, 0)}, decisions


def loss_for(objective: str, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if objective == "ce":
        return torch.nn.functional.cross_entropy(logits, labels)
    if objective == "rps":
        return rps_loss(logits, labels)
    raise ValueError(f"unsupported objective {objective}")


def margin_rows(labels: np.ndarray, logits: np.ndarray, condition: str) -> list[dict]:
    rows = []
    for target, left, right in ((4, 4, 3), (4, 4, 2), (0, 0, 1)):
        values = logits[labels == target, left] - logits[labels == target, right]
        rows.append({"condition": condition, "target": target, "margin": f"z{left}-z{right}", "mean": float(values.mean()),
                     "median": float(np.median(values)), "fraction_positive": float((values > 0).mean())})
    return rows


def load_reused_c(ids: np.ndarray, labels: np.ndarray, folds: np.ndarray, features: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> tuple[np.ndarray, list[dict], list[dict], float, float]:
    rows = list(csv.DictReader((P10C / "oof_predictions/predictions.csv").open()))
    if len(rows) != len(ids) or any(int(row["sample_id"]) != int(ids[index]) or int(row["label"]) != int(labels[index]) or int(row["fold"]) != int(folds[index]) for index, row in enumerate(rows)):
        raise RuntimeError("Phase 3.10C OOF provenance does not align")
    logits = np.asarray([json.loads(row["logits"]) for row in rows], dtype=np.float32)
    parameters, integrity = [], []
    max_replay, max_norm = 0.0, 0.0
    for fold in range(5):
        payload = torch.load(P10C / "fold_states" / f"fold_{fold}.pt", map_location="cpu", weights_only=False)
        head = DirectionOnlyLinear(weight, bias)
        head.load_state_dict(payload["head_state_dict"])
        held = folds == fold
        with torch.no_grad():
            replay = head(torch.from_numpy(features[held])).numpy()
            final_weight = head.effective_weight().numpy()
        replay_error = float(np.abs(replay - logits[held]).max())
        norm_error = float(head.max_norm_error().detach())
        bias_error = float((head.fixed_bias - bias).abs().max())
        max_replay, max_norm = max(max_replay, replay_error), max(max_norm, norm_error)
        if replay_error > 2e-6 or norm_error > 1e-6 or bias_error != 0.0:
            raise RuntimeError(f"invalid reused C state in fold {fold}")
        torch.save(payload, OUT / "head_states" / f"C_balanced_ce_fold_{fold}.pt")
        for class_index in range(5):
            cosine = float(np.dot(final_weight[class_index], weight[class_index].numpy()) / (np.linalg.norm(final_weight[class_index]) * float(weight[class_index].norm())))
            parameters.append({"condition": "C_balanced_ce", "fold": fold, "class": class_index, "original_norm": float(weight[class_index].norm()),
                               "final_norm": float(np.linalg.norm(final_weight[class_index])), "norm_error": abs(float(np.linalg.norm(final_weight[class_index])) - float(weight[class_index].norm())),
                               "original_bias": float(bias[class_index]), "final_bias": float(head.fixed_bias[class_index]), "bias_error": abs(float(head.fixed_bias[class_index] - bias[class_index])),
                               "cosine_vs_original": cosine, "angle_from_original_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))) )})
        integrity.append({"condition": "C_balanced_ce", "fold": fold, "sampling": "balanced", "objective": "ce", "initial_replay_error": float(payload["max_init_logit_error"]), "final_oof_replay_error": replay_error, "max_norm_error": norm_error, "max_bias_error": bias_error})
    return logits, parameters, integrity, max_replay, max_norm


def train_cell(name: str, sampling: str, objective: str, ids: np.ndarray, labels: np.ndarray, features: np.ndarray, folds: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> tuple[np.ndarray, list[dict], list[dict], list[dict]]:
    logits = np.empty((len(labels), 5), dtype=np.float32)
    # C was validly reused rather than retrained; retain its original per-fold
    # loss history with an explicit cell label in this phase's artifact.
    histories: list[dict] = [
        {"condition": "C_balanced_ce", "fold": int(row["fold"]), "epoch": int(row["epoch"]),
         "sampling": "balanced", "objective": "ce", "loss": float(row["balanced_ce"]),
         "max_norm_error": float(row["max_norm_error"]), "reused_from_phase": "3.10C"}
        for row in csv.DictReader((P10C / "summary/training.csv").open())
    ]
    parameters: list[dict] = []
    integrity: list[dict] = []
    for fold in range(5):
        fitting, held = folds != fold, folds == fold
        random.seed(fold); np.random.seed(fold); torch.manual_seed(fold)
        head = DirectionOnlyLinear(weight, bias)
        with torch.no_grad():
            init_error = float((head(torch.from_numpy(features[held])) - (torch.from_numpy(features[held]) @ weight.T + bias)).abs().max())
        optimizer = torch.optim.AdamW([head.direction], lr=.001, weight_decay=0.0)
        train_x, train_y = torch.from_numpy(features[fitting]), torch.from_numpy(labels[fitting])
        generator = torch.Generator().manual_seed(10000 + fold)
        for epoch in range(1, EPOCHS + 1):
            losses = []
            if sampling == "balanced":
                batches = (balanced_batch_indices(train_y, BATCH_SIZE, generator) for _ in range(int(np.ceil(len(train_y) / BATCH_SIZE))))
            else:
                batches = natural_batch_indices(len(train_y), BATCH_SIZE, generator)
            for indices in batches:
                loss = loss_for(objective, head(train_x[indices]), train_y[indices])
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
            norm_error = float(head.max_norm_error().detach())
            if norm_error > 1e-6:
                raise RuntimeError(f"norm invariant failed for {name}, fold {fold}")
            histories.append({"condition": name, "fold": fold, "epoch": epoch, "sampling": sampling, "objective": objective, "loss": float(np.mean(losses)), "max_norm_error": norm_error})
        with torch.no_grad():
            logits[held] = head(torch.from_numpy(features[held])).numpy()
            final_weight = head.effective_weight().numpy()
        bias_error = float((head.fixed_bias - bias).abs().max())
        torch.save({"condition": name, "fold": fold, "head_state_dict": head.state_dict(), "sampling": sampling, "objective": objective,
                    "fixed_original_norms": weight.norm(dim=1).tolist(), "fixed_original_bias": bias.tolist(), "max_initial_logit_error": init_error,
                    "max_norm_error": float(head.max_norm_error().detach())}, OUT / "head_states" / f"{name}_fold_{fold}.pt")
        for class_index in range(5):
            cosine = float(np.dot(final_weight[class_index], weight[class_index].numpy()) / (np.linalg.norm(final_weight[class_index]) * float(weight[class_index].norm())))
            parameters.append({"condition": name, "fold": fold, "class": class_index, "original_norm": float(weight[class_index].norm()),
                               "final_norm": float(np.linalg.norm(final_weight[class_index])), "norm_error": abs(float(np.linalg.norm(final_weight[class_index])) - float(weight[class_index].norm())),
                               "original_bias": float(bias[class_index]), "final_bias": float(head.fixed_bias[class_index]), "bias_error": abs(float(head.fixed_bias[class_index] - bias[class_index])),
                               "cosine_vs_original": cosine, "angle_from_original_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))) )})
        integrity.append({"condition": name, "fold": fold, "sampling": sampling, "objective": objective, "initial_replay_error": init_error,
                          "final_oof_replay_error": 0.0, "max_norm_error": float(head.max_norm_error()), "max_bias_error": bias_error})
    return logits, histories, parameters, integrity


def replay_saved_cell(name: str, sampling: str, objective: str, features: np.ndarray, folds: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> tuple[np.ndarray, list[dict], list[dict]]:
    logits = np.empty((len(features), 5), dtype=np.float32)
    parameters, integrity = [], []
    for fold in range(5):
        payload = torch.load(OUT / "head_states" / f"{name}_fold_{fold}.pt", map_location="cpu", weights_only=False)
        head = DirectionOnlyLinear(weight, bias); head.load_state_dict(payload["head_state_dict"])
        held = folds == fold
        with torch.no_grad():
            logits[held] = head(torch.from_numpy(features[held])).numpy()
            final_weight = head.effective_weight().numpy()
        norm_error, bias_error = float(head.max_norm_error().detach()), float((head.fixed_bias - bias).abs().max())
        if norm_error > 1e-6 or bias_error != 0.0:
            raise RuntimeError(f"invalid saved {name} state in fold {fold}")
        integrity.append({"condition": name, "fold": fold, "sampling": sampling, "objective": objective, "initial_replay_error": float(payload["max_initial_logit_error"]), "final_oof_replay_error": 0.0, "max_norm_error": norm_error, "max_bias_error": bias_error})
        for class_index in range(5):
            cosine = float(np.dot(final_weight[class_index], weight[class_index].numpy()) / (np.linalg.norm(final_weight[class_index]) * float(weight[class_index].norm())))
            parameters.append({"condition": name, "fold": fold, "class": class_index, "original_norm": float(weight[class_index].norm()), "final_norm": float(np.linalg.norm(final_weight[class_index])), "norm_error": abs(float(np.linalg.norm(final_weight[class_index])) - float(weight[class_index].norm())), "original_bias": float(bias[class_index]), "final_bias": float(head.fixed_bias[class_index]), "bias_error": abs(float(head.fixed_bias[class_index] - bias[class_index])), "cosine_vs_original": cosine, "angle_from_original_degrees": float(np.degrees(np.arccos(np.clip(cosine, -1, 1))) )})
    return logits, parameters, integrity


def subgroup_rows(labels: np.ndarray, nearest: np.ndarray, decisions: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for condition, decision in decisions.items():
        for name, mask in (("feature_nearest_4", (labels == 4) & (nearest == 4)), ("representation_inward", (labels == 4) & (nearest != 4))):
            routing = decision[mask]
            rows.append({"condition": condition, "subgroup": name, "support": int(mask.sum()), "routing_l1": [int((routing == x).sum()) for x in range(5)],
                         "exact": int((routing == 4).sum()), "mae": float(np.abs(4 - routing).mean()), "outward_routing": int((routing >= 3).sum())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("E_natural_ce", "F_balanced_rps", "G_natural_rps"))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.only and args.finalize:
        raise ValueError("--only and --finalize are mutually exclusive")
    if OUT.exists() and not (args.only or args.finalize):
        raise FileExistsError(f"refusing to overwrite historical or partial result: {OUT}")
    for directory in ("metadata", "head_states", "histories", "predictions", "metrics", "parameters", "subgroups", "summary"):
        (OUT / directory).mkdir(parents=True, exist_ok=args.only or args.finalize)
    torch.set_num_threads(1)
    with np.load(P10A / "frozen_features/train_rps_features.npz") as archive:
        ids, labels, features, original_logits = (archive[key].astype(np.int64 if key in ("sample_id", "labels") else np.float32) for key in ("sample_id", "labels", "features", "original_logits"))
    fold_rows = list(csv.DictReader((P10A / "fold_assignments/assignments.csv").open()))
    if features.shape != (1080, 512) or len(fold_rows) != len(ids) or not np.isfinite(features).all() or any(int(row["sample_id"]) != int(ids[index]) or int(row["label"]) != int(labels[index]) for index, row in enumerate(fold_rows)):
        raise RuntimeError("invalid Phase 3.10A training-only frozen RPS provenance")
    folds = np.asarray([int(row["fold"]) for row in fold_rows], dtype=np.int64)
    if set(folds) != set(range(5)) or np.any(np.bincount(folds, minlength=5) == 0):
        raise RuntimeError("invalid saved OOF fold assignments")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    weight, bias = state["fc.weight"].float(), state["fc.bias"].float()
    with torch.no_grad():
        a_logits = (torch.from_numpy(features) @ weight.T + bias).numpy()
    a_replay_error = float(np.abs(a_logits - original_logits).max())
    if a_replay_error > 2e-5:
        raise RuntimeError(f"original RPS head replay failed: {a_replay_error}")
    c_logits, parameter_rows, integrity_rows, c_replay_error, c_norm_error = load_reused_c(ids, labels, folds, features, weight, bias)
    all_logits = {"A_original_rps": a_logits, "C_balanced_ce": c_logits}
    histories: list[dict] = []
    requested_cells = (args.only,) if args.only else ("E_natural_ce", "F_balanced_rps", "G_natural_rps")
    for name in requested_cells:
        sampling, objective = CELLS[name]
        if args.finalize:
            logits, cell_parameters, cell_integrity = replay_saved_cell(name, sampling, objective, features, folds, weight, bias)
            cell_history = list(csv.DictReader((OUT / "histories" / f"{name}.csv").open()))
        else:
            logits, cell_history, cell_parameters, cell_integrity = train_cell(name, sampling, objective, ids, labels, features, folds, weight, bias)
            if args.only:
                write_csv(OUT / "histories" / f"{name}.csv", cell_history)
        all_logits[name] = logits; histories += cell_history; parameter_rows += cell_parameters; integrity_rows += cell_integrity
    if args.only:
        return
    reports, decisions, margin_output = {}, {}, []
    prediction_rows = []
    for name, logits in all_logits.items():
        probabilities, report_data, decision_data = report(name, labels, logits)
        reports[name] = report_data; decisions[name] = decision_data["l1_bayes_decision"]
        margin_output += margin_rows(labels, logits, name)
        for index in range(len(ids)):
            prediction_rows.append({"condition": name, "sample_id": int(ids[index]), "fold": int(folds[index]), "label": int(labels[index]),
                                    "logits": json.dumps(logits[index].tolist()), "probabilities": json.dumps(probabilities[index].tolist()),
                                    "mode_decision": int(decision_data["mode_decision"][index]), "l1_decision": int(decision_data["l1_bayes_decision"][index]), "l2_decision": int(decision_data["l2_bayes_decision"][index]), "l1_bayes_risk": float(decision_data["l1_bayes_risk"][index])})
    with np.load(P33) as feature_archive:
        if not np.array_equal(feature_archive["train_sample_id"], ids) or not np.array_equal(feature_archive["train_labels"], labels) or not np.allclose(feature_archive["train_features"], features, atol=1e-6):
            raise RuntimeError("Phase 3.3 RPS feature-nearest provenance does not align")
    centroids = np.stack([features[labels == item].mean(axis=0) for item in range(5)])
    nearest = ((features[:, None, :] - centroids[None, :, :]) ** 2).sum(2).argmin(1)
    metric_specs = (("class4_mae", "class4", "mae", "lower"), ("class4_exact", "class4", "exact", "higher"), ("class4_shrinkage", "class4", "inward_shrinkage", "lower"), ("class4_mean_p4", "class4", "mean_p4", "higher"), ("class4_predictive_mean", "class4", "predictive_mean", "higher"), ("class4_z4_minus_z3", None, None, "higher"))
    margin_means = {row["condition"]: row["mean"] for row in margin_output if row["target"] == 4 and row["margin"] == "z4-z3"}
    contrasts = []
    for metric, group, key, preference in metric_specs:
        values = margin_means if group is None else {name: reports[name][group][key] for name in CELLS}
        for contrast, left, right in (("sampling_given_ce_C_minus_E", "C_balanced_ce", "E_natural_ce"), ("sampling_given_rps_F_minus_G", "F_balanced_rps", "G_natural_rps"), ("objective_given_balanced_C_minus_F", "C_balanced_ce", "F_balanced_rps"), ("objective_given_natural_E_minus_G", "E_natural_ce", "G_natural_rps")):
            contrasts.append({"metric": metric, "preference": preference, "contrast": contrast, "left_value": values[left], "right_value": values[right], "left_minus_right": values[left] - values[right]})
    write_csv(OUT / "predictions/oof_predictions.csv", prediction_rows)
    write_csv(OUT / "histories/training.csv", histories)
    write_csv(OUT / "parameters/direction_cosines.csv", parameter_rows)
    write_csv(OUT / "metrics/integrity.csv", integrity_rows)
    write_csv(OUT / "metrics/margins.csv", margin_output)
    write_csv(OUT / "metrics/factorial_contrasts.csv", contrasts)
    write_csv(OUT / "subgroups/feature_nearest_c4.csv", subgroup_rows(labels, nearest, decisions))
    (OUT / "metadata/provenance.json").write_text(json.dumps({"feature_archive": str(P10A / "frozen_features/train_rps_features.npz"), "original_head_checkpoint": str(CHECKPOINT), "fold_assignments": str(P10A / "fold_assignments/assignments.csv"), "reused_c_artifact": str(P10C), "phase3_3_feature_archive": str(P33), "train_samples": int(len(ids)), "feature_shape": list(features.shape), "fold_counts": np.bincount(folds, minlength=5).tolist(), "class_counts": np.bincount(labels, minlength=5).tolist(), "original_head_replay_error": a_replay_error, "reused_c_oof_replay_error": c_replay_error, "reused_c_max_norm_error": c_norm_error, "validation_or_test_loaded": False}, indent=2) + "\n")
    (OUT / "metadata/training_config.json").write_text(json.dumps({"cells": {name: {"sampling": sampling, "objective": objective, "direction_only": True, "fixed_original_norms": True, "fixed_original_biases": True} for name, (sampling, objective) in CELLS.items()}, "optimizer": "AdamW", "learning_rate": .001, "weight_decay": 0.0, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "balanced_sampling": "established replacement balanced_batch_indices", "natural_sampling": "one empirical-distribution shuffled pass per epoch via natural_batch_indices", "seed_policy": "fold seed; sampler generator 10000 + fold", "rps_definition": "ordinal_uncertainty.models.ordinal.rps_loss"}, indent=2) + "\n")
    (OUT / "summary/summary.json").write_text(json.dumps({"A_original_rps": reports["A_original_rps"], "C_balanced_ce": reports["C_balanced_ce"], "E_natural_ce": reports["E_natural_ce"], "F_balanced_rps": reports["F_balanced_rps"], "G_natural_rps": reports["G_natural_rps"], "integrity": {"original_head_replay_error": a_replay_error, "reused_c_oof_replay_error": c_replay_error, "all_fold_assignments_saved": True, "no_duplicate_oof_predictions": True, "validation_or_test_loaded": False}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
