#!/usr/bin/env python3
"""Training-only Phase 3.10A frozen RPS-head OOF falsification.

This script intentionally never loads the RetinaMNIST validation or test split.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.evaluation.oof import balanced_batch_indices, natural_batch_indices, stratified_five_fold_assignments
from ordinal_uncertainty.evaluation.risk_order import detached_l1_bayes_risk, risk_order_preservation_loss
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics


FEATURES = Path("outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/rps/seed_0/features.npz")
CHECKPOINT = Path("outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt")
OUTPUT = Path("outputs/retinamnist/phase3_10a_rop_objective_falsification")
LAMBDAS = (0.1, 0.5, 1.0)
SEED = 0
EPOCHS = 100
BATCH_SIZE = 64
COVERAGES = np.round(np.arange(1.0, 0.099, -0.05), 2)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    values = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(values); return values / values.sum(axis=1, keepdims=True)


def load_training_arrays() -> dict[str, np.ndarray]:
    """Load only train-prefixed replay-verified frozen arrays."""
    with np.load(FEATURES) as archive:
        required = ("train_sample_id", "train_labels", "train_features", "train_logits", "train_probabilities")
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"feature archive missing train arrays: {sorted(missing)}")
        arrays = {name.removeprefix("train_"): archive[name] for name in required}
        forbidden = [name for name in archive.files if name.startswith(("val_", "test_"))]
    if arrays["features"].shape != (1080, 512):
        raise ValueError(f"unexpected frozen feature shape: {arrays['features'].shape}")
    if not np.array_equal(arrays["sample_id"], np.arange(1080)):
        raise ValueError("training sample IDs are not canonical 0..1079")
    if np.bincount(arrays["labels"], minlength=5).tolist() != [486, 128, 206, 194, 66]:
        raise ValueError("canonical training counts do not match")
    if forbidden:  # Evidence that those arrays were not consumed is recorded in metadata.
        arrays["available_but_unread_split_arrays"] = np.asarray(forbidden, dtype=str)
    return arrays


def original_head_state() -> dict[str, torch.Tensor]:
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved.get("state_dict"))
    if state is None or "fc.weight" not in state or "fc.bias" not in state:
        raise ValueError("canonical RPS checkpoint lacks fc weights")
    if tuple(state["fc.weight"].shape) != (5, 512) or tuple(state["fc.bias"].shape) != (5,):
        raise ValueError("canonical RPS head has unexpected dimensions")
    return {"weight": state["fc.weight"].detach().clone(), "bias": state["fc.bias"].detach().clone()}


def train_fold(features: np.ndarray, labels: np.ndarray, teacher_probabilities: np.ndarray, fit: np.ndarray, state: dict[str, torch.Tensor], lam: float, fold: int, device: torch.device) -> tuple[np.ndarray, list[dict[str, float]], dict[str, torch.Tensor]]:
    """Fit one head with separate balanced and natural batch streams."""
    random.seed(SEED + fold); np.random.seed(SEED + fold); torch.manual_seed(SEED + fold)
    x = torch.as_tensor(features[fit], dtype=torch.float32)
    y = torch.as_tensor(labels[fit], dtype=torch.long)
    teacher = torch.as_tensor(teacher_probabilities[fit], dtype=torch.float32)
    head = torch.nn.Linear(x.shape[1], 5).to(device)
    head.load_state_dict(state)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    balanced_generator = torch.Generator().manual_seed(10_000 + fold)
    natural_generator = torch.Generator().manual_seed(20_000 + fold)
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train(); totals = {"ce": 0.0, "rop": 0.0, "viol": 0.0, "wviol": 0.0, "delta": 0.0, "ties": 0.0, "steps": 0.0}
        natural_batches = natural_batch_indices(len(y), BATCH_SIZE, natural_generator)
        for natural_indices in natural_batches:
            balanced_indices = balanced_batch_indices(y, len(natural_indices), balanced_generator)
            balanced_logits = head(x[balanced_indices].to(device))
            ce = torch.nn.functional.cross_entropy(balanced_logits, y[balanced_indices].to(device))
            if lam:
                natural_probabilities = torch.softmax(head(x[natural_indices].to(device)), dim=1)
                student_risk, _ = detached_l1_bayes_risk(natural_probabilities)
                teacher_risk, _ = detached_l1_bayes_risk(teacher[natural_indices].to(device))
                rop, diag = risk_order_preservation_loss(student_risk, teacher_risk)
            else:
                rop = ce.detach() * 0.0
                diag = {key: torch.zeros((), device=device) for key in ("violation_fraction", "weighted_violation_fraction", "mean_absolute_teacher_difference", "tied_pair_fraction")}
            optimizer.zero_grad(set_to_none=True); (ce + lam * rop).backward(); optimizer.step()
            totals["ce"] += float(ce.detach()); totals["rop"] += float(rop.detach()); totals["viol"] += float(diag["violation_fraction"]); totals["wviol"] += float(diag["weighted_violation_fraction"]); totals["delta"] += float(diag["mean_absolute_teacher_difference"]); totals["ties"] += float(diag["tied_pair_fraction"]); totals["steps"] += 1
        history.append({"fold": fold, "lambda": lam, "epoch": epoch, "mean_balanced_ce": totals["ce"] / totals["steps"], "mean_rop_loss": totals["rop"] / totals["steps"], "natural_pair_violation_fraction": totals["viol"] / totals["steps"], "weighted_natural_pair_violation_fraction": totals["wviol"] / totals["steps"], "mean_absolute_teacher_risk_difference": totals["delta"] / totals["steps"], "natural_pair_tied_fraction": totals["ties"] / totals["steps"]})
    with torch.no_grad():
        logits = head(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()
    return logits, history, {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}


def evaluate(condition: str, lam: float | None, ids: np.ndarray, folds: np.ndarray, labels: np.ndarray, logits: np.ndarray, teacher_risk: np.ndarray) -> tuple[list[dict[str, object]], dict[str, object]]:
    probabilities = softmax(logits); decisions = bayes_decisions(probabilities); rows = []
    for index in range(len(labels)):
        rows.append({"sample_id": int(ids[index]), "fold": int(folds[index]), "label": int(labels[index]), "logits": json.dumps(logits[index].tolist()), "probabilities": json.dumps(probabilities[index].tolist()), "mode_decision": int(decisions["mode_decision"][index]), "l1_decision": int(decisions["l1_bayes_decision"][index]), "l2_decision": int(decisions["l2_bayes_decision"][index]), "l1_bayes_risk": float(decisions["l1_bayes_risk"][index]), "reference_teacher_risk": float(teacher_risk[index]), "condition": condition, "lambda": lam})
    def dm(pred: np.ndarray) -> dict[str, float]:
        error = np.abs(labels - pred); return {"accuracy": float((pred == labels).mean()), "mae": float(error.mean()), "qwk": float(cohen_kappa_score(labels, pred, weights="quadratic")), "severe_prevalence": float((error >= 2).mean())}
    global_metrics = {name: dm(decisions[key]) for name, key in (("mode", "mode_decision"), ("l1", "l1_bayes_decision"), ("l2", "l2_bayes_decision"))}
    error = np.abs(labels - decisions["l1_bayes_decision"]); severe = error >= 2; risk = decisions["l1_bayes_risk"]
    order = np.argsort(risk, kind="stable"); coverage = [{"coverage": float(c), "ordinal_mae": float(error[order[:max(1, int(np.ceil(c * len(labels))))]].mean())} for c in COVERAGES]
    probability = prediction_metrics(labels, probabilities); probability["ece"] = expected_calibration_error(labels, probabilities)[0]
    risk_metrics = {"spearman": float(spearmanr(risk, error).statistic), "severe_auroc": float(roc_auc_score(severe, risk)), "severe_auprc": float(average_precision_score(severe, risk)), "mean_selective_mae": float(np.mean([r["ordinal_mae"] for r in coverage]))}
    endpoints = {}
    for cls in (0, 4):
        mask = labels == cls; pred = decisions["l1_bayes_decision"][mask]; p = probabilities[mask]; e = np.abs(pred - cls); mean = p @ np.arange(5)
        endpoints[str(cls)] = {"support": int(mask.sum()), "routing_l1": [int((pred == j).sum()) for j in range(5)], "accuracy": float((pred == cls).mean()), "mae": float(e.mean()), "severe_prevalence": float((e >= 2).mean()), "predictive_mean": float(mean.mean()), "inward_displacement": float(mean.mean()) if cls == 0 else float(4 - mean.mean()), "mean_l1_bayes_risk": float(risk[mask].mean())}
        if cls == 0: endpoints[str(cls)].update({"mean_p0": float(p[:, 0].mean()), "mean_p1": float(p[:, 1].mean()), "mean_p0_p1": float(p[:, :2].sum(1).mean())})
        else: endpoints[str(cls)].update({"mean_p4": float(p[:, 4].mean()), "median_p4": float(np.median(p[:, 4])), "mean_p3": float(p[:, 3].mean()), "median_p3": float(np.median(p[:, 3])), "mean_p3_p4": float(p[:, 3:].sum(1).mean()), "median_p3_p4": float(np.median(p[:, 3:].sum(1))), "median_l1_bayes_risk": float(np.median(risk[mask]))})
    i, j = np.triu_indices(len(labels), k=1); delta = teacher_risk[i] - teacher_risk[j]; valid = delta != 0; same = np.sign(delta[valid]) == np.sign(risk[i][valid] - risk[j][valid]); weights = np.abs(delta[valid])
    pair = {"pair_order_preserved_fraction": float(same.mean()), "weighted_pair_order_preserved_fraction": float(weights[same].sum() / weights.sum()), "valid_pair_count": int(valid.sum()), "tied_pair_fraction": float(1 - valid.mean())}
    return rows, {"condition": condition, "lambda": lam, "global": global_metrics, "probability": probability, "risk": risk_metrics, "risk_coverage": coverage, "class_0": endpoints["0"], "class_4": endpoints["4"], "pair_order": pair}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="resume only this script's incomplete output")
    parser.add_argument("--restore-balanced-head-checkpoints", action="store_true", help="integrity repair: recreate and verify missing B fold states")
    args = parser.parse_args()
    # This tiny 512-D linear-head workload is substantially slower when small
    # matrix operations fan out across the host's many CPU threads.
    torch.set_num_threads(1)
    if OUTPUT.exists() and not args.resume: raise FileExistsError(f"refusing to overwrite existing Phase 3.10A output: {OUTPUT}")
    arrays = load_training_arrays(); state = original_head_state(); OUTPUT.mkdir(parents=True, exist_ok=args.resume)
    ids, labels, features, original_logits, original_probabilities = (arrays[k] for k in ("sample_id", "labels", "features", "logits", "probabilities"))
    teacher_risk = bayes_decisions(original_probabilities)["l1_bayes_risk"]
    folds = stratified_five_fold_assignments(labels, SEED)
    fold_rows = [{"sample_id": int(ids[n]), "label": int(labels[n]), "fold": int(folds[n])} for n in range(len(ids))]
    if not args.resume:
        write_csv(OUTPUT / "fold_assignments" / "assignments.csv", fold_rows)
        write_csv(OUTPUT / "fold_assignments" / "per_class_counts.csv", [{"fold": fold, "class": cls, "count": int(((folds == fold) & (labels == cls)).sum())} for fold in range(5) for cls in range(5)])
    (OUTPUT / "frozen_features").mkdir(exist_ok=True)
    if not args.resume:
        np.savez_compressed(OUTPUT / "frozen_features" / "train_rps_features.npz", sample_id=ids, labels=labels, features=features, original_logits=original_logits, original_probabilities=original_probabilities, original_mode_decision=bayes_decisions(original_probabilities)["mode_decision"], original_l1_decision=bayes_decisions(original_probabilities)["l1_bayes_decision"], original_l2_decision=bayes_decisions(original_probabilities)["l2_bayes_decision"], original_l1_bayes_risk=teacher_risk)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.restore_balanced_head_checkpoints:
        saved = {int(row["sample_id"]): np.asarray(json.loads(row["logits"]), dtype=np.float32) for row in csv.DictReader((OUTPUT / "condition_b_balanced_head" / "oof_predictions.csv").open(encoding="utf-8"))}
        checkpoint_dir = OUTPUT / "condition_b_balanced_head" / "fold_checkpoints"
        if checkpoint_dir.exists():
            raise FileExistsError("refusing to overwrite existing balanced-head checkpoint repair")
        repaired = []
        for fold in range(5):
            fit, held_out = folds != fold, folds == fold
            logits, _, head_state = train_fold(features, labels, original_probabilities, fit, state, 0.0, fold, device)
            expected = np.stack([saved[int(sample_id)] for sample_id in ids[held_out]])
            maximum_error = float(np.abs(logits[held_out] - expected).max())
            if maximum_error > 1e-6:
                raise RuntimeError(f"fold {fold} integrity replay mismatch: {maximum_error}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"head_state_dict": head_state, "fold": fold, "condition": "B_balanced_head", "verified_oof_max_logit_error": maximum_error, "frozen_backbone": True}, checkpoint_dir / f"fold_{fold}.pt")
            repaired.append({"fold": fold, "verified_oof_max_logit_error": maximum_error})
        write_csv(checkpoint_dir / "integrity_replay.csv", repaired)
        return
    all_results = []
    for condition, lam, location, fixed_logits in (("A_original_rps", None, OUTPUT / "condition_a_original_rps", original_logits), ("B_balanced_head", 0.0, OUTPUT / "condition_b_balanced_head", None), *[(f"C_rop_lambda_{str(value).replace('.', 'p')}", value, OUTPUT / "condition_c_rop" / f"lambda_{str(value).replace('.', 'p')}", None) for value in LAMBDAS]):
        location.mkdir(parents=True, exist_ok=True)
        if args.resume and (location / "summary.json").is_file():
            all_results.append(json.loads((location / "summary.json").read_text(encoding="utf-8")))
            continue
        histories = []
        if fixed_logits is None:
            oof_logits = np.empty((len(labels), 5), dtype=np.float32)
            for fold in range(5):
                fit, held_out = folds != fold, folds == fold
                logits, history, _ = train_fold(features, labels, original_probabilities, fit, state, float(lam), fold, device)
                oof_logits[held_out] = logits[held_out]; histories.extend(history)
            write_csv(location / "training_diagnostics.csv", histories)
        else: oof_logits = fixed_logits
        rows, result = evaluate(condition, lam, ids, folds, labels, oof_logits, teacher_risk)
        write_csv(location / "oof_predictions.csv", rows); write_csv(OUTPUT / "oof_predictions" / f"{condition}.csv", rows)
        (location / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        all_results.append(result)
    metadata = {"phase": "3.10A", "training_only": True, "validation_loaded": False, "test_loaded": False, "feature_archive": str(FEATURES), "checkpoint": str(CHECKPOINT), "feature_dimension": int(features.shape[1]), "train_samples": int(len(labels)), "fold_seed": SEED, "head_initialization": "canonical RPS checkpoint fc.weight/fc.bias", "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "batch_size": BATCH_SIZE, "epochs": EPOCHS, "lambdas": list(LAMBDAS), "backbone_updates": False, "representation_method_added": False, "rg_acr_variant_added": False, "class_4_specific_weighting": False}
    (OUTPUT / "summary" / "metadata.json").parent.mkdir(parents=True, exist_ok=True); (OUTPUT / "summary" / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "summary" / "oof_summary.json").write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
