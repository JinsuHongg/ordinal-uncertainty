#!/usr/bin/env python3
"""Phase 3.10E: fixed-alpha controlled-scale direction head plus fixed ROP."""
from __future__ import annotations

import argparse, csv, json, random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear
from ordinal_uncertainty.evaluation.oof import balanced_batch_indices, natural_batch_indices
from ordinal_uncertainty.evaluation.risk_order import detached_l1_bayes_risk, risk_order_preservation_loss
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics


BASE = Path("outputs/retinamnist/phase3_10a_rop_objective_falsification")
D_OUT = Path("outputs/retinamnist/phase3_10d_controlled_scale_head")
CKPT = Path("outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt")
OUT = Path("outputs/retinamnist/phase3_10e_controlled_scale_rop_interaction")
ALPHA = 0.5
LAMBDA = 1.0
EPOCHS = 100
BATCH_SIZE = 64


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def probabilities(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    value = np.exp(logits)
    return value / value.sum(axis=1, keepdims=True)


def endpoint(labels: np.ndarray, probs: np.ndarray, decisions: np.ndarray, cls: int) -> dict:
    mask = labels == cls
    error = np.abs(decisions[mask] - cls)
    mean = probs[mask] @ np.arange(5)
    risk = bayes_decisions(probs)["l1_bayes_risk"][mask]
    output = {
        "routing": [int((decisions[mask] == value).sum()) for value in range(5)],
        "mae": float(error.mean()), "exact": int((decisions[mask] == cls).sum()),
        "severe": float((error >= 2).mean()), "predictive_mean": float(mean.mean()),
        "risk": float(risk.mean()),
    }
    if cls == 4:
        output.update(p4=float(probs[mask, 4].mean()), p3=float(probs[mask, 3].mean()),
                      p3p4=float(probs[mask, 3:].sum(axis=1).mean()), shrinkage=float(4 - mean.mean()))
    else:
        output.update(p0=float(probs[mask, 0].mean()), p1=float(probs[mask, 1].mean()),
                      p0p1=float(probs[mask, :2].sum(axis=1).mean()))
    return output


def pair_diagnostics(teacher: np.ndarray, student: np.ndarray) -> dict:
    i, j = np.triu_indices(len(teacher), k=1)
    delta = teacher[i] - teacher[j]
    weights = np.abs(delta)
    valid = weights > 0
    signed = np.sign(delta[valid]) * (student[i][valid] - student[j][valid])
    preserved = signed >= 0
    violations = signed < 0
    return {
        "pair_preservation": float(preserved.mean()),
        "weighted_pair_preservation": float(weights[valid][preserved].sum() / weights[valid].sum()),
        "violation_fraction": float(violations.mean()),
        "weighted_violation_fraction": float(weights[valid][violations].sum() / weights[valid].sum()),
        "tied_pair_fraction": float(1 - valid.mean()),
    }


def summarize(name: str, labels: np.ndarray, logits: np.ndarray, teacher_risk: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    probs = probabilities(logits)
    decision = bayes_decisions(probs)
    l1 = decision["l1_bayes_decision"]
    error = np.abs(labels - l1)
    severe = error >= 2
    ordered = np.argsort(decision["l1_bayes_risk"])
    selective = [error[ordered[:max(1, int(np.ceil(frac * len(labels))))]].mean() for frac in np.arange(1, .09, -.05)]
    decisions = []
    for rule, key in (("mode", "mode_decision"), ("l1", "l1_bayes_decision"), ("l2", "l2_bayes_decision")):
        d = decision[key]
        e = np.abs(labels - d)
        decisions.append({"rule": rule, "accuracy": float((d == labels).mean()), "mae": float(e.mean()),
                          "qwk": float(cohen_kappa_score(labels, d, weights="quadratic")), "severe": float((e >= 2).mean())})
    margins = []
    for target, left, right in ((4, 4, 3), (4, 4, 2), (0, 0, 1)):
        value = logits[labels == target, left] - logits[labels == target, right]
        margins.append({"target": target, "margin": f"z{left}-z{right}", "mean": float(value.mean()),
                        "median": float(np.median(value)), "q10": float(np.quantile(value, .1)),
                        "q90": float(np.quantile(value, .9)), "positive": float((value > 0).mean())})
    return probs, l1, {"condition": name, "decisions": decisions,
        "probability": prediction_metrics(labels, probs) | {"ece": expected_calibration_error(labels, probs)[0]},
        "risk": {"spearman": float(spearmanr(decision["l1_bayes_risk"], error).statistic),
                 "auroc": float(roc_auc_score(severe, decision["l1_bayes_risk"])),
                 "auprc": float(average_precision_score(severe, decision["l1_bayes_risk"])),
                 "selective_mae": float(np.mean(selective))},
        "pair_order": pair_diagnostics(teacher_risk, decision["l1_bayes_risk"]),
        "c4": endpoint(labels, probs, l1, 4), "c0": endpoint(labels, probs, l1, 0), "margins": margins}


def loaded_logits(path: Path) -> np.ndarray:
    return np.asarray([json.loads(row["logits"]) for row in csv.DictReader(path.open())], dtype=np.float32)


def effective_weight(state: dict) -> np.ndarray:
    direction = state["direction"].float()
    return (torch.nn.functional.normalize(direction, dim=1) * state["fixed_norms"].float()[:, None]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if OUT.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=args.resume)
    (OUT / "fold_states").mkdir(exist_ok=True)
    with np.load(BASE / "frozen_features/train_rps_features.npz") as archive:
        ids, labels, features = archive["sample_id"], archive["labels"], archive["features"]
        teacher_probs, teacher_risk = archive["original_probabilities"], archive["original_l1_bayes_risk"]
        original_logits = archive["original_logits"]
    folds = np.asarray([int(row["fold"]) for row in csv.DictReader((BASE / "fold_assignments/assignments.csv").open())])
    checkpoint = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    original_weight, original_bias = state["fc.weight"].float(), state["fc.bias"].float()
    logits = np.empty((len(labels), 5), dtype=np.float32)
    histories, parameters, activities = [], [], []
    init_direction_error, max_norm_error = 0.0, 0.0
    for fold in range(5):
        fitting, held = folds != fold, folds == fold
        random.seed(fold); np.random.seed(fold); torch.manual_seed(fold)
        b_state = torch.load(BASE / "condition_b_balanced_head/fold_checkpoints" / f"fold_{fold}.pt", map_location="cpu", weights_only=False)["head_state_dict"]
        b_weight = b_state["weight"].float()
        target_norms = .5 * original_weight.norm(dim=1) + .5 * b_weight.norm(dim=1)
        head = DirectionOnlyLinear(original_weight, original_bias, target_norms)
        init_direction_error = max(init_direction_error, float((torch.nn.functional.normalize(head.direction, dim=1) - torch.nn.functional.normalize(original_weight, dim=1)).abs().max().detach()))
        state_path = OUT / "fold_states" / f"fold_{fold}.pt"
        if args.resume and state_path.exists():
            saved = torch.load(state_path, map_location="cpu", weights_only=False)
            head.load_state_dict(saved["head_state_dict"])
            with torch.no_grad():
                logits[held] = head(torch.tensor(features[held])).numpy()
                weight = head.effective_weight().numpy()
            d_state = torch.load(D_OUT / "fold_states/alpha_0p5" / f"fold_{fold}.pt", map_location="cpu", weights_only=False)["head_state_dict"]
            d_weight = effective_weight(d_state)
            for cls in range(5):
                cosine = lambda left, right: float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
                parameters.append({"fold": fold, "class": cls, "target_norm": float(target_norms[cls]), "final_norm": float(np.linalg.norm(weight[cls])), "norm_error": float(abs(np.linalg.norm(weight[cls]) - target_norms[cls])), "cosine_a": cosine(weight[cls], original_weight[cls].numpy()), "cosine_d": cosine(weight[cls], d_weight[cls]), "cosine_b": cosine(weight[cls], b_weight[cls].numpy())})
            continue
        optimizer = torch.optim.AdamW([head.direction], lr=.001, weight_decay=0.0)
        x, y, teacher = torch.tensor(features[fitting]), torch.tensor(labels[fitting]), torch.tensor(teacher_probs[fitting])
        balanced_generator = torch.Generator().manual_seed(10000 + fold)
        natural_generator = torch.Generator().manual_seed(20000 + fold)
        for epoch in range(1, EPOCHS + 1):
            ce_values, rop_values, violation_values, weighted_values, tied_values = [], [], [], [], []
            for natural_index in natural_batch_indices(len(y), BATCH_SIZE, natural_generator):
                balanced_index = balanced_batch_indices(y, len(natural_index), balanced_generator)
                ce = torch.nn.functional.cross_entropy(head(x[balanced_index]), y[balanced_index])
                student_probs = torch.softmax(head(x[natural_index]), dim=1)
                student_risk, action = detached_l1_bayes_risk(student_probs)
                if action.requires_grad:
                    raise RuntimeError("detached L1 action unexpectedly requires gradients")
                teacher_batch_risk, _ = detached_l1_bayes_risk(teacher[natural_index])
                rop, diagnostic = risk_order_preservation_loss(student_risk, teacher_batch_risk.detach())
                loss = ce + LAMBDA * rop
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                ce_values.append(float(ce.detach())); rop_values.append(float(rop.detach()))
                violation_values.append(float(diagnostic["violation_fraction"])); weighted_values.append(float(diagnostic["weighted_violation_fraction"])); tied_values.append(float(diagnostic["tied_pair_fraction"]))
            error = float(head.max_norm_error().detach()); max_norm_error = max(max_norm_error, error)
            if error > 1e-6: raise RuntimeError(f"fixed-norm contract failed: {error}")
            histories.append({"fold": fold, "epoch": epoch, "ce": float(np.mean(ce_values)), "rop": float(np.mean(rop_values)), "violation_fraction": float(np.mean(violation_values)), "weighted_violation_fraction": float(np.mean(weighted_values)), "tied_pair_fraction": float(np.mean(tied_values)), "norm_error": error})
        with torch.no_grad():
            logits[held] = head(torch.tensor(features[held])).numpy()
            weight = head.effective_weight().numpy()
        torch.save({"head_state_dict": head.state_dict(), "fold": fold, "alpha": ALPHA, "lambda": LAMBDA,
                    "target_norms": target_norms.tolist(), "max_norm_error": float(head.max_norm_error())}, state_path)
        d_state = torch.load(D_OUT / "fold_states/alpha_0p5" / f"fold_{fold}.pt", map_location="cpu", weights_only=False)["head_state_dict"]
        d_weight = effective_weight(d_state)
        for cls in range(5):
            cosine = lambda left, right: float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
            parameters.append({"fold": fold, "class": cls, "target_norm": float(target_norms[cls]), "final_norm": float(np.linalg.norm(weight[cls])), "norm_error": float(abs(np.linalg.norm(weight[cls]) - target_norms[cls])), "cosine_a": cosine(weight[cls], original_weight[cls].numpy()), "cosine_d": cosine(weight[cls], d_weight[cls]), "cosine_b": cosine(weight[cls], b_weight[cls].numpy())})
        fold_rows = [row for row in histories if row["fold"] == fold]
        activities.append({"fold": fold, "mean_rop": float(np.mean([row["rop"] for row in fold_rows])),
                           "mean_violation_fraction": float(np.mean([row["violation_fraction"] for row in fold_rows])),
                           "mean_weighted_violation_fraction": float(np.mean([row["weighted_violation_fraction"] for row in fold_rows])),
                           "mean_tied_pair_fraction": float(np.mean([row["tied_pair_fraction"] for row in fold_rows]))})
    e_probs, e_l1, result_e = summarize("E_controlled_scale_rop_alpha_0.5_lambda_1.0", labels, logits, teacher_risk)
    d_logits = loaded_logits(D_OUT / "oof_predictions/alpha_0p5.csv")
    _, d_l1, result_d = summarize("D_controlled_scale_alpha_0.5", labels, d_logits, teacher_risk)
    b_logits = loaded_logits(BASE / "oof_predictions/B_balanced_head.csv")
    _, _, result_b = summarize("B_balanced_head", labels, b_logits, teacher_risk)
    _, _, result_a = summarize("A_original_rps", labels, original_logits, teacher_risk)
    a_l1 = np.asarray([int(row["l1_decision"]) for row in csv.DictReader((BASE / "oof_predictions/A_original_rps.csv").open())])
    b_l1 = np.asarray([int(row["l1_decision"]) for row in csv.DictReader((BASE / "oof_predictions/B_balanced_head.csv").open())])
    c4, c0 = labels == 4, labels == 0
    exact_b = c4 & (a_l1 != 4) & (b_l1 == 4)
    inward_b = c4 & (np.abs(4 - b_l1) < np.abs(4 - a_l1))
    damage_b = c0 & (np.abs(b_l1) > np.abs(a_l1))
    subsets = {"b_exact_total": int(exact_b.sum()), "exact_retained_e": int((exact_b & (e_l1 == 4)).sum()), "b_inward_total": int(inward_b.sum()), "inward_retained_e": int((inward_b & (np.abs(4 - e_l1) < np.abs(4 - a_l1))).sum()), "b_c0_damaged": int(damage_b.sum()), "c0_restored_e": int((damage_b & (np.abs(e_l1) < np.abs(b_l1))).sum())}
    # If resumed after a completed fold state, reconstruct the final-state ROP
    # activity on the same natural fitting batches for every fold.
    if not activities:
        for fold in range(5):
            fitting = folds != fold
            saved = torch.load(OUT / "fold_states" / f"fold_{fold}.pt", map_location="cpu", weights_only=False)
            head = DirectionOnlyLinear(original_weight, original_bias, torch.tensor(saved["target_norms"]))
            head.load_state_dict(saved["head_state_dict"])
            x, teacher = torch.tensor(features[fitting]), torch.tensor(teacher_probs[fitting])
            generator = torch.Generator().manual_seed(20000 + fold)
            values = []
            with torch.no_grad():
                for index in natural_batch_indices(len(x), BATCH_SIZE, generator):
                    student_risk, _ = detached_l1_bayes_risk(torch.softmax(head(x[index]), dim=1))
                    teacher_batch_risk, _ = detached_l1_bayes_risk(teacher[index])
                    loss, diagnostic = risk_order_preservation_loss(student_risk, teacher_batch_risk)
                    values.append((float(loss), float(diagnostic["violation_fraction"]), float(diagnostic["weighted_violation_fraction"]), float(diagnostic["tied_pair_fraction"])))
            activities.append({"fold": fold, "mean_rop": float(np.mean([v[0] for v in values])), "mean_violation_fraction": float(np.mean([v[1] for v in values])), "mean_weighted_violation_fraction": float(np.mean([v[2] for v in values])), "mean_tied_pair_fraction": float(np.mean([v[3] for v in values]),)})
    rows = []
    decisions = bayes_decisions(e_probs)
    for index in range(len(labels)):
        rows.append({"sample_id": int(ids[index]), "fold": int(folds[index]), "label": int(labels[index]), "logits": json.dumps(logits[index].tolist()), "probabilities": json.dumps(e_probs[index].tolist()), "mode": int(decisions["mode_decision"][index]), "l1": int(e_l1[index]), "l2": int(decisions["l2_bayes_decision"][index]), "risk": float(decisions["l1_bayes_risk"][index]), "reference_teacher_risk": float(teacher_risk[index]), "condition": "E", "alpha": ALPHA, "lambda": LAMBDA})
    write_csv(OUT / "oof_predictions/E_controlled_scale_rop.csv", rows)
    write_csv(OUT / "parameters/per_fold.csv", parameters); write_csv(OUT / "summary/training.csv", histories); write_csv(OUT / "summary/rop_activity.csv", activities); write_csv(OUT / "margins/E.csv", result_e["margins"]); write_csv(OUT / "recovery_subsets/retention.csv", [subsets])
    payload = {"protocol": {"alpha": ALPHA, "lambda": LAMBDA, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "AdamW", "lr": .001, "weight_decay": 0.0, "initial_direction_error": init_direction_error, "max_norm_error": max_norm_error}, "A": result_a, "B": result_b, "D_alpha_0p5": result_d, "E": result_e, "subsets": subsets}
    (OUT / "summary/results.json").write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
