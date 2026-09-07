#!/usr/bin/env python3
"""Phase 3.11 one-shot validation gate; fitting and validation are separate commands."""
from __future__ import annotations

import argparse, csv, json, random, sys
from pathlib import Path

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

FEATURES = Path("outputs/retinamnist/native28/phase3_3_representation_audit_replay_verified/rps/seed_0/features.npz")
CHECKPOINT = Path("outputs/retinamnist/native28/phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt")
OUT = Path("outputs/retinamnist/phase3_11_frozen_candidate_validation")
EPOCHS, BATCH, ALPHA, LAMBDA, SEED = 100, 64, .5, 1.0, 0


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: raise ValueError(f"empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def original_head() -> tuple[torch.Tensor, torch.Tensor]:
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = saved.get("model_state_dict", saved.get("state_dict"))
    w, b = state["fc.weight"].float(), state["fc.bias"].float()
    if w.shape != (5, 512) or b.shape != (5,): raise ValueError("unexpected canonical head shape")
    return w, b


def training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Materialize only train-prefixed arrays; validation/test arrays stay unread."""
    with np.load(FEATURES) as archive:
        values = tuple(archive[k] for k in ("train_sample_id", "train_labels", "train_features", "train_probabilities"))
    ids, labels, features, probabilities = values
    if features.shape != (1080, 512) or np.bincount(labels, minlength=5).tolist() != [486, 128, 206, 194, 66]:
        raise ValueError("canonical train archive integrity failure")
    return ids, labels, features, probabilities


def fit() -> None:
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    torch.set_num_threads(1); OUT.mkdir(parents=True); (OUT / "heads").mkdir()
    ids, labels, features, teacher_probs = training_arrays()
    original_w, original_b = original_head()
    x, y, teacher = torch.tensor(features), torch.tensor(labels), torch.tensor(teacher_probs)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    balanced = torch.nn.Linear(512, 5); balanced.weight.data.copy_(original_w); balanced.bias.data.copy_(original_b)
    bopt = torch.optim.AdamW(balanced.parameters(), lr=.001, weight_decay=1e-4)
    bgen = torch.Generator().manual_seed(10_000)
    history = {"B": [], "D": [], "E": []}
    for epoch in range(1, EPOCHS + 1):
        losses = []
        for _ in range(int(np.ceil(len(y) / BATCH))):
            index = balanced_batch_indices(y, BATCH, bgen)
            loss = torch.nn.functional.cross_entropy(balanced(x[index]), y[index])
            bopt.zero_grad(set_to_none=True); loss.backward(); bopt.step(); losses.append(float(loss.detach()))
        history["B"].append({"epoch": epoch, "balanced_ce": float(np.mean(losses))})
    b_state = {k: v.detach().clone() for k, v in balanced.state_dict().items()}
    torch.save({"head_state_dict": b_state, "condition": "B", "training_samples": len(labels)}, OUT / "heads/B_balanced.pt")
    target = .5 * original_w.norm(dim=1) + .5 * b_state["weight"].norm(dim=1)
    for condition, lambda_ in (("D", 0.0), ("E", LAMBDA)):
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        head = DirectionOnlyLinear(original_w, original_b, target)
        optimizer = torch.optim.AdamW([head.direction], lr=.001, weight_decay=0.0)
        balanced_generator = torch.Generator().manual_seed(10_000)
        natural_generator = torch.Generator().manual_seed(20_000)
        for epoch in range(1, EPOCHS + 1):
            ce_values, rop_values, violation_values = [], [], []
            batches = natural_batch_indices(len(y), BATCH, natural_generator) if lambda_ else [None] * int(np.ceil(len(y) / BATCH))
            for natural_index in batches:
                batch_size = len(natural_index) if natural_index is not None else BATCH
                balanced_index = balanced_batch_indices(y, batch_size, balanced_generator)
                ce = torch.nn.functional.cross_entropy(head(x[balanced_index]), y[balanced_index])
                if lambda_:
                    risk, action = detached_l1_bayes_risk(torch.softmax(head(x[natural_index]), dim=1))
                    if action.requires_grad: raise RuntimeError("L1 action is not detached")
                    teacher_risk, _ = detached_l1_bayes_risk(teacher[natural_index])
                    rop, diag = risk_order_preservation_loss(risk, teacher_risk.detach())
                else:
                    rop, diag = ce.detach() * 0, {"violation_fraction": torch.zeros(())}
                optimizer.zero_grad(set_to_none=True); (ce + lambda_ * rop).backward(); optimizer.step()
                ce_values.append(float(ce.detach())); rop_values.append(float(rop.detach())); violation_values.append(float(diag["violation_fraction"]))
            error = float(head.max_norm_error().detach())
            if error > 1e-6: raise RuntimeError(f"fixed norm failure: {error}")
            history[condition].append({"epoch": epoch, "balanced_ce": float(np.mean(ce_values)), "rop": float(np.mean(rop_values)), "violation_fraction": float(np.mean(violation_values)), "norm_error": error})
        state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        torch.save({"head_state_dict": state, "condition": condition, "alpha": ALPHA, "lambda": lambda_, "target_norms": target.tolist(), "max_norm_error": float(head.max_norm_error().detach()), "training_samples": len(labels)}, OUT / f"heads/{condition}.pt")
    for condition, rows in history.items(): write(OUT / f"training/{condition}.csv", rows)
    (OUT / "fitting_complete.json").write_text(json.dumps({"phase": "3.11", "validation_loaded": False, "test_loaded": False, "fit_complete_before_validation": True, "train_samples": len(labels), "train_counts": np.bincount(labels, minlength=5).tolist(), "features": str(FEATURES), "checkpoint": str(CHECKPOINT), "alpha": ALPHA, "lambda": LAMBDA, "epochs": EPOCHS, "all_required_states": ["heads/B_balanced.pt", "heads/D.pt", "heads/E.pt"]}, indent=2) + "\n")


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.exp(logits - logits.max(1, keepdims=True)); return values / values.sum(1, keepdims=True)


def pair(teacher: np.ndarray, student: np.ndarray) -> dict:
    i, j = np.triu_indices(len(teacher), 1); delta = teacher[i] - teacher[j]; valid = delta != 0; weights = np.abs(delta[valid]); same = np.sign(delta[valid]) == np.sign(student[i][valid] - student[j][valid])
    return {"preserved": float(same.mean()), "weighted_preserved": float(weights[same].sum() / weights.sum())}


def endpoint(labels: np.ndarray, probs: np.ndarray, decision: np.ndarray, risk: np.ndarray, cls: int) -> dict:
    mask = labels == cls; error = np.abs(decision[mask] - cls); p = probs[mask]; mean = p @ np.arange(5)
    data = {"support": int(mask.sum()), "routing": [int((decision[mask] == k).sum()) for k in range(5)], "accuracy": float((decision[mask] == cls).mean()), "mae": float(error.mean()), "severe": float((error >= 2).mean()), "predictive_mean": float(mean.mean()), "mean_risk": float(risk[mask].mean()), "median_risk": float(np.median(risk[mask]))}
    if cls == 4: data |= {"p4": float(p[:, 4].mean()), "median_p4": float(np.median(p[:, 4])), "p3": float(p[:, 3].mean()), "median_p3": float(np.median(p[:, 3])), "p3p4": float(p[:, 3:].sum(1).mean()), "median_p3p4": float(np.median(p[:, 3:].sum(1))), "shrinkage": float(4 - mean.mean())}
    else: data |= {"p0": float(p[:, 0].mean()), "p1": float(p[:, 1].mean()), "p0p1": float(p[:, :2].sum(1).mean()), "inward_displacement": float(mean.mean())}
    return data


def report(name: str, ids: np.ndarray, labels: np.ndarray, logits: np.ndarray, teacher_risk: np.ndarray) -> tuple[list[dict], dict]:
    probs = softmax(logits); d = bayes_decisions(probs); l1, risk = d["l1_bayes_decision"], d["l1_bayes_risk"]; error = np.abs(labels - l1); severe = error >= 2
    global_ = {}
    for key in ("mode_decision", "l1_bayes_decision", "l2_bayes_decision"):
        decision, e = d[key], np.abs(labels - d[key]); global_[key] = {"accuracy": float((decision == labels).mean()), "mae": float(e.mean()), "qwk": float(cohen_kappa_score(labels, decision, weights="quadratic")), "severe": float((e >= 2).mean())}
    order = np.argsort(risk); coverage = [error[order[:max(1, int(np.ceil(c * len(labels))))]].mean() for c in np.arange(1, .09, -.05)]
    result = {"condition": name, "global": global_, "probability": prediction_metrics(labels, probs) | {"ece": expected_calibration_error(labels, probs)[0]}, "risk": {"spearman": float(spearmanr(risk, error).statistic), "auroc": float(roc_auc_score(severe, risk)), "auprc": float(average_precision_score(severe, risk)), "selective_mae": float(np.mean(coverage))}, "pair_order": pair(teacher_risk, risk), "class_0": endpoint(labels, probs, l1, risk, 0), "class_4": endpoint(labels, probs, l1, risk, 4)}
    rows = [{"sample_id": int(ids[i]), "label": int(labels[i]), "logits": json.dumps(logits[i].tolist()), "probabilities": json.dumps(probs[i].tolist()), "mode": int(d["mode_decision"][i]), "l1": int(l1[i]), "l2": int(d["l2_bayes_decision"][i]), "l1_bayes_risk": float(risk[i]), "reference_teacher_risk": float(teacher_risk[i]), "condition": name} for i in range(len(labels))]
    return rows, result


def evaluate() -> None:
    required = [OUT / "fitting_complete.json", *(OUT / f"heads/{name}.pt" for name in ("B_balanced", "D", "E"))]
    if not all(path.is_file() for path in required): raise RuntimeError("all full-training head states must exist before validation is loaded")
    # This is the first command that materializes validation arrays. Test arrays
    # in the shared archive are intentionally never indexed or materialized.
    with np.load(FEATURES) as archive:
        ids, labels, features, original_logits, teacher_probs = (archive[k] for k in ("val_sample_id", "val_labels", "val_features", "val_logits", "val_probabilities"))
    if features.shape != (120, 512) or np.bincount(labels, minlength=5).tolist() != [54, 12, 28, 20, 6]: raise ValueError("validation archive integrity failure")
    teacher_risk = bayes_decisions(teacher_probs)["l1_bayes_risk"]
    original_w, original_b = original_head(); conditions = {"A": original_logits}
    for name in ("B_balanced", "D", "E"):
        saved = torch.load(OUT / f"heads/{name}.pt", map_location="cpu", weights_only=False)["head_state_dict"]
        if name == "B_balanced":
            head = torch.nn.Linear(512, 5); head.load_state_dict(saved)
        else:
            head = DirectionOnlyLinear(original_w, original_b, saved["fixed_norms"]); head.load_state_dict(saved)
            if float(head.max_norm_error().detach()) > 1e-6: raise RuntimeError("saved norm contract failed")
        with torch.no_grad(): conditions[name] = head(torch.tensor(features)).numpy()
    results = {"integrity": {"validation_loaded_after_fitting_complete": True, "test_loaded": False, "validation_samples": 120, "validation_counts": np.bincount(labels, minlength=5).tolist(), "alpha": ALPHA, "lambda": LAMBDA}}
    for name, logits in conditions.items():
        rows, result = report(name, ids, labels, logits, teacher_risk); write(OUT / f"validation/predictions_{name}.csv", rows); results[name] = result
    (OUT / "validation/results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("fit", "evaluate")); args = parser.parse_args()
    fit() if args.command == "fit" else evaluate()
