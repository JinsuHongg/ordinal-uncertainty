#!/usr/bin/env python3
"""Seed-0 Endpoint-Neighborhood RPS training and fixed evaluation artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.evaluation.probability_pipeline import finalize_probability_evaluation
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage
from ordinal_uncertainty.models.ordinal import (
    endpoint_neighborhood_rps_loss,
    endpoint_preference_rps_loss,
    endpoint_neighborhood_weights,
    rps_loss,
)
from ordinal_uncertainty.models.resnet import make_resnet18
from ordinal_uncertainty.utils.reproducibility import set_seed


COVERAGES = np.round(np.arange(1.0, 0.099, -0.05), 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean_loss(model, loader, device, endpoint_weights, lambda_, objective, rho, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0; count = 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            labels = labels.reshape(-1).long().to(device)
            logits = model(images.to(device))
            loss = (
                endpoint_neighborhood_rps_loss(logits, labels, endpoint_weights, lambda_)
                if objective == "endpoint_neighborhood"
                else endpoint_preference_rps_loss(logits, labels, endpoint_weights, lambda_, rho)
            )
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.item()) * len(labels); count += len(labels)
    return total / count


def validation_rps(model, loader, device) -> float:
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for images, labels in loader:
            labels = labels.reshape(-1).long().to(device)
            loss = rps_loss(model(images.to(device)), labels)
            total += float(loss.item()) * len(labels); count += len(labels)
    return total / count


def evaluate_extended(labels: np.ndarray, probabilities: np.ndarray, output: Path, method: str) -> None:
    decisions = bayes_decisions(probabilities); classes = np.arange(probabilities.shape[1])
    shrinkage = inward_shrinkage(labels, probabilities); predictive_mean = probabilities @ classes
    decision_rows = []; risk_rows = []; routing_rows = []; extreme_rows = []
    for rule, key, risk_key in (("mode", "mode_decision", "mode_l1_risk"), ("l1", "l1_bayes_decision", "l1_bayes_risk"), ("l2", "l2_bayes_decision", "l2_bayes_risk")):
        prediction = decisions[key]; error = np.abs(labels - prediction); severe = error >= 2; risk = decisions[risk_key]
        decision_rows.append({"method": method, "decision": rule, "accuracy": float((prediction == labels).mean()), "mae": float(error.mean()), "qwk": float(cohen_kappa_score(labels, prediction, weights="quadratic")), "severe_count": int(severe.sum()), "severe_prevalence": float(severe.mean())})
        order = np.argsort(risk, kind="stable")
        coverage_rows = [{"method": method, "decision": rule, "coverage": float(coverage), "retained_count": int(max(1, np.ceil(coverage * len(labels)))), "ordinal_risk_mae": float(error[order[:max(1, int(np.ceil(coverage * len(labels))))]].mean())} for coverage in COVERAGES]
        write_csv(output / f"{rule}_risk_coverage.csv", coverage_rows)
        risk_rows.append({"method": method, "decision": rule, "spearman": float(spearmanr(risk, error).statistic), "severe_auroc": float(roc_auc_score(severe, risk)), "severe_auprc": float(average_precision_score(severe, risk)), "mean_mae_selective_risk": float(np.mean([row["ordinal_risk_mae"] for row in coverage_rows])), "mode_to_l1_changed_fraction": float((decisions["mode_decision"] != decisions["l1_bayes_decision"]).mean()) if rule == "l1" else None})
        if rule == "l1":
            for true_class in (0, probabilities.shape[1] - 1):
                mask = labels == true_class
                for predicted_class in range(probabilities.shape[1]):
                    routing_rows.append({"method": method, "true_class": true_class, "predicted_class": predicted_class, "count": int(((prediction == predicted_class) & mask).sum()), "fraction": float((prediction[mask] == predicted_class).mean())})
    l1_prediction = decisions["l1_bayes_decision"]
    for true_class in (0, probabilities.shape[1] - 1):
        mask = labels == true_class; error = np.abs(labels[mask] - l1_prediction[mask]); near = probabilities[mask, :2].sum(1) if true_class == 0 else probabilities[mask, -2:].sum(1)
        true_probability = probabilities[mask, true_class]
        adjacent_probability = probabilities[mask, 1 if true_class == 0 else probabilities.shape[1] - 2]
        extreme_rows.append({"method": method, "true_class": true_class, "count": int(mask.sum()), "accuracy": float((l1_prediction[mask] == true_class).mean()), "mae": float(error.mean()), "severe_count": int((error >= 2).sum()), "severe_prevalence": float((error >= 2).mean()), "mean_p_true": float(true_probability.mean()), "median_p_true": float(np.median(true_probability)), "mean_p_adjacent": float(adjacent_probability.mean()), "median_p_adjacent": float(np.median(adjacent_probability)), "mean_near_mass": float(near.mean()), "median_near_mass": float(np.median(near)), "predictive_mean": float(predictive_mean[mask].mean()), "inward_shrinkage": float(shrinkage[mask].mean()), "l1_bayes_risk": float(decisions["l1_bayes_risk"][mask].mean())})
    write_csv(output / "decision_metrics.csv", decision_rows)
    write_csv(output / "risk_metrics.csv", risk_rows)
    write_csv(output / "l1_endpoint_routing.csv", routing_rows)
    write_csv(output / "extreme_class_metrics.csv", extreme_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", choices=("endpoint_neighborhood", "endpoint_preference"), default="endpoint_neighborhood")
    parser.add_argument("--lambda", dest="lambda_", type=float, required=True, choices=(0.1, 0.3, 1.0))
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=Path("data/medmnist"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs <= 0: raise ValueError("epochs must be positive")
    if args.objective == "endpoint_preference" and (args.rho != 0.5 or args.lambda_ not in (0.1, 0.3)):
        raise ValueError("endpoint_preference is predeclared only for rho=0.5 and lambda in {0.1, 0.3}")
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders, metadata = retinamnist_loaders(args.data_root, args.batch_size, args.num_workers, args.download, 28)
    counts = torch.tensor([Counter(loaders["train"].dataset.labels.reshape(-1).tolist())[i] for i in range(metadata["num_classes"])])
    endpoint_weights = endpoint_neighborhood_weights(counts).to(device)
    model = make_resnet18(metadata["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []; best_rps = float("inf"); best_epoch = 0; best_state = None
    for epoch in range(1, args.epochs + 1):
        train_loss = mean_loss(model, loaders["train"], device, endpoint_weights, args.lambda_, args.objective, args.rho, optimizer)
        val_objective = mean_loss(model, loaders["val"], device, endpoint_weights, args.lambda_, args.objective, args.rho)
        val_rps = validation_rps(model, loaders["val"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_endpoint_neighborhood_rps": val_objective, "validation_rps": val_rps})
        if val_rps < best_rps:
            best_rps = val_rps; best_epoch = epoch; best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state); model.eval(); all_logits = []; all_labels = []
    with torch.no_grad():
        for images, labels in loaders["test"]:
            all_logits.append(model(images.to(device)).cpu()); all_labels.append(labels.reshape(-1))
    logits = torch.cat(all_logits).numpy(); labels = torch.cat(all_labels).numpy()
    finalized = finalize_probability_evaluation(labels, logits, args.output / "evaluation")
    probabilities = finalized["probabilities"]
    np.savez_compressed(args.output / "test_arrays.npz", sample_ids=np.arange(len(labels)), labels=labels, logits=logits, probabilities=probabilities)
    evaluate_extended(labels, probabilities, args.output, f"{args.objective}_rps")
    torch.save({"state_dict": best_state, "method": args.objective, "lambda": args.lambda_, "rho": args.rho, "best_epoch": best_epoch, "best_validation_rps": best_rps}, args.output / "best_checkpoint.pt")
    write_csv(args.output / "training_history.csv", history)
    config = {"method": args.objective, "seed": args.seed, "lambda": args.lambda_, "rho": args.rho, "epochs": args.epochs, "image_size": 28, "batch_size": args.batch_size, "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4, "training_class_counts": counts.tolist(), "endpoint_weights": endpoint_weights.detach().cpu().tolist(), "checkpoint_selection": "minimum validation RPS", "device": str(device), "dataset": metadata}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps({"best_epoch": best_epoch, "best_validation_rps": best_rps, "lambda": args.lambda_, "rho": args.rho}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"objective": args.objective, "lambda": args.lambda_, "rho": args.rho, "best_epoch": best_epoch, "best_validation_rps": best_rps, **finalized["summary"]["prediction_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
