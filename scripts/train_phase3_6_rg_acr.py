#!/usr/bin/env python3
"""Phase 3.6 seed-0 RG-ACR falsification; fixed λ grid and matched RPS reference."""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.evaluation.frozen_head import class_priors, logit_adjusted_ce_loss
from ordinal_uncertainty.evaluation.representation import (
    class_centroids, cosine_distances, euclidean_distances, l2_normalize,
    nearest_centroid, within_class_dispersion,
)
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics
from ordinal_uncertainty.models.ordinal import rg_acr_loss, rps_loss
from ordinal_uncertainty.models.resnet import make_resnet18
from ordinal_uncertainty.utils.reproducibility import set_seed


LAMBDA_GRID = (0.05, 0.10, 0.20)
MARGIN = 0.05
RISK_CAP = 2.0
EPSILON = 1e-8


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def softmax(logits: np.ndarray) -> np.ndarray:
    value = logits - logits.max(axis=1, keepdims=True)
    value = np.exp(value)
    return value / value.sum(axis=1, keepdims=True)


class FeatureRecorder:
    def __init__(self, model: torch.nn.Module):
        self.features: torch.Tensor | None = None
        self.handle = model.fc.register_forward_pre_hook(self._record)

    def _record(self, _module: torch.nn.Module, values: tuple[torch.Tensor, ...]) -> None:
        self.features = values[0]

    def close(self) -> None:
        self.handle.remove()


def aggregate_participation(
    total: dict[str, object], labels: torch.Tensor, diagnostics: dict[str, torch.Tensor], contribution: torch.Tensor
) -> None:
    valid = diagnostics["valid_mask"].detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    counts = total["per_class"]
    for klass in range(5):
        observed = labels_np == klass
        count = int(observed.sum())
        valid_count = int((valid & observed).sum())
        counts[str(klass)]["samples"] += count
        counts[str(klass)]["valid"] += valid_count
        counts[str(klass)]["contribution"] += float(contribution.detach()) * valid_count
    total["batches"] += 1
    total["zero_batches"] += int(not valid.any())
    total["class4_valid_per_batch"].append(int((valid & (labels_np == 4)).sum()))
    total["class4_batch_present"].append(int((labels_np == 4).any()))
    total["class4_adjacent_available"].append(int(((labels_np == 4) & valid).sum()))
    total["valid_total"] += int(valid.sum())
    total["sample_total"] += int(labels_np.size)


def participation_summary(total: dict[str, object]) -> dict:
    per_class = total["per_class"]
    class4 = per_class["4"]
    values = np.asarray(total["class4_valid_per_batch"], dtype=float)
    batches = max(1, int(total["batches"]))
    return {
        "batches": int(total["batches"]),
        "zero_representation_loss_batch_rate": total["zero_batches"] / batches,
        "overall_valid_anchor_rate": total["valid_total"] / max(1, total["sample_total"]),
        "per_class": {
            klass: {
                "samples_observed": row["samples"], "valid_anchors": row["valid"],
                "valid_anchor_fraction": row["valid"] / max(1, row["samples"]),
                "total_rg_acr_contribution_proxy": row["contribution"],
            } for klass, row in per_class.items()
        },
        "class4_valid_anchor_rate": class4["valid"] / max(1, class4["samples"]),
        "class4_mean_valid_anchors_per_batch": float(values.mean()),
        "class4_median_valid_anchors_per_batch": float(np.median(values)),
        "class4_batch_present_fraction": float(np.mean(total["class4_batch_present"])),
        "class4_valid_anchor_batch_fraction": float(np.mean(values > 0)),
        "class4_adjacent_class3_available_for_anchor_fraction": float(
            sum(total["class4_adjacent_available"]) / max(1, class4["samples"])
        ),
    }


def new_participation() -> dict[str, object]:
    return {
        "batches": 0, "zero_batches": 0, "valid_total": 0, "sample_total": 0,
        "per_class": {str(k): {"samples": 0, "valid": 0, "contribution": 0.0} for k in range(5)},
        "class4_valid_per_batch": [], "class4_batch_present": [], "class4_adjacent_available": [],
    }


def collect(model: torch.nn.Module, loader, device: torch.device, recorder: FeatureRecorder) -> dict[str, np.ndarray]:
    model.eval()
    logits, features, labels = [], [], []
    with torch.no_grad():
        for images, target in loader:
            values = model(images.to(device))
            assert recorder.features is not None
            logits.append(values.cpu().numpy())
            features.append(recorder.features.cpu().numpy())
            labels.append(target.reshape(-1).numpy())
    return {
        "logits": np.concatenate(logits).astype(np.float64),
        "features": np.concatenate(features).astype(np.float64),
        "labels": np.concatenate(labels).astype(np.int64),
    }


def decision_rows(labels: np.ndarray, probabilities: np.ndarray) -> tuple[list[dict], dict[str, np.ndarray]]:
    decisions = bayes_decisions(probabilities)
    rows = []
    for name, key in (("mode", "mode_decision"), ("l1", "l1_bayes_decision"), ("l2", "l2_bayes_decision")):
        prediction = decisions[key]
        error = np.abs(labels - prediction)
        rows.append({
            "decision": name, "accuracy": float((prediction == labels).mean()),
            "mae": float(error.mean()), "qwk": float(cohen_kappa_score(labels, prediction, weights="quadratic")),
            "severe_prevalence": float((error >= 2).mean()), "severe_count": int((error >= 2).sum()),
        })
    return rows, decisions


def risk_rows(labels: np.ndarray, decisions: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    prediction = decisions["l1_bayes_decision"]
    risk = decisions["l1_bayes_risk"]
    error = np.abs(labels - prediction)
    severe = error >= 2
    rows = [{
        "risk_error_spearman": float(spearmanr(risk, error).statistic),
        "severe_auroc": float(roc_auc_score(severe, risk)),
        "severe_auprc": float(average_precision_score(severe, risk)),
    }]
    curve = []
    order = np.argsort(risk, kind="stable")
    for coverage in np.round(np.arange(1.0, .099, -.05), 2):
        retained = order[:max(1, int(np.ceil(coverage * len(order))))]
        curve.append({
            "coverage": float(coverage), "retained_count": int(len(retained)),
            "ordinal_mae": float(error[retained].mean()),
        })
    rows[0]["mean_selective_mae"] = float(np.mean([row["ordinal_mae"] for row in curve]))
    return rows, curve


def endpoint_rows(labels: np.ndarray, probabilities: np.ndarray, decisions: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    classes = np.arange(probabilities.shape[1])
    predictive_mean = probabilities @ classes
    shrinkage = inward_shrinkage(labels, probabilities)
    rows, routing = [], []
    for true_class in (0, 4):
        mask = labels == true_class
        l1 = decisions["l1_bayes_decision"][mask]
        error = np.abs(l1 - true_class)
        row = {
            "true_class": true_class, "count": int(mask.sum()), "l1_accuracy": float((l1 == true_class).mean()),
            "l1_mae": float(error.mean()), "l1_severe_prevalence": float((error >= 2).mean()),
            "mean_p_true": float(probabilities[mask, true_class].mean()),
            "median_p_true": float(np.median(probabilities[mask, true_class])),
            "predictive_mean": float(predictive_mean[mask].mean()),
            "inward_shrinkage": float(shrinkage[mask].mean()),
            "l1_bayes_risk": float(decisions["l1_bayes_risk"][mask].mean()),
        }
        if true_class == 0:
            row.update({"mean_p0": float(probabilities[mask, 0].mean()), "mean_p1": float(probabilities[mask, 1].mean()),
                        "mean_p0_plus_p1": float(probabilities[mask, :2].sum(axis=1).mean())})
        else:
            row.update({"mean_p4": float(probabilities[mask, 4].mean()), "median_p4": float(np.median(probabilities[mask, 4])),
                        "mean_p3": float(probabilities[mask, 3].mean()), "median_p3": float(np.median(probabilities[mask, 3])),
                        "mean_p3_plus_p4": float(probabilities[mask, 3:].sum(axis=1).mean()),
                        "median_p3_plus_p4": float(np.median(probabilities[mask, 3:].sum(axis=1)))})
        rows.append(row)
        for predicted in range(5):
            routing.append({
                "true_class": true_class, "predicted_class": predicted,
                "count": int((l1 == predicted).sum()), "fraction": float((l1 == predicted).mean()),
            })
    return rows, routing


def geometry(train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> tuple[dict, list[dict]]:
    rows, summary = [], {}
    for name, reference, values, distance_fn in (
        ("raw_euclidean", train["features"], test["features"], euclidean_distances),
        ("l2_normalized_cosine", l2_normalize(train["features"]), l2_normalize(test["features"]), cosine_distances),
    ):
        centroids = class_centroids(reference, train["labels"], 5)
        distances = distance_fn(values, centroids)
        centroid_distances = distance_fn(centroids, centroids)
        nearest = nearest_centroid(distances)
        d4 = distances[test["labels"] == 4]
        d0 = distances[test["labels"] == 0]
        margins = {"delta_4_3": d4[:, 3] - d4[:, 4], "delta_4_2": d4[:, 2] - d4[:, 4], "delta_0_1": d0[:, 1] - d0[:, 0]}
        dispersion = within_class_dispersion(values, test["labels"], centroids)
        entry = {
            "class4_routing": [int(((test["labels"] == 4) & (nearest == k)).sum()) for k in range(5)],
            "class0_nearest_accuracy": float(nearest[test["labels"] == 0].mean() == 0) if False else float((nearest[test["labels"] == 0] == 0).mean()),
            "class3_class4_centroid_separation": float(centroid_distances[3, 4]),
            "class2_class4_centroid_separation": float(centroid_distances[2, 4]),
            "class0_class1_centroid_separation": float(centroid_distances[0, 1]),
            "class4_dispersion_mean": float(dispersion["mean"][4]),
            "class0_dispersion_mean": float(dispersion["mean"][0]),
            "class4_ordinal_centroid_ordering": bool(
                centroid_distances[4,3] < centroid_distances[4,2] < centroid_distances[4,1] < centroid_distances[4,0]
            ),
        }
        for metric, array in margins.items():
            entry[metric] = {
                "mean": float(array.mean()), "median": float(np.median(array)),
                "fraction_positive": float((array > 0).mean()), "fraction_negative": float((array < 0).mean()),
            }
        summary[name] = entry
        for source in range(5):
            for target in range(5):
                rows.append({"space": name, "source_class": source, "target_class": target, "centroid_distance": float(centroid_distances[source, target])})
    return summary, rows


def evaluate(condition: str, output: Path, train: dict[str, np.ndarray], test: dict[str, np.ndarray]) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    probabilities = softmax(test["logits"])
    decisions_table, decisions = decision_rows(test["labels"], probabilities)
    risk_table, coverage = risk_rows(test["labels"], decisions)
    endpoint, routing = endpoint_rows(test["labels"], probabilities, decisions)
    ece, reliability = expected_calibration_error(test["labels"], probabilities)
    predictive = prediction_metrics(test["labels"], probabilities)
    predictive["ece"] = float(ece)
    predictive["mode_severe_prevalence"] = decisions_table[0]["severe_prevalence"]
    geometry_summary, centroid_rows = geometry(train, test)
    prediction_rows = []
    for i in range(len(test["labels"])):
        prediction_rows.append({
            "sample_id": i, "true_label": int(test["labels"][i]),
            "logits": json.dumps(test["logits"][i].tolist()), "probabilities": json.dumps(probabilities[i].tolist()),
            "mode_decision": int(decisions["mode_decision"][i]), "l1_bayes_decision": int(decisions["l1_bayes_decision"][i]),
            "l2_bayes_decision": int(decisions["l2_bayes_decision"][i]),
            "mode_l1_risk": float(decisions["mode_l1_risk"][i]), "l1_bayes_risk": float(decisions["l1_bayes_risk"][i]),
        })
    write_csv(output / "predictions.csv", prediction_rows)
    write_csv(output / "decision_metrics.csv", decisions_table)
    write_csv(output / "risk_metrics.csv", risk_table)
    write_csv(output / "risk_coverage.csv", coverage)
    write_csv(output / "endpoint_metrics.csv", endpoint)
    write_csv(output / "l1_routing.csv", routing)
    write_csv(output / "reliability_bins.csv", reliability)
    write_csv(output / "centroid_distances.csv", centroid_rows)
    np.savez_compressed(output / "features.npz", train_features=train["features"], train_labels=train["labels"],
                        test_features=test["features"], test_labels=test["labels"], test_logits=test["logits"])
    result = {"condition": condition, "predictive": predictive, "decisions": decisions_table, "risk": risk_table[0],
              "endpoints": endpoint, "geometry": geometry_summary}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def validation_rps(model: torch.nn.Module, loader, device: torch.device) -> float:
    model.eval()
    values = []
    with torch.no_grad():
        for images, labels in loader:
            values.append(float(rps_loss(model(images.to(device)), labels.reshape(-1).long().to(device))))
    return float(np.mean(values))


def run_condition(
    output: Path, name: str, lambda_: float, loaders, metadata: dict, device: torch.device, epochs: int, batch_size: int, seed: int,
    smoke: bool = False,
) -> tuple[dict, dict]:
    output.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    model = make_resnet18(metadata["num_classes"]).to(device)
    recorder = FeatureRecorder(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_rps, best_epoch, best_state = float("inf"), 0, None
    history, total_participation = [], new_participation()
    smoke_checks: dict[str, bool] = {}
    for epoch in range(1, epochs + 1):
        model.train()
        running_rps, running_repr, running_total, count = 0.0, 0.0, 0.0, 0
        epoch_participation = new_participation()
        for images, labels in loaders["train"]:
            labels = labels.reshape(-1).long().to(device)
            logits = model(images.to(device))
            assert recorder.features is not None
            if lambda_:
                repr_loss, diagnostic = rg_acr_loss(logits, recorder.features, labels, margin=MARGIN, risk_cap=RISK_CAP, epsilon=EPSILON)
            else:
                repr_loss, diagnostic = recorder.features.sum() * 0.0, {
                    "valid_mask": torch.zeros(labels.shape[0], dtype=torch.bool, device=device),
                    "class_counts": torch.bincount(labels, minlength=5), "adjacent_terms": torch.zeros(5, dtype=torch.long, device=device),
                }
            base = rps_loss(logits, labels)
            loss = base + lambda_ * repr_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if smoke and not smoke_checks:
                smoke_checks = {
                    "rps_finite": bool(torch.isfinite(base)),
                    "rg_acr_finite": bool(torch.isfinite(repr_loss)),
                    "backbone_gradient_finite": bool(
                        model.conv1.weight.grad is not None and torch.isfinite(model.conv1.weight.grad).all()
                    ),
                    "feature_gradient_path_present": bool(recorder.features.requires_grad),
                }
            optimizer.step()
            if lambda_:
                aggregate_participation(epoch_participation, labels, diagnostic, lambda_ * repr_loss)
                aggregate_participation(total_participation, labels, diagnostic, lambda_ * repr_loss)
            n = labels.numel()
            running_rps += float(base.detach()) * n
            running_repr += float(repr_loss.detach()) * n
            running_total += float(loss.detach()) * n
            count += n
        value = validation_rps(model, loaders["val"], device)
        record = {
            "epoch": epoch, "training_rps": running_rps / count, "training_rg_acr": running_repr / count,
            "training_total": running_total / count, "validation_rps": value,
        }
        if lambda_:
            record.update(participation_summary(epoch_participation))
        history.append(record)
        if value < best_rps:
            best_rps, best_epoch = value, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if smoke:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    if smoke:
        torch.save({"state_dict": best_state, "method": name, "lambda": lambda_, "best_epoch": best_epoch},
                   output / "best_checkpoint.pt")
        write_csv(output / "training_history.csv", history)
        smoke_record = {
            "smoke": True, "best_epoch": best_epoch, "best_validation_rps": best_rps,
            "checks": smoke_checks, "participation": participation_summary(total_participation),
        }
        (output / "smoke.json").write_text(json.dumps(smoke_record, indent=2) + "\n", encoding="utf-8")
        recorder.close()
        return {"smoke": smoke_record}, {"best_epoch": best_epoch, "best_validation_rps": best_rps,
                                          "participation": smoke_record["participation"]}
    train = collect(model, loaders["train"], device, recorder)
    val = collect(model, loaders["val"], device, recorder)
    test = collect(model, loaders["test"], device, recorder)
    result = evaluate(name, output / "evaluation", train, test)
    torch.save({
        "state_dict": best_state, "method": name, "lambda": lambda_, "margin": MARGIN, "risk_cap": RISK_CAP,
        "best_epoch": best_epoch, "best_validation_rps": best_rps,
    }, output / "best_checkpoint.pt")
    write_csv(output / "training_history.csv", history)
    config = {
        "method": name, "seed": seed, "epochs": epochs, "batch_size": batch_size, "image_size": 28,
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-4,
        "checkpoint_selection": "minimum validation RPS", "best_epoch": best_epoch, "best_validation_rps": best_rps,
        "lambda": lambda_, "margin": MARGIN, "risk_cap": RISK_CAP, "epsilon": EPSILON,
        "feature": "penultimate model.fc input (512-D), L2 normalized",
        "distance": "1 - normalized feature dot normalized centroid",
        "risk": "L1 Bayes risk min_a sum_k p_k |k-a| from current softmax probabilities, detached",
        "risk_weight_formula": "min(2, stopgrad(R_i)/(mean_batch stopgrad(R)+1e-8)); mean over all B",
        "valid_anchor": "own batch count >=2 and at least one present adjacent class; use all present adjacent classes",
        "preprocessing": "retinamnist_loaders canonical native 28 ToTensor",
        "smoke": smoke,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    participation = participation_summary(total_participation) if lambda_ else {"not_applicable_reference_rps": True}
    (output / "participation.json").write_text(json.dumps(participation, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(output / "feature_artifacts.npz", train_features=train["features"], train_labels=train["labels"],
                        val_features=val["features"], val_labels=val["labels"], test_features=test["features"],
                        test_labels=test["labels"], test_logits=test["logits"])
    recorder.close()
    return result, {"best_epoch": best_epoch, "best_validation_rps": best_rps, "participation": participation}


def run_secondary_head(feature_path: Path, output: Path, device: torch.device) -> dict:
    with np.load(feature_path) as data:
        train_x, train_y = data["train_features"], data["train_labels"]
        val_x, val_y = data["val_features"], data["val_labels"]
        test_x, test_y = data["test_features"], data["test_labels"]
    set_seed(0)
    head = torch.nn.Linear(train_x.shape[1], 5).to(device)
    priors = torch.tensor(class_priors(train_y, 5), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(0)
    best, best_epoch, state = float("inf"), 0, None
    for epoch in range(1, 101):
        head.train()
        for indices in torch.randperm(len(train_y), generator=generator).split(64):
            logits = head(torch.as_tensor(train_x[indices], dtype=torch.float32, device=device))
            loss = logit_adjusted_ce_loss(logits, torch.as_tensor(train_y[indices], dtype=torch.long, device=device), priors)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        head.eval()
        with torch.no_grad():
            score = float(rps_loss(head(torch.as_tensor(val_x, dtype=torch.float32, device=device)), torch.as_tensor(val_y, dtype=torch.long, device=device)))
        if score < best:
            best, best_epoch = score, epoch
            state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    assert state is not None
    head.load_state_dict(state)
    with torch.no_grad():
        logits = head(torch.as_tensor(test_x, dtype=torch.float32, device=device)).cpu().numpy().astype(np.float64)
    train = {"features": train_x, "labels": train_y}
    test = {"features": test_x, "labels": test_y, "logits": logits}
    result = evaluate("selected_rg_acr/logit_adjusted_frozen_head", output, train, test)
    (output / "head_control.json").write_text(json.dumps({
        "objective": "CE(z + log(pi_train), y), tau=1; evaluate softmax(z)", "best_epoch": best_epoch,
        "best_validation_rps": best, "frozen_backbone": True,
    }, indent=2) + "\n", encoding="utf-8")
    return result | {"best_epoch": best_epoch, "best_validation_rps": best}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/retinamnist/native28/phase3_6_rg_acr"))
    parser.add_argument("--data-root", type=Path, default=Path("data/medmnist"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, choices=(0,))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite historical/output evidence: {args.output}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    loaders, metadata = retinamnist_loaders(args.data_root, args.batch_size, args.num_workers, False, 28)
    args.output.mkdir(parents=True)
    smoke_output = args.output / "smoke_lambda_0p10"
    _, smoke_meta = run_condition(smoke_output, "rg_acr_smoke", .10, loaders, metadata, device, 1, args.batch_size, args.seed, smoke=True)
    if args.smoke:
        print(json.dumps({"smoke": smoke_meta}, indent=2)); return
    reference, reference_meta = run_condition(args.output / "reference_rps_retrained", "rps_seed0_retrained", 0., loaders, metadata, device, args.epochs, args.batch_size, args.seed)
    all_results, all_meta = {"reference_rps_retrained": reference}, {"reference_rps_retrained": reference_meta}
    for lambda_ in LAMBDA_GRID:
        label = f"lambda_{lambda_:.2f}".replace(".", "p")
        result, metadata_run = run_condition(args.output / label, f"rg_acr_lambda_{lambda_:.2f}", lambda_, loaders, metadata, device, args.epochs, args.batch_size, args.seed)
        all_results[label], all_meta[label] = result, metadata_run
    selected = min((label for label in all_meta if label.startswith("lambda_")), key=lambda label: all_meta[label]["best_validation_rps"])
    secondary = run_secondary_head(args.output / selected / "feature_artifacts.npz", args.output / selected / "secondary_logit_adjusted_head", device)
    summary = args.output / "summary"; summary.mkdir()
    (summary / "phase3_6_results.json").write_text(json.dumps({
        "reference": "reference_rps_retrained", "selected_by_validation_rps": selected,
        "runs": all_results, "run_metadata": all_meta, "secondary_logit_adjusted_head": secondary,
        "seed": 0, "no_test_driven_selection": True,
    }, indent=2) + "\n", encoding="utf-8")
    write_csv(summary / "checkpoint_selection.csv", [
        {"condition": label, "best_epoch": data["best_epoch"], "best_validation_rps": data["best_validation_rps"]}
        for label, data in all_meta.items()
    ])
    print(json.dumps({"selected_by_validation_rps": selected, "metadata": all_meta}, indent=2))


if __name__ == "__main__":
    main()
