#!/usr/bin/env python3
"""Validation-only temperature-scaling audit for frozen Phase 2 CE/RPS runs."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score

from ordinal_uncertainty.data.retinamnist import retinamnist_loaders
from ordinal_uncertainty.metrics.calibration import fit_temperature, temperature_probabilities
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics
from ordinal_uncertainty.models.resnet import make_resnet18

SEEDS = range(5)
COVERAGES = np.round(np.arange(1.0, 0.099, -0.05), 2)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_paths(model: str, seed: int) -> tuple[Path, Path]:
    if model == "ce":
        root = Path("outputs/retinamnist/native28/single_model_baseline") / f"seed_{seed}"
        return root / "best_checkpoint.pt", root / "predictions.csv"
    root = Path("outputs/retinamnist/native28/phase2_model_comparison/rps") / f"seed_{seed}_artifact_complete"
    return root / "best_checkpoint.pt", root / "evaluation/predictions.csv"


def read_predictions(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["true_label"]) for row in rows], dtype=int)
    sample_ids = np.asarray([int(row["sample_id"]) for row in rows], dtype=int)
    logits = np.asarray([json.loads(row["logits"]) for row in rows], dtype=np.float32)
    probabilities = np.asarray([json.loads(row["probabilities"]) for row in rows], dtype=np.float64)
    if not np.isfinite(logits).all() or not np.isfinite(probabilities).all() or np.any(probabilities < -1e-7):
        raise ValueError(f"invalid saved predictions: {path}")
    if not np.allclose(probabilities.sum(1), 1.0, atol=1e-5):
        raise ValueError(f"probabilities do not sum to one: {path}")
    return sample_ids, labels, logits, probabilities


def validation_logits(model_name: str, checkpoint_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"] if model_name == "ce" else checkpoint["state_dict"]
    model = make_resnet18(5).to(device)
    model.load_state_dict(state)
    model.eval()
    loaders, _ = retinamnist_loaders(Path("data/medmnist"), 128, 0, False, 28)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in loaders["val"]:
            all_logits.append(model(images.to(device)).cpu())
            all_labels.append(labels.reshape(-1).cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def decision_rows(model: str, calibration: str, labels: np.ndarray, probabilities: np.ndarray) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], list[dict], dict[str, np.ndarray]]:
    decisions = bayes_decisions(probabilities)
    rows, alignment, detection, coverage, classwise, risk_by_error = [], [], [], [], [], []
    for decision, key, risk_key in (("mode", "mode_decision", "mode_l1_risk"), ("l1", "l1_bayes_decision", "l1_bayes_risk"), ("l2", "l2_bayes_decision", "l2_bayes_risk")):
        prediction, risk = decisions[key], decisions[risk_key]
        error = np.abs(labels - prediction)
        severe = error >= 2
        rows.append({"model": model, "calibration": calibration, "decision": decision, "accuracy": float((labels == prediction).mean()), "mae": float(error.mean()), "qwk": float(cohen_kappa_score(labels, prediction, weights="quadratic")), "severe_count": int(severe.sum()), "severe_prevalence": float(severe.mean())})
        alignment.append({"model": model, "calibration": calibration, "decision": decision, "spearman": float(spearmanr(risk, error).statistic)})
        detection.append({"model": model, "calibration": calibration, "decision": decision, "auroc": float(roc_auc_score(severe, risk)), "auprc": float(average_precision_score(severe, risk)), "severe_count": int(severe.sum()), "severe_prevalence": float(severe.mean())})
        ordering = np.argsort(risk, kind="stable")
        for cov in COVERAGES:
            retained = ordering[: max(1, int(np.ceil(cov * len(labels))))]
            coverage.append({"model": model, "calibration": calibration, "decision": decision, "coverage": float(cov), "ordinal_risk_mae": float(error[retained].mean()), "classification_risk": float((labels[retained] != prediction[retained]).mean())})
        if decision == "l1":
            for group, mask in (("correct", error == 0), ("adjacent", error == 1), ("severe", error >= 2)):
                values = risk[mask]
                risk_by_error.append({"model": model, "calibration": calibration, "error_group": group, "count": int(mask.sum()), "mean_l1_bayes_risk": float(values.mean()) if len(values) else None, "median_l1_bayes_risk": float(np.median(values)) if len(values) else None, "std_l1_bayes_risk": float(values.std(ddof=1)) if len(values) > 1 else None})
            for true_class in range(probabilities.shape[1]):
                mask = labels == true_class
                classwise.append({"model": model, "calibration": calibration, "true_class": true_class, "count": int(mask.sum()), "accuracy": float((prediction[mask] == labels[mask]).mean()), "mae": float(error[mask].mean()), "severe_count": int(severe[mask].sum()), "severe_prevalence": float(severe[mask].mean()), "mean_l1_bayes_risk": float(risk[mask].mean())})
    return rows, alignment, detection, coverage, classwise, risk_by_error, decisions


def aggregate(rows: list[dict], keys: list[str], group_keys: list[str]) -> list[dict]:
    results = []
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in group_keys), []).append(row)
    for group, members in groups.items():
        output = dict(zip(group_keys, group))
        for key in keys:
            values = np.asarray([float(row[key]) for row in members])
            output[f"{key}_mean"] = float(values.mean())
            output[f"{key}_std"] = float(values.std(ddof=1))
        results.append(output)
    return results


def main() -> None:
    output_root = Path("outputs/retinamnist/native28/phase2_5_calibration_audit")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root.mkdir(parents=True)
    all_probability, all_decision, all_alignment, all_detection, all_coverage, all_classwise, all_changes = [], [], [], [], [], [], []
    reference_labels: np.ndarray | None = None
    reference_sample_ids: np.ndarray | None = None
    all_risk_by_error = []
    for model in ("ce", "rps"):
        for seed in SEEDS:
            checkpoint, prediction_file = source_paths(model, seed)
            sample_ids, labels, test_logits, saved_probabilities = read_predictions(prediction_file)
            if reference_labels is None:
                reference_labels = labels
                reference_sample_ids = sample_ids
            elif not np.array_equal(reference_labels, labels):
                raise ValueError(f"test labels do not align for {model} seed {seed}")
            elif not np.array_equal(reference_sample_ids, sample_ids):
                raise ValueError(f"test sample identifiers do not align for {model} seed {seed}")
            val_logits, val_labels = validation_logits(model, checkpoint, device)
            temperature, val_nll_before, val_nll_after = fit_temperature(val_logits, val_labels)
            raw_probabilities = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
            if not np.allclose(raw_probabilities, saved_probabilities, atol=2e-5):
                raise ValueError(f"saved probabilities do not agree with logits for {model} seed {seed}")
            calibrated = temperature_probabilities(torch.from_numpy(test_logits), temperature).numpy()
            seed_output = output_root / model / f"seed_{seed}"
            seed_output.mkdir(parents=True)
            states = {"raw": raw_probabilities, "temperature_scaled": calibrated}
            seed_decisions: dict[str, dict[str, np.ndarray]] = {}
            for calibration, probabilities in states.items():
                probability_metrics = prediction_metrics(labels, probabilities)
                ece, _ = expected_calibration_error(labels, probabilities)
                probability_metrics["ece"] = ece
                all_probability.append({"model": model, "seed": seed, "calibration": calibration, **probability_metrics})
                rows, alignment, detection, coverage, classwise, risk_by_error, decisions = decision_rows(model, calibration, labels, probabilities)
                for collection in (rows, alignment, detection, coverage, classwise, risk_by_error):
                    for row in collection:
                        row["seed"] = seed
                all_decision.extend(rows); all_alignment.extend(alignment); all_detection.extend(detection); all_coverage.extend(coverage); all_classwise.extend(classwise); all_risk_by_error.extend(risk_by_error)
                seed_decisions[calibration] = decisions
            raw_l1, calibrated_l1 = seed_decisions["raw"]["l1_bayes_decision"], seed_decisions["temperature_scaled"]["l1_bayes_decision"]
            shift = calibrated_l1 - raw_l1
            raw_mode = seed_decisions["raw"]["mode_decision"]
            for calibration, decisions in seed_decisions.items():
                mode_to_l1 = decisions["l1_bayes_decision"] - decisions["mode_decision"]
                all_changes.append({"model": model, "seed": seed, "change_type": "mode_to_l1", "calibration": calibration, "changed_fraction": float((mode_to_l1 != 0).mean()), "mean_abs_shift": float(np.abs(mode_to_l1).mean()), "shift_1_fraction": float((np.abs(mode_to_l1) == 1).mean()), "shift_2plus_fraction": float((np.abs(mode_to_l1) >= 2).mean()), "upward_fraction": float((mode_to_l1 > 0).mean()), "downward_fraction": float((mode_to_l1 < 0).mean())})
            all_changes.append({"model": model, "seed": seed, "change_type": "raw_to_temperature_l1", "calibration": "temperature_scaled", "changed_fraction": float((shift != 0).mean()), "mean_abs_shift": float(np.abs(shift).mean()), "shift_1_fraction": float((np.abs(shift) == 1).mean()), "shift_2plus_fraction": float((np.abs(shift) >= 2).mean()), "upward_fraction": float((shift > 0).mean()), "downward_fraction": float((shift < 0).mean())})
            prediction_rows = [{"sample_id": int(sample_ids[index]), "true_label": int(label), "logits": json.dumps(test_logits[index].tolist()), "raw_probabilities": json.dumps(raw_probabilities[index].tolist()), "temperature_scaled_probabilities": json.dumps(calibrated[index].tolist())} for index, label in enumerate(labels)]
            write_csv(seed_output / "predictions_calibrated.csv", prediction_rows)
            (seed_output / "calibration.json").write_text(json.dumps({"model": model, "seed": seed, "temperature": temperature, "validation_nll_before": val_nll_before, "validation_nll_after": val_nll_after, "validation_sample_count": int(len(val_labels)), "test_sample_count": int(len(labels)), "fit_split": "official validation", "evaluation_split": "official test", "checkpoint": str(checkpoint)}, indent=2) + "\n")
    summary = output_root / "summary"; summary.mkdir()
    write_csv(summary / "probability_metrics_per_seed.csv", all_probability)
    write_csv(summary / "decision_metrics_per_seed.csv", all_decision)
    write_csv(summary / "risk_alignment_per_seed.csv", all_alignment)
    write_csv(summary / "severe_detection_per_seed.csv", all_detection)
    write_csv(summary / "risk_coverage_per_seed.csv", all_coverage)
    write_csv(summary / "classwise_l1_per_seed.csv", all_classwise)
    write_csv(summary / "l1_risk_by_error_per_seed.csv", all_risk_by_error)
    write_csv(summary / "decision_changes_per_seed.csv", all_changes)
    probability_summary = aggregate(all_probability, ["accuracy", "mae", "quadratic_weighted_kappa", "nll", "brier_score", "ranked_probability_score", "ece"], ["model", "calibration"])
    decision_summary = aggregate([row for row in all_decision if row["decision"] == "l1"], ["accuracy", "mae", "qwk", "severe_prevalence"], ["model", "calibration", "decision"])
    alignment_summary = aggregate([row for row in all_alignment if row["decision"] == "l1"], ["spearman"], ["model", "calibration", "decision"])
    detection_summary = aggregate([row for row in all_detection if row["decision"] == "l1"], ["auroc", "auprc", "severe_prevalence"], ["model", "calibration", "decision"])
    selective_seed = []
    for model in ("ce", "rps"):
        for calibration in ("raw", "temperature_scaled"):
            for seed in SEEDS:
                values = [row["ordinal_risk_mae"] for row in all_coverage if row["model"] == model and row["calibration"] == calibration and row["decision"] == "l1" and row["seed"] == seed]
                selective_seed.append({"model": model, "calibration": calibration, "seed": seed, "mean_mae_selective_risk": float(np.mean(values))})
    selective_summary = aggregate(selective_seed, ["mean_mae_selective_risk"], ["model", "calibration"])
    change_summary = aggregate(all_changes, ["changed_fraction", "mean_abs_shift", "shift_1_fraction", "shift_2plus_fraction", "upward_fraction", "downward_fraction"], ["model", "change_type", "calibration"])
    for filename, rows in (("probability_summary.csv", probability_summary), ("l1_decision_summary.csv", decision_summary), ("l1_risk_alignment_summary.csv", alignment_summary), ("l1_severe_detection_summary.csv", detection_summary), ("l1_selective_prediction_summary.csv", selective_summary), ("decision_change_summary.csv", change_summary)):
        write_csv(summary / filename, rows)
    (summary / "phase2_5_summary.json").write_text(json.dumps({"models": ["ce", "rps"], "seeds": list(SEEDS), "calibration": "scalar temperature scaling fit by validation NLL", "test_labels_used_for_fitting": False, "test_sample_count": int(len(reference_labels))}, indent=2) + "\n")


if __name__ == "__main__":
    main()
