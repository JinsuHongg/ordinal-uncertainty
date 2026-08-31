"""Write all fixed Experiment 0 diagnostics from test predictions."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from ordinal_uncertainty.metrics.predictive import expected_calibration_error, prediction_metrics
from ordinal_uncertainty.metrics.uncertainty import uncertainty_metrics

UNCERTAINTY_COLUMNS = ("predictive_entropy", "confidence_uncertainty", "margin_uncertainty", "ordinal_variance", "ordinal_absolute_deviation")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _detection(target: np.ndarray, scores: np.ndarray) -> tuple[float | None, float | None]:
    if np.unique(target).size < 2:
        return None, None
    return float(roc_auc_score(target, scores)), float(average_precision_score(target, scores))


def evaluate_predictions(labels: np.ndarray, logits: np.ndarray, probabilities: np.ndarray, output_dir: Path, sample_ids: np.ndarray | None = None) -> dict[str, Any]:
    """Persist per-sample predictions and all specified raw-probability diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    logits = np.asarray(logits, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if logits.shape != probabilities.shape or len(labels) != len(probabilities):
        raise ValueError("labels, logits, and probabilities have incompatible shapes")
    sample_ids = np.arange(len(labels)) if sample_ids is None else np.asarray(sample_ids)
    predicted = probabilities.argmax(axis=1)
    ordinal_error = np.abs(labels - predicted)
    severity = np.where(ordinal_error == 0, "correct", np.where(ordinal_error == 1, "adjacent", "severe"))
    uncertainty = uncertainty_metrics(probabilities)
    prediction = prediction_metrics(labels, probabilities)
    ece, reliability = expected_calibration_error(labels, probabilities)
    prediction["ece"] = ece

    prediction_rows = []
    for i in range(len(labels)):
        row: dict[str, Any] = {"sample_id": int(sample_ids[i]), "true_label": int(labels[i]), "predicted_label": int(predicted[i]), "correct": bool(labels[i] == predicted[i]), "ordinal_error": int(ordinal_error[i]), "error_severity": str(severity[i]), "logits": json.dumps(logits[i].tolist()), "probabilities": json.dumps(probabilities[i].tolist())}
        row.update({name: float(values[i]) for name, values in uncertainty.items()})
        prediction_rows.append(row)
    _write_csv(output_dir / "predictions.csv", prediction_rows)
    _write_csv(output_dir / "reliability_bins.csv", reliability)

    severity_rows: list[dict[str, Any]] = []
    for measure in UNCERTAINTY_COLUMNS:
        for label, mask in (("correct", ordinal_error == 0), ("adjacent", ordinal_error == 1), ("severe", ordinal_error >= 2)):
            values = uncertainty[measure][mask]
            severity_rows.append({"measure": measure, "error_severity": label, "count": int(mask.sum()), "mean": float(values.mean()) if len(values) else None, "std": float(values.std(ddof=1)) if len(values) > 1 else None, "median": float(np.median(values)) if len(values) else None})
    _write_csv(output_dir / "uncertainty_by_error.csv", severity_rows)

    detection_rows: list[dict[str, Any]] = []
    association: dict[str, dict[str, float | None]] = {}
    for measure in UNCERTAINTY_COLUMNS:
        correlation = spearmanr(uncertainty[measure], ordinal_error)
        association[measure] = {"spearman_correlation": float(correlation.statistic) if np.isfinite(correlation.statistic) else None, "spearman_p_value": float(correlation.pvalue) if np.isfinite(correlation.pvalue) else None}
        for target_name, target in (("any_error", ordinal_error > 0), ("severe_error", ordinal_error >= 2)):
            auroc, auprc = _detection(target.astype(int), uncertainty[measure])
            detection_rows.append({"measure": measure, "target": target_name, "positive_count": int(target.sum()), "negative_count": int((~target).sum()), "auroc": auroc, "auprc": auprc})
    _write_csv(output_dir / "uncertainty_detection.csv", detection_rows)

    coverages = np.round(np.arange(1.0, 0.099, -0.05), 2)
    curve_rows: list[dict[str, Any]] = []
    for measure in UNCERTAINTY_COLUMNS:
        ordering = np.argsort(uncertainty[measure], kind="stable")
        for coverage in coverages:
            retained = ordering[:max(1, int(np.ceil(coverage * len(labels))))]
            curve_rows.append({"measure": measure, "coverage": float(coverage), "retained_count": int(len(retained)), "classification_risk": float(np.mean(labels[retained] != predicted[retained])), "ordinal_risk_mae": float(ordinal_error[retained].mean())})
    _write_csv(output_dir / "risk_coverage.csv", curve_rows)

    class_rows: list[dict[str, Any]] = []
    for true_class in range(probabilities.shape[1]):
        mask = labels == true_class
        class_rows.append({"true_class": true_class, "count": int(mask.sum()), "accuracy": float(np.mean(predicted[mask] == labels[mask])) if mask.any() else None, "mae": float(ordinal_error[mask].mean()) if mask.any() else None, "severe_error_count": int((ordinal_error[mask] >= 2).sum()), "severe_error_rate": float(np.mean(ordinal_error[mask] >= 2)) if mask.any() else None, **{f"mean_{measure}": float(uncertainty[measure][mask].mean()) if mask.any() else None for measure in UNCERTAINTY_COLUMNS}})
    _write_csv(output_dir / "classwise_metrics.csv", class_rows)
    summary = {"prediction_metrics": prediction, "association_with_ordinal_error": association, "test_count": int(len(labels)), "any_error_count": int((ordinal_error > 0).sum()), "severe_error_count": int((ordinal_error >= 2).sum()), "severe_error_note": "Detection metrics are undefined when only one target class is present; inspect severe_error_count when interpreting estimates."}
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary
