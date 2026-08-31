"""Ordinal-aware predictive and calibration metric helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, log_loss


def ranked_probability_score(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Multiclass RPS: mean summed squared CDF error, divided by K - 1.

    Lower is better; a one-hot CDF is used for each observed ordinal label.
    """
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    k = probabilities.shape[1]
    if k < 2:
        raise ValueError("RPS requires at least two classes")
    observed = np.eye(k)[labels]
    return float(np.mean(np.sum((np.cumsum(probabilities, axis=1)[:, :-1] - np.cumsum(observed, axis=1)[:, :-1]) ** 2, axis=1) / (k - 1)))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict[str, float | int]]]:
    """Top-label ECE and reliability-bin records using fixed-width bins."""
    labels = np.asarray(labels, dtype=int)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    rows: list[dict[str, float | int]] = []
    ece = 0.0
    for index in range(n_bins):
        lower, upper = index / n_bins, (index + 1) / n_bins
        mask = (confidence >= lower) & ((confidence < upper) if index < n_bins - 1 else (confidence <= upper))
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else float("nan")
        mean_confidence = float(confidence[mask].mean()) if count else float("nan")
        if count:
            ece += count / len(labels) * abs(accuracy - mean_confidence)
        rows.append({"bin": index, "lower": lower, "upper": upper, "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    return float(ece), rows


def prediction_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return the fixed Experiment 0 predictive metrics (lower is better except accuracy/QWK)."""
    labels = np.asarray(labels, dtype=int)
    predicted = probabilities.argmax(axis=1)
    one_hot = np.eye(probabilities.shape[1])[labels]
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "mae": float(np.abs(labels - predicted).mean()),
        "quadratic_weighted_kappa": float(cohen_kappa_score(labels, predicted, weights="quadratic")),
        "nll": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
        "brier_score": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ranked_probability_score": ranked_probability_score(labels, probabilities),
    }
