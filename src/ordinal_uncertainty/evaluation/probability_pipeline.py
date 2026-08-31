"""Shared, staged probability-evaluation finalization."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from ordinal_uncertainty.evaluation.ordinal_uncertainty import evaluate_predictions
from ordinal_uncertainty.metrics.decision import bayes_decisions
from ordinal_uncertainty.metrics.extreme_class import inward_shrinkage


def finalize_probability_evaluation(
    labels: np.ndarray,
    logits: np.ndarray,
    output_dir: Path,
    marker: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Validate Softmax outputs and persist the standard evaluation artifacts."""
    emit = marker or (lambda _message: None)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] != labels.size:
        raise ValueError("labels/logits shape mismatch")
    if not np.isfinite(logits).all():
        raise ValueError("logits contain NaN or Inf")
    emit("STAGE 6: logits/probabilities constructed")
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).double().numpy()
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(1), 1, atol=1e-7):
        raise ValueError("invalid Softmax probabilities")
    emit("STAGE 7: decision evaluation start")
    decisions = bayes_decisions(probabilities)
    emit("STAGE 8: classwise evaluation start")
    # The standard evaluator performs classwise aggregation after input validation.
    emit("STAGE 9: Phase 3.0 shrinkage diagnostics start")
    shrinkage = inward_shrinkage(labels, probabilities)
    if not np.isfinite(shrinkage).all():
        raise ValueError("shrinkage diagnostics are non-finite")
    emit("STAGE 10: evaluate_predictions start")
    emit("STAGE 11: artifact finalization start")
    summary = evaluate_predictions(labels, logits, probabilities, output_dir)
    emit("STAGE 12: artifact finalization complete")
    return {"probabilities": probabilities, "decisions": decisions, "shrinkage": shrinkage, "summary": summary}
