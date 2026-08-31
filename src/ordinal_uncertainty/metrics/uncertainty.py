"""Uncertainty diagnostics derived from categorical predictive probabilities."""
from __future__ import annotations

import numpy as np


def uncertainty_metrics(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-example diagnostics, all oriented as higher means less certain.

    ``probabilities`` must be a two-dimensional, row-normalized array ordered by
    ordinal class label (0 through K-1).
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes >= 2)")
    if np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("probabilities must be non-negative and sum to one per row")
    clipped = np.clip(probabilities, np.finfo(np.float64).tiny, 1.0)
    classes = np.arange(probabilities.shape[1], dtype=np.float64)
    ordinal_mean = probabilities @ classes
    deviations = classes[None, :] - ordinal_mean[:, None]
    top_two = np.partition(probabilities, -2, axis=1)[:, -2:]
    return {
        "predictive_entropy": -(probabilities * np.log(clipped)).sum(axis=1),
        "confidence_uncertainty": 1.0 - probabilities.max(axis=1),
        "margin_uncertainty": 1.0 - (top_two.max(axis=1) - top_two.min(axis=1)),
        "ordinal_predictive_mean": ordinal_mean,
        "ordinal_variance": (probabilities * deviations**2).sum(axis=1),
        "ordinal_absolute_deviation": (probabilities * np.abs(deviations)).sum(axis=1),
    }


def phase1_5_uncertainty_metrics(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Literature and decision-centered ordinal measures for the Phase 1.5 audit.

    OCS uses the unnormalised sum over the K-1 order-consistent binary splits
    in Haas & Hüllermeier (2025), Eq. (10).  Ranking diagnostics are invariant
    to a positive normalization, so the paper's sum is retained verbatim.
    """
    base = uncertainty_metrics(probabilities)
    p = np.asarray(probabilities, dtype=np.float64)
    k = p.shape[1]
    cdf = np.cumsum(p, axis=1)[:, :-1]
    q = 1.0 - cdf
    binary_entropy = -(cdf * np.log(np.clip(cdf, np.finfo(float).tiny, 1)) + q * np.log(np.clip(q, np.finfo(float).tiny, 1)))
    classes = np.arange(k, dtype=float)
    prediction = p.argmax(axis=1)
    mean = p @ classes
    # Haas & Hüllermeier (2025), Eq. (6): Dnt = 1 - Cns.
    normalized_distance = np.abs(classes[None, :] - mean[:, None]) / (k - 1)
    term = np.where(p == 0, 0.0, p * np.log2(np.clip(1.0 - normalized_distance, np.finfo(float).tiny, 1.0)))
    # The only zero log argument has zero probability; force its limiting value.
    term[(p == 0) & (normalized_distance == 1)] = 0.0
    cns_dissention = -term.sum(axis=1)
    # Haas & Hüllermeier (2025), Eq. (5), transformed to dispersion 1-C2.
    c2_dispersion = 1.0 - ((cdf - 0.5) ** 2).sum(axis=1) / ((k - 1) / 4.0)
    decision_distances = np.abs(classes[None, :] - prediction[:, None])
    base.update({
        "ocs_entropy": binary_entropy.sum(axis=1),
        "ocs_variance": (cdf * q).sum(axis=1),
        "consensus_cns_dissention": cns_dissention,
        "consensus_c2_dispersion": c2_dispersion,
        "bayes_risk_l2": (p * decision_distances**2).sum(axis=1),
        "prediction_distance_l1": (p * decision_distances).sum(axis=1),
    })
    return base
