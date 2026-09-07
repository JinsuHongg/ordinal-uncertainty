"""Read-only linear-head decomposition helpers for frozen-feature audits."""
from __future__ import annotations

import numpy as np


def linear_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return feature terms and logits for ``z = h W^T + b``."""
    feature_terms = np.asarray(features, float) @ np.asarray(weight, float).T
    return feature_terms, feature_terms + np.asarray(bias, float)[None, :]


def cosine_alignment(features: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return sample-by-class cosine feature/classifier alignment."""
    numerator = np.asarray(features, float) @ np.asarray(weight, float).T
    denominator = np.linalg.norm(features, axis=1, keepdims=True) * np.linalg.norm(weight, axis=1)[None, :]
    return numerator / denominator


def margin_decomposition(feature_terms: np.ndarray, bias: np.ndarray, left: int, right: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Return feature, bias, and total contribution to ``z_left-z_right``."""
    feature = feature_terms[:, left] - feature_terms[:, right]
    bias_term = float(np.asarray(bias)[left] - np.asarray(bias)[right])
    return feature, bias_term, feature + bias_term


def swapped_parameters(original_weight: np.ndarray, original_bias: np.ndarray, balanced_weight: np.ndarray, balanced_bias: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Create one declared diagnostic-only original/balanced parameter swap."""
    if kind == "original_weights_balanced_biases":
        return np.asarray(original_weight), np.asarray(balanced_bias)
    if kind == "balanced_weights_original_biases":
        return np.asarray(balanced_weight), np.asarray(original_bias)
    raise ValueError(f"unknown swap kind: {kind}")


def common_norm_weights(weight: np.ndarray, reference_norm: float) -> np.ndarray:
    """Preserve directions while assigning every row one fixed diagnostic norm."""
    values = np.asarray(weight, float)
    return values / np.linalg.norm(values, axis=1, keepdims=True) * float(reference_norm)
