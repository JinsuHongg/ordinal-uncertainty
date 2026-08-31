"""Small, testable diagnostics for ordinal extreme-class failure analysis."""
from __future__ import annotations

import numpy as np


def class_probability_means(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Mean predictive probability vector for each observed ordinal class."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    return np.stack([probabilities[labels == true_class].mean(0) for true_class in range(probabilities.shape[1])])


def ordinal_bias(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Predictive-mean bias ``E[mu - Y | Y=y]`` by true class."""
    labels = np.asarray(labels, dtype=int)
    classes = np.arange(probabilities.shape[1])
    means = np.asarray(probabilities, dtype=float) @ classes
    return np.asarray([(means[labels == true_class] - true_class).mean() for true_class in classes])


def inward_shrinkage(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Positive values indicate predictive means shifted toward the ordinal centre."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    centre = (probabilities.shape[1] - 1) / 2
    predictive_mean = probabilities @ np.arange(probabilities.shape[1])
    return np.abs(labels - centre) - np.abs(predictive_mean - centre)
