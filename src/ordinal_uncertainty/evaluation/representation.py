"""Train-referenced representation geometry utilities for frozen-model audits."""
from __future__ import annotations

import numpy as np


def l2_normalize(features: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    """Normalize each feature row; reject zero or non-finite feature vectors."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("features must be a finite 2-D array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("cannot normalize zero-norm feature vectors")
    return values / norms


def class_centroids(features: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Compute one centroid per class from the supplied reference features only."""
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != targets.size:
        raise ValueError("features/labels shape mismatch")
    if not np.isfinite(values).all() or np.any(targets < 0) or np.any(targets >= num_classes):
        raise ValueError("invalid features or labels")
    centroids = []
    for class_index in range(num_classes):
        members = values[targets == class_index]
        if not len(members):
            raise ValueError(f"reference split has no examples of class {class_index}")
        centroids.append(members.mean(axis=0))
    return np.vstack(centroids)


def euclidean_distances(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return the sample-by-centroid Euclidean distance matrix."""
    values = np.asarray(features, dtype=np.float64)
    reference = np.asarray(centroids, dtype=np.float64)
    if values.ndim != 2 or reference.ndim != 2 or values.shape[1] != reference.shape[1]:
        raise ValueError("feature and centroid dimensions must match")
    distances = np.linalg.norm(values[:, None, :] - reference[None, :, :], axis=2)
    if not np.isfinite(distances).all():
        raise ValueError("non-finite Euclidean distances")
    return distances


def cosine_distances(normalized_features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Return cosine distances using normalized samples and normalized centroids."""
    values = l2_normalize(normalized_features)
    reference = l2_normalize(centroids)
    distances = 1.0 - values @ reference.T
    if not np.isfinite(distances).all():
        raise ValueError("non-finite cosine distances")
    return distances


def nearest_centroid(distances: np.ndarray) -> np.ndarray:
    """Select the smallest-index nearest centroid under deterministic ties."""
    matrix = np.asarray(distances, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("distances must be a finite 2-D array")
    return matrix.argmin(axis=1)


def within_class_dispersion(features: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize each class's distances to its train-derived centroid."""
    distances = euclidean_distances(features, centroids)
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    own = distances[np.arange(targets.size), targets]
    rows = {"mean": [], "median": [], "std": [], "count": []}
    for class_index in range(centroids.shape[0]):
        values = own[targets == class_index]
        rows["mean"].append(values.mean())
        rows["median"].append(np.median(values))
        rows["std"].append(values.std(ddof=1) if len(values) > 1 else 0.0)
        rows["count"].append(len(values))
    return {name: np.asarray(values) for name, values in rows.items()}
