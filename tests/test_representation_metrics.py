import numpy as np

from ordinal_uncertainty.evaluation.representation import (
    class_centroids,
    cosine_distances,
    euclidean_distances,
    l2_normalize,
    nearest_centroid,
    within_class_dispersion,
)


def test_train_only_centroids_and_nearest_centroid_predictions():
    train_features = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 0.0], [12.0, 0.0]])
    train_labels = np.array([0, 0, 1, 1])
    centroids = class_centroids(train_features, train_labels, 2)
    assert np.allclose(centroids, [[1.0, 0.0], [11.0, 0.0]])
    held_out = np.array([[1.5, 0.0], [10.5, 0.0]])
    assert nearest_centroid(euclidean_distances(held_out, centroids)).tolist() == [0, 1]


def test_normalized_distances_margin_direction_and_dispersion():
    features = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    labels = np.array([0, 1, 1])
    centroids = class_centroids(features, labels, 2)
    normalized = l2_normalize(features)
    cosine = cosine_distances(normalized, class_centroids(normalized, labels, 2))
    assert cosine.shape == (3, 2)
    assert np.isfinite(cosine).all()
    raw = euclidean_distances(features, centroids)
    # Positive d_0 - d_1 means the sample is closer to class 1.
    assert raw[2, 0] - raw[2, 1] > 0
    dispersion = within_class_dispersion(features, labels, centroids)
    assert dispersion["count"].tolist() == [1, 2]
    assert np.isfinite(dispersion["mean"]).all()
