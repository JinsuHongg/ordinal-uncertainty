import numpy as np

from ordinal_uncertainty.metrics.extreme_class import class_probability_means, inward_shrinkage, ordinal_bias


def test_class_probability_means_and_ordinal_bias_signs():
    labels = np.array([0, 0, 1, 2, 2])
    probabilities = np.array([[.8, .2, 0], [.6, .4, 0], [.2, .6, .2], [0, .2, .8], [0, .4, .6]])
    means = class_probability_means(labels, probabilities)
    assert np.allclose(means[0], [.7, .3, 0])
    assert np.allclose(means[2], [0, .3, .7])
    bias = ordinal_bias(labels, probabilities)
    assert bias[0] > 0  # class 0 is biased upward
    assert bias[2] < 0  # class 2 is biased downward


def test_inward_shrinkage_is_positive_for_extremes_moved_to_centre():
    labels = np.array([0, 2])
    probabilities = np.array([[.25, .5, .25], [.25, .5, .25]])
    values = inward_shrinkage(labels, probabilities)
    assert np.allclose(values, [1.0, 1.0])
