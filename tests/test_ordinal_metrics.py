import numpy as np

from ordinal_uncertainty.metrics.predictive import ranked_probability_score


def test_ranked_probability_score_is_zero_for_perfect_predictions():
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert ranked_probability_score(np.array([0, 2]), probabilities) == 0.0


def test_ranked_probability_score_penalizes_more_distant_probability():
    labels = np.array([0])
    adjacent = ranked_probability_score(labels, np.array([[0.0, 1.0, 0.0]]))
    distant = ranked_probability_score(labels, np.array([[0.0, 0.0, 1.0]]))
    assert distant > adjacent
