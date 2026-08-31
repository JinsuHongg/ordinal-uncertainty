import numpy as np

from ordinal_uncertainty.metrics.uncertainty import phase1_5_uncertainty_metrics, uncertainty_metrics


def test_deterministic_distribution_has_zero_ordinal_spread():
    values = uncertainty_metrics(np.array([[0, 0, 1, 0, 0]], dtype=float))
    assert values["ordinal_variance"][0] == 0
    assert values["ordinal_absolute_deviation"][0] == 0


def test_adjacent_ambiguity_has_small_ordinal_spread():
    values = uncertainty_metrics(np.array([[0, 0.5, 0.5, 0, 0]], dtype=float))
    assert values["ordinal_variance"][0] == 0.25
    assert values["ordinal_absolute_deviation"][0] == 0.5


def test_distant_ambiguity_matches_entropy_but_has_much_larger_ordinal_spread():
    adjacent = uncertainty_metrics(np.array([[0, 0.5, 0.5, 0, 0]], dtype=float))
    distant = uncertainty_metrics(np.array([[0.5, 0, 0, 0, 0.5]], dtype=float))
    assert np.isclose(adjacent["predictive_entropy"][0], distant["predictive_entropy"][0])
    assert distant["ordinal_variance"][0] > adjacent["ordinal_variance"][0] * 10
    assert distant["ordinal_absolute_deviation"][0] > adjacent["ordinal_absolute_deviation"][0] * 3


def test_ocs_and_consensus_measures_are_order_sensitive_and_zero_for_dirac():
    deterministic = phase1_5_uncertainty_metrics(np.array([[0, 0, 1, 0, 0.]], dtype=float))
    adjacent = phase1_5_uncertainty_metrics(np.array([[0, .5, .5, 0, 0.]], dtype=float))
    extreme = phase1_5_uncertainty_metrics(np.array([[.5, 0, 0, 0, .5]], dtype=float))
    assert deterministic['ocs_entropy'][0] == 0 and deterministic['ocs_variance'][0] == 0
    assert np.isclose(adjacent['predictive_entropy'][0], extreme['predictive_entropy'][0])
    assert extreme['ocs_entropy'][0] > adjacent['ocs_entropy'][0]
    assert extreme['consensus_c2_dispersion'][0] > adjacent['consensus_c2_dispersion'][0]
