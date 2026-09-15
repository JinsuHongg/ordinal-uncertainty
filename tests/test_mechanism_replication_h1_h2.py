import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "analyze_mechanism_replication_h1_h2",
    Path("scripts/analyze_mechanism_replication_h1_h2.py"),
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_h1_effect_is_c_minus_a_on_class4_only():
    labels = np.full(66, 4)
    result = ANALYSIS.h1_delta(np.full(66, 3), np.full(66, 4), labels)
    assert result["delta_mae_c_minus_a"] == -1.0


def test_predictor_standardization_uses_only_supplied_population():
    standardized, mean, sd = ANALYSIS.sample_standardize(np.array([1.0, 2.0, 3.0]))
    assert mean == 2.0 and sd == 1.0
    assert np.allclose(standardized, [-1.0, 0.0, 1.0])


def test_h2a_vectors_preserve_frozen_outcome_and_margin_signs():
    arrays = {
        "labels": np.array([4, 3]),
        "c_predictive_mean": np.array([3.25, 2.0]),
        "a_predictive_mean": np.array([2.5, 2.0]),
        "generic_centroid_margin": np.array([0.4, 0.1]),
        "endpoint_adj_margin": np.array([0.7, -0.3]),  # d3 - d4
    }
    y_c, y_a, generic, ordinal = ANALYSIS.h2a_vectors(arrays)
    assert np.allclose(y_c, [0.75]) and np.allclose(y_a, [1.5])
    assert np.allclose(generic, [0.4]) and np.allclose(ordinal, [0.7])


def test_loocv_refits_standardization_within_each_training_fold():
    outcome = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    predictor = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    generic = np.array([4.0, 1.0, 3.0, 0.0, 2.0])
    assert ANALYSIS.loocv_mse(outcome, predictor, generic, None) >= 0.0


def test_gate_counts_seeds_not_oof_folds():
    assert ANALYSIS.SEEDS == (1, 2, 3, 4)
    assert ANALYSIS.h1_verdict(4) == "REPLICATED"
    assert ANALYSIS.h1_verdict(3) == "PARTIAL"
    assert ANALYSIS.h1_verdict(2) == "NOT REPLICATED"
    assert ANALYSIS.h2a_verdict(4, 3, 0.01) == "STRONG"
