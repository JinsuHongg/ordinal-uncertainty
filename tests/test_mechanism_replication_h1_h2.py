import importlib.util
import json
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "analyze_mechanism_replication_h1_h2",
    Path("scripts/analyze_mechanism_replication_h1_h2.py"),
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def write_condition(tmp_path, dataset, *, sample_ids, folds, objective="ce", seed=1, manifest_dataset=None):
    root = tmp_path / dataset / objective / f"seed_{seed}"
    root.mkdir(parents=True)
    n, end_n = (1080, 66) if dataset == "retina" else (28006, 921)
    labels = np.zeros(n, dtype=np.int64)
    labels[-end_n:] = 4
    scalar = np.zeros(n, dtype=float)
    np.savez_compressed(
        root / "per_sample_arrays.npz",
        sample_ids=np.asarray(sample_ids), labels=labels, folds=np.asarray(folds),
        endpoint_adj_margin=scalar, generic_centroid_margin=scalar,
        a_logits=scalar, c_logits=scalar, a_probabilities=scalar, c_probabilities=scalar,
        a_l1=np.zeros(n, dtype=np.int64), c_l1=np.zeros(n, dtype=np.int64),
        a_predictive_mean=scalar, c_predictive_mean=scalar,
        a_inward_shrinkage=scalar, c_inward_shrinkage=scalar,
    )
    (root / "manifest.json").write_text(json.dumps({
        "dataset": manifest_dataset or dataset,
        "backbone_objective": objective,
        "backbone_seed": seed,
        "seed_role": "confirmatory",
    }))


def test_h1_effect_is_c_minus_a_on_class4_only():
    labels = np.full(66, 4)
    result = ANALYSIS.h1_delta(np.full(66, 3), np.full(66, 4), labels)
    assert result["delta_mae_c_minus_a"] == -1.0


def test_h1_delta_accepts_dataset_specific_endpoint_support():
    labels = np.array([4, 4, 0])
    result = ANALYSIS.h1_delta(np.array([3, 4, 0]), np.array([4, 4, 0]), labels, expected_end_n=2)
    assert result["delta_mae_c_minus_a"] == -0.5


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


def test_default_dataset_remains_retina():
    assert ANALYSIS.build_parser().parse_args([]).dataset == "retina"


def test_retina_population_and_oof_rules_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(ANALYSIS, "INPUT_BASE", tmp_path)
    write_condition(tmp_path, "retina", sample_ids=np.arange(1080), folds=np.arange(1080) % 5)
    _, arrays, _ = ANALYSIS.load_condition("retina", "ce", 1)
    assert arrays["labels"].shape == (1080,)
    assert int((arrays["labels"] == 4).sum()) == 66


def test_solar_accepts_unique_non_retina_ids_and_non_oof_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(ANALYSIS, "INPUT_BASE", tmp_path)
    write_condition(tmp_path, "solar", sample_ids=np.arange(50000, 78006), folds=np.full(28006, -1))
    _, arrays, _ = ANALYSIS.load_condition("solar", "ce", 1)
    assert arrays["labels"].shape == (28006,)
    assert int((arrays["labels"] == 4).sum()) == 921


def test_retina_rejects_noncanonical_ids_and_dataset_provenance_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(ANALYSIS, "INPUT_BASE", tmp_path)
    write_condition(tmp_path, "retina", sample_ids=np.arange(100, 1180), folds=np.arange(1080) % 5)
    try:
        ANALYSIS.load_condition("retina", "ce", 1)
    except ValueError as error:
        assert "sample ID coverage" in str(error)
    else:
        raise AssertionError("Retina accepted noncanonical sample IDs")
    write_condition(tmp_path, "solar", sample_ids=np.arange(28006), folds=np.full(28006, -1), seed=2, manifest_dataset="retina")
    try:
        ANALYSIS.load_condition("solar", "ce", 2)
    except ValueError as error:
        assert "provenance mismatch" in str(error)
    else:
        raise AssertionError("dataset provenance mismatch was accepted")
