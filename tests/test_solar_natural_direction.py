import importlib.util
from pathlib import Path

import numpy as np
import torch


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load("scripts/run_solar_natural_direction.py", "solar_natural")
ANALYZER = load("scripts/analyze_solar_rps_natural_direction.py", "solar_natural_analysis")


def test_natural_batches_preserve_empirical_counts():
    labels = torch.tensor([0] * 7 + [1] * 3)
    batches = RUNNER.natural_batch_indices(len(labels), 4, torch.Generator().manual_seed(7))
    indices = torch.cat(batches)
    assert torch.equal(torch.sort(indices).values, torch.arange(len(labels)))
    assert torch.equal(torch.bincount(labels[indices]), torch.tensor([7, 3]))


def test_direction_only_fit_preserves_norms_biases_and_draw_counts():
    features = np.eye(5, dtype=np.float32)
    labels = np.arange(5, dtype=np.int64)
    head, _, draws, constraints = RUNNER.fit_natural(
        features, labels, torch.eye(5), torch.zeros(5), torch.device("cpu")
    )
    assert np.array_equal(draws, np.full(5, RUNNER.EPOCHS))
    assert constraints["max_norm_error"] <= RUNNER.TOL
    assert constraints["max_bias_error"] == 0.0
    assert constraints["stored_norm_error"] == 0.0
    assert head.fixed_norms.requires_grad is False
    assert head.fixed_bias.requires_grad is False


def test_exact_l1_metrics_and_endpoint_mass_are_computed_per_class():
    y = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    decision = np.array([0, 1, 2, 3, 3], dtype=np.int64)
    probability = np.eye(5, dtype=np.float64)
    values = ANALYZER.metric(y, decision, probability)
    assert values["global_mae"] == 0.2
    assert values["mae_4"] == 1.0
    assert values["recall_4"] == 0.0
    assert values["p_end_4"] == 1.0
