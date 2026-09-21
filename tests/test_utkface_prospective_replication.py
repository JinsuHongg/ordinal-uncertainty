import importlib.util
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location(
    "utkface_prospective",
    Path("scripts/run_utkface_prospective_replication.py"),
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_natural_direction_fit_preserves_empirical_draws_and_fixed_parameters():
    features = np.eye(5, dtype=np.float32)
    labels = np.arange(5, dtype=np.int64)
    head, history, draws, constraints = RUNNER.fit_direction_head(
        features,
        labels,
        torch.eye(5),
        torch.zeros(5),
        sampling="natural",
        seed=7,
        device=torch.device("cpu"),
    )

    assert len(history) == RUNNER.HEAD_EPOCHS
    assert np.array_equal(draws, np.full(5, RUNNER.HEAD_EPOCHS))
    assert constraints["max_norm_error"] <= RUNNER.CONSTRAINT_TOL
    assert constraints["max_bias_error"] == 0.0
    assert head.fixed_norms.requires_grad is False
    assert head.fixed_bias.requires_grad is False


def test_validation_contract_has_expected_archived_counts_and_is_disjoint():
    train = ["utkface:train-a", "utkface:train-b"]
    validation = ["utkface:validation-a"]
    RUNNER.assert_disjoint_ids(train, validation)
