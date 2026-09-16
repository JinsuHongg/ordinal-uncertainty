import importlib.util
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location("retina_natural", Path("scripts/run_retina_natural_direction.py"))
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_natural_batches_preserve_empirical_epoch_counts():
    labels = torch.tensor([0] * 7 + [1] * 3)
    batches = RUNNER.natural_batch_indices(len(labels), 4, torch.Generator().manual_seed(7))
    indices = torch.cat(batches)
    assert torch.equal(torch.sort(indices).values, torch.arange(len(labels)))
    assert torch.equal(torch.bincount(labels[indices]), torch.tensor([7, 3]))


def test_fixed_norm_direction_head_keeps_norms_and_biases_fixed():
    head, _, _, constraints = RUNNER.fit_natural(np.eye(5, dtype=np.float32), np.arange(5), torch.eye(5), torch.zeros(5), 0, torch.device("cpu"))
    assert constraints["max_norm_error"] <= RUNNER.TOL
    assert constraints["max_bias_error"] == 0.0
    assert head.fixed_bias.requires_grad is False
