import torch

from ordinal_uncertainty.metrics.slace import inverse_frequency_weights, slace_loss, slace_matrices, weighted_cross_entropy


def test_weighted_ce_weights_and_gradients():
    weights = inverse_frequency_weights(torch.tensor([100, 10, 5]))
    assert torch.isclose(weights.mean(), torch.tensor(1.0)) and weights[2] > weights[1] > weights[0]
    logits = torch.randn(4, 3, requires_grad=True); loss = weighted_cross_entropy(logits, torch.tensor([0, 1, 2, 2]), weights); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_slace_probability_target_and_gradients():
    proximity, targets, dominance = slace_matrices(torch.tensor([100, 20, 5]), alpha=1.0)
    assert torch.allclose(targets.sum(1), torch.ones(3)) and dominance.shape == (3, 3, 3)
    logits = torch.randn(3, 3, requires_grad=True); loss = slace_loss(logits, torch.tensor([0, 1, 2]), targets, dominance); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all() and torch.isfinite(proximity).all()
