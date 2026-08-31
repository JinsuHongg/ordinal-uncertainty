import torch

from ordinal_uncertainty.metrics.calibration import fit_temperature, temperature_probabilities
from ordinal_uncertainty.metrics.decision import bayes_decisions


def test_temperature_one_reproduces_softmax_and_preserves_argmax():
    logits = torch.tensor([[2.0, -1.0, 0.5], [-2.0, 0.25, 1.5]])
    probabilities = temperature_probabilities(logits, 1.0)
    assert torch.allclose(probabilities, torch.softmax(logits, dim=1))
    assert torch.allclose(probabilities.sum(1), torch.ones(2))
    assert torch.equal(temperature_probabilities(logits, 2.3).argmax(1), logits.argmax(1))


def test_temperature_fit_is_finite_positive_and_decision_risks_are_finite():
    logits = torch.tensor([[4.0, 0.0], [3.0, 0.0], [0.0, 3.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1, 1, 0])
    temperature, before, after = fit_temperature(logits, labels)
    probabilities = temperature_probabilities(logits, temperature)
    assert temperature > 0 and torch.isfinite(torch.tensor(temperature))
    assert torch.isfinite(torch.tensor([before, after])).all()
    assert torch.allclose(probabilities.sum(1), torch.ones(4))
    risks = bayes_decisions(probabilities.numpy())
    assert all(torch.isfinite(torch.from_numpy(value)).all() for value in risks.values())
