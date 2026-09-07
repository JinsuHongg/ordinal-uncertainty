import pytest
import torch
from torch import nn

from ordinal_uncertainty.evaluation.risk_order import (
    detached_l1_bayes_risk,
    discrete_l1_action_risks,
    risk_order_preservation_loss,
)


def test_exact_discrete_l1_risk_and_detached_action_gradient():
    probabilities = torch.tensor([[0.1, 0.2, 0.7]], requires_grad=True)
    risks = discrete_l1_action_risks(probabilities)
    risk, action = detached_l1_bayes_risk(probabilities)
    assert torch.allclose(risks, torch.tensor([[1.6, 0.8, 0.4]]))
    assert action.tolist() == [2]
    assert not action.requires_grad and risk.requires_grad
    risk.backward()
    assert torch.isfinite(probabilities.grad).all()


def test_rop_zero_for_correct_order_positive_for_reversal_and_ignores_ties():
    teacher = torch.tensor([0.1, 0.4, 0.1])
    correct, correct_stats = risk_order_preservation_loss(torch.tensor([0.2, 0.8, 0.2], requires_grad=True), teacher)
    reversed_loss, reversed_stats = risk_order_preservation_loss(torch.tensor([0.8, 0.2, 0.8], requires_grad=True), teacher)
    assert correct.item() == 0.0
    assert reversed_loss.item() > 0.0
    assert correct_stats["valid_pair_count"].item() == 2.0
    assert correct_stats["tied_pair_fraction"].item() == pytest.approx(1 / 3)
    assert reversed_stats["weighted_violation_fraction"].item() == 1.0


def test_rop_gradient_reaches_head_but_not_frozen_features_or_teacher():
    features = torch.randn(5, 4).detach()
    teacher_probabilities = torch.softmax(torch.randn(5, 3), dim=1).detach()
    teacher_risk, _ = detached_l1_bayes_risk(teacher_probabilities)
    head = nn.Linear(4, 3)
    student_risk, _ = detached_l1_bayes_risk(torch.softmax(head(features), dim=1))
    loss, _ = risk_order_preservation_loss(student_risk, teacher_risk)
    loss.backward()
    assert features.grad is None
    assert teacher_probabilities.grad is None
    assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
