import numpy as np
import torch
from torch import nn

from ordinal_uncertainty.evaluation.frozen_head import (
    balanced_ce_loss,
    class_priors,
    feature_nearest_strata,
    freeze_backbone,
    head_loss,
    inverse_frequency_weights,
    logit_adjusted_ce_loss,
)


def test_backbone_freezing_and_head_dimension_compatibility():
    backbone = nn.Sequential(nn.Linear(3, 512), nn.ReLU())
    freeze_backbone(backbone)
    assert all(not parameter.requires_grad for parameter in backbone.parameters())
    head = nn.Linear(512, 5)
    assert head(torch.randn(4, 512)).shape == (4, 5)


def test_training_only_priors_and_balanced_ce_weights():
    labels = np.array([0, 0, 0, 1, 2, 3, 4])
    priors = class_priors(labels, 5)
    weights = inverse_frequency_weights(labels, 5)
    assert np.allclose(priors, [3 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7])
    assert weights[4] > weights[0]
    assert np.isclose(weights.mean(), 1.0)
    logits = torch.randn(7, 5, requires_grad=True)
    loss = balanced_ce_loss(logits, torch.tensor(labels), torch.tensor(weights))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_logit_adjustment_is_training_time_plus_log_prior():
    logits = torch.zeros(1, 2, requires_grad=True)
    priors = torch.tensor([0.8, 0.2])
    loss = logit_adjusted_ce_loss(logits, torch.tensor([1]), priors, tau=1.0)
    expected = -torch.log(torch.tensor(0.2))
    assert torch.allclose(loss.detach(), expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_rps_head_loss_and_frozen_features_have_no_gradient_flow():
    features = torch.randn(3, 512).detach()
    head = nn.Linear(512, 5)
    logits = head(features)
    loss = head_loss("rps", logits, torch.tensor([0, 2, 4]))
    loss.backward()
    assert features.grad is None
    assert torch.isfinite(loss) and torch.isfinite(head.weight.grad).all()


def test_feature_nearest_stratification():
    labels = np.array([4, 4, 3, 4])
    nearest = np.array([4, 2, 4, 3])
    assert feature_nearest_strata(labels, nearest).tolist() == [
        "feature_nearest_true", "feature_nearest_other", "not_target", "feature_nearest_other"
    ]
