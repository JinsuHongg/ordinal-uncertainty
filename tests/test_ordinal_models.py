import torch
import pytest
from ordinal_uncertainty.models.ordinal import (
    CoralHead,
    coral_loss,
    coral_probabilities,
    coral_targets,
    endpoint_neighborhood_loss,
    endpoint_neighborhood_rps_loss,
    endpoint_neighborhood_weights,
    endpoint_preference_loss,
    endpoint_preference_rps_loss,
    adjacent_classes,
    l1_bayes_risk,
    rg_acr_loss,
    rps_loss,
)
def test_coral_targets_probs_and_gradients():
 x=torch.tensor([[3.,2.,1.]],requires_grad=True); p=coral_probabilities(x); assert torch.allclose(p.sum(1),torch.ones(1)); assert torch.all(p>=0); coral_loss(x,torch.tensor([2])).backward(); assert torch.isfinite(x.grad).all(); assert coral_targets(torch.tensor([2]),4).tolist()==[[1.,1.,0.]]
def test_rps_distance_normalization_and_gradients():
 logits=torch.tensor([[20.,-20.,-20.]],requires_grad=True); assert rps_loss(logits,torch.tensor([0])) < 1e-6; rps_loss(logits,torch.tensor([2])).backward();assert torch.isfinite(logits.grad).all()


def test_endpoint_neighborhood_interior_and_lambda_zero_equal_rps():
 logits = torch.tensor([[0.2, -0.1, 0.4, 0.0, -0.3]], requires_grad=True)
 labels = torch.tensor([2])
 weights = endpoint_neighborhood_weights(torch.tensor([486, 128, 206, 194, 66]))
 assert endpoint_neighborhood_loss(logits, labels, weights) == 0
 assert torch.allclose(endpoint_neighborhood_rps_loss(logits, labels, weights, 0.0), rps_loss(logits, labels))


def test_endpoint_neighborhood_mirror_symmetry_except_frequency_weight():
 probabilities = torch.tensor([[0.10, 0.20, 0.15, 0.25, 0.30]])
 logits = probabilities.log()
 mirrored_logits = probabilities.flip(1).log()
 equal_weights = torch.ones(5)
 assert torch.allclose(
  endpoint_neighborhood_loss(logits, torch.tensor([4]), equal_weights),
  endpoint_neighborhood_loss(mirrored_logits, torch.tensor([0]), equal_weights),
 )
 weights = endpoint_neighborhood_weights(torch.tensor([486, 128, 206, 194, 66]))
 assert weights[4] == 1 and weights[0] < 1 and torch.all(weights[1:4] == 0)


def test_endpoint_neighborhood_prefers_near_endpoint_and_penalizes_far_mass():
 weights = torch.ones(5)
 labels = torch.tensor([4])
 near = torch.tensor([[0.10, 0.10, 0.10, 0.30, 0.40]]).log()
 central = torch.tensor([[0.10, 0.10, 0.50, 0.20, 0.10]]).log()
 assert endpoint_neighborhood_loss(near, labels, weights) < endpoint_neighborhood_loss(central, labels, weights)


def test_endpoint_neighborhood_loss_has_finite_gradients():
 logits = torch.randn(4, 5, requires_grad=True)
 labels = torch.tensor([0, 1, 3, 4])
 weights = endpoint_neighborhood_weights(torch.tensor([486, 128, 206, 194, 66]))
 loss = endpoint_neighborhood_rps_loss(logits, labels, weights, 0.3)
 loss.backward()
 assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_endpoint_preference_interior_lambda_zero_and_endpoint_symmetry():
 logits = torch.tensor([[0.1, 0.2, 0.3, -0.1, -0.2]])
 weights = endpoint_neighborhood_weights(torch.tensor([486, 128, 206, 194, 66]))
 assert endpoint_preference_loss(logits, torch.tensor([2]), weights) == 0
 assert torch.allclose(endpoint_preference_rps_loss(logits, torch.tensor([2]), weights, 0.0), rps_loss(logits, torch.tensor([2])))
 probabilities = torch.tensor([[0.1, 0.2, 0.15, 0.25, 0.3]])
 assert torch.allclose(
  endpoint_preference_loss(probabilities.log(), torch.tensor([4]), torch.ones(5)),
  endpoint_preference_loss(probabilities.flip(1).log(), torch.tensor([0]), torch.ones(5)),
 )


def test_endpoint_preference_distinguishes_true_from_adjacent_at_equal_neighborhood_mass():
 weights = torch.ones(5); label = torch.tensor([4])
 true_high = torch.tensor([[0.1, 0.1, 0.2, 0.2, 0.4]]).log()
 adjacent_high = torch.tensor([[0.1, 0.1, 0.2, 0.4, 0.2]]).log()
 assert torch.allclose(endpoint_neighborhood_loss(true_high, label, weights), endpoint_neighborhood_loss(adjacent_high, label, weights))
 assert endpoint_preference_loss(true_high, label, weights) < endpoint_preference_loss(adjacent_high, label, weights)


def test_endpoint_preference_penalizes_far_mass_and_has_finite_gradients():
 weights = torch.ones(5); label = torch.tensor([4])
 near = torch.tensor([[0.1, 0.1, 0.1, 0.3, 0.4]]).log()
 far = torch.tensor([[0.1, 0.1, 0.5, 0.2, 0.1]]).log()
 assert endpoint_preference_loss(near, label, weights) < endpoint_preference_loss(far, label, weights)
 logits = torch.randn(4, 5, requires_grad=True)
 loss = endpoint_preference_rps_loss(logits, torch.tensor([0, 1, 3, 4]), weights, 0.3)
 loss.backward()
 assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()

def test_coral_conversion_rejects_nonmonotone_cumulative_outputs():
 with pytest.raises(ValueError): coral_probabilities(torch.tensor([[0., 2., 1.]]))

def test_coral_head_does_not_hard_constrain_threshold_ordering():
 head=CoralHead(2,4)
 with torch.no_grad(): head.bias.copy_(torch.tensor([0.,2.,1.]))
 logits=head(torch.randn(8,2)); assert not torch.all(logits[:,:-1] >= logits[:,1:])


def test_rg_acr_adjacency_leave_one_out_and_missing_adjacent_handling():
 assert adjacent_classes(0, 5) == (1,) and adjacent_classes(4, 5) == (3,)
 assert adjacent_classes(2, 5) == (1, 3)
 features = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.]], requires_grad=True)
 logits = torch.tensor([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.]], requires_grad=True)
 labels = torch.tensor([0, 0, 1])
 loss, diagnostic = rg_acr_loss(logits, features, labels, margin=.05)
 assert diagnostic["valid_mask"].tolist() == [True, True, False]
 assert diagnostic["adjacent_terms"].tolist() == [2, 0, 0]
 assert torch.isfinite(loss) and loss >= 0


def test_rg_acr_empty_valid_batch_detaches_risk_and_reaches_features():
 features = torch.randn(3, 4, requires_grad=True)
 logits = torch.randn(3, 3, requires_grad=True)
 labels = torch.tensor([0, 1, 2])
 empty, diagnostic = rg_acr_loss(logits, features, labels)
 assert empty == 0 and not diagnostic["valid_mask"].any()
 empty.backward()
 assert torch.isfinite(features.grad).all()
 features = torch.tensor([[1., 0.], [.8, .2], [0., 1.]], requires_grad=True)
 logits = torch.tensor([[1., .2, -.2], [.3, 1., -.2], [0., 1., .1]], requires_grad=True)
 labels = torch.tensor([0, 0, 1])
 loss, diagnostic = rg_acr_loss(logits, features, labels, margin=.05, risk_cap=2.)
 loss.backward()
 assert torch.isfinite(features.grad).all() and logits.grad is None
 assert diagnostic["weights"].max() <= 2 and diagnostic["weights"][diagnostic["valid_mask"]].mean() > 0


def test_l1_bayes_risk_and_rg_acr_margin_are_finite():
 logits = torch.tensor([[4., -2., -2.], [0., 0., 0.]], requires_grad=True)
 risk = l1_bayes_risk(logits)
 assert risk.shape == (2,) and risk[0] < risk[1]
 features = torch.randn(4, 3, requires_grad=True)
 labels = torch.tensor([0, 0, 1, 1])
 loss, _ = rg_acr_loss(torch.randn(4, 3), features, labels, margin=.05)
 assert torch.isfinite(loss)
