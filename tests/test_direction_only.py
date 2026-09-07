import torch

from ordinal_uncertainty.evaluation.direction_only import DirectionOnlyLinear


def test_direction_only_replays_initial_head_and_preserves_norms_biases():
    weight = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    bias = torch.tensor([.3, -.2])
    features = torch.randn(4, 2)
    head = DirectionOnlyLinear(weight, bias)
    assert torch.allclose(head(features), features @ weight.T + bias)
    assert head.max_norm_error().item() <= 1e-6
    loss = head(features).sum(); loss.backward()
    assert head.direction.grad is not None
    assert head.fixed_norms.grad is None and head.fixed_bias.grad is None and features.grad is None
    with torch.no_grad(): head.direction.add_(torch.randn_like(head.direction))
    assert head.max_norm_error().item() <= 1e-6
    scaled = DirectionOnlyLinear(weight, bias, fixed_norms=torch.tensor([1.0, 3.0]))
    assert torch.allclose(scaled.effective_weight().norm(dim=1), torch.tensor([1.0, 3.0]))
