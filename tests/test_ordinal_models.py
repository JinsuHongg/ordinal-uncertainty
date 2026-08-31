import torch
import pytest
from ordinal_uncertainty.models.ordinal import CoralHead, coral_loss,coral_probabilities,coral_targets,rps_loss
def test_coral_targets_probs_and_gradients():
 x=torch.tensor([[3.,2.,1.]],requires_grad=True); p=coral_probabilities(x); assert torch.allclose(p.sum(1),torch.ones(1)); assert torch.all(p>=0); coral_loss(x,torch.tensor([2])).backward(); assert torch.isfinite(x.grad).all(); assert coral_targets(torch.tensor([2]),4).tolist()==[[1.,1.,0.]]
def test_rps_distance_normalization_and_gradients():
 logits=torch.tensor([[20.,-20.,-20.]],requires_grad=True); assert rps_loss(logits,torch.tensor([0])) < 1e-6; rps_loss(logits,torch.tensor([2])).backward();assert torch.isfinite(logits.grad).all()

def test_coral_conversion_rejects_nonmonotone_cumulative_outputs():
 with pytest.raises(ValueError): coral_probabilities(torch.tensor([[0., 2., 1.]]))

def test_coral_head_does_not_hard_constrain_threshold_ordering():
 head=CoralHead(2,4)
 with torch.no_grad(): head.bias.copy_(torch.tensor([0.,2.,1.]))
 logits=head(torch.randn(8,2)); assert not torch.all(logits[:,:-1] >= logits[:,1:])
