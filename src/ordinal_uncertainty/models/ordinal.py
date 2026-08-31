"""CORAL output conversion and differentiable ordinal probability losses."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
 return (labels[:,None] > torch.arange(num_classes-1,device=labels.device)).float()


def coral_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
 return F.binary_cross_entropy_with_logits(logits,coral_targets(labels,logits.shape[1] + 1),reduction='mean')


def coral_probabilities(logits: torch.Tensor) -> torch.Tensor:
 q=torch.sigmoid(logits); p=torch.cat([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],1)
 if torch.any(p < -1e-6): raise ValueError('CORAL cumulative probabilities are not monotone; refusing to repair them')
 return p.clamp_min(0) / p.clamp_min(0).sum(1,keepdim=True)


def coral_prediction(logits: torch.Tensor) -> torch.Tensor:
 """Official CORAL threshold-count prediction (sigmoid(logit) > .5)."""
 return (torch.sigmoid(logits) > .5).sum(1)


def rps_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
 p=F.softmax(logits,1); target=F.one_hot(labels,num_classes=logits.shape[1]).float()
 return ((p.cumsum(1)[:,:-1]-target.cumsum(1)[:,:-1])**2).sum(1).mean()/(logits.shape[1]-1)


def endpoint_neighborhood_weights(training_counts: torch.Tensor) -> torch.Tensor:
 """Return symmetric endpoint weights from positive ordinal training counts.

 The rarer endpoint has weight one; the more common endpoint is attenuated.
 Interior entries are zero because the endpoint correction does not apply there.
 """
 counts = torch.as_tensor(training_counts, dtype=torch.float)
 if counts.ndim != 1 or counts.numel() < 2 or bool((counts <= 0).any()):
  raise ValueError("training_counts must be positive 1-D counts for at least two classes")
 weights = torch.zeros_like(counts)
 endpoint_counts = counts[[0, -1]]
 weights[0] = torch.sqrt(endpoint_counts.min() / endpoint_counts[0])
 weights[-1] = torch.sqrt(endpoint_counts.min() / endpoint_counts[-1])
 return weights


def endpoint_neighborhood_loss(
 logits: torch.Tensor, labels: torch.Tensor, endpoint_weights: torch.Tensor
) -> torch.Tensor:
 """Mean endpoint-neighborhood negative log mass for Softmax ordinal outputs.

 For true label zero, the event is classes ``{0, 1}``; for true label ``K-1``,
 it is ``{K-2, K-1}``.  Interior labels contribute exactly zero.
 """
 if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
  raise ValueError("logits must be [batch, classes] and labels must be [batch]")
 if endpoint_weights.ndim != 1 or endpoint_weights.numel() != logits.shape[1]:
  raise ValueError("endpoint_weights must have one entry per class")
 if bool((labels < 0).any()) or bool((labels >= logits.shape[1]).any()):
  raise ValueError("labels are outside the ordinal class range")
 probabilities = F.softmax(logits, dim=1)
 if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
  raise ValueError("Softmax probabilities must be finite and non-negative")
 lower_mass = probabilities[:, :2].sum(dim=1)
 upper_mass = probabilities[:, -2:].sum(dim=1)
 is_lower = labels == 0
 is_upper = labels == logits.shape[1] - 1
 neighborhood_mass = torch.where(is_lower, lower_mass, upper_mass)
 active = is_lower | is_upper
 safety_floor = torch.finfo(probabilities.dtype).tiny
 per_example = torch.where(
  active,
  -neighborhood_mass.clamp_min(safety_floor).log() * endpoint_weights.to(logits.device)[labels],
  torch.zeros_like(neighborhood_mass),
 )
 return per_example.mean()


def endpoint_neighborhood_rps_loss(
 logits: torch.Tensor, labels: torch.Tensor, endpoint_weights: torch.Tensor, lambda_: float
) -> torch.Tensor:
 """RPS plus the fixed-radius, endpoint-neighborhood mass correction."""
 if lambda_ < 0:
  raise ValueError("lambda_ must be non-negative")
 return rps_loss(logits, labels) + lambda_ * endpoint_neighborhood_loss(logits, labels, endpoint_weights)


def endpoint_preference_loss(
 logits: torch.Tensor, labels: torch.Tensor, endpoint_weights: torch.Tensor, rho: float = 0.5
) -> torch.Tensor:
 """Endpoint correction that gives partial (``rho``) credit to its neighbor."""
 if not 0 < rho < 1:
  raise ValueError("rho must be strictly between zero and one")
 if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
  raise ValueError("logits must be [batch, classes] and labels must be [batch]")
 if endpoint_weights.ndim != 1 or endpoint_weights.numel() != logits.shape[1]:
  raise ValueError("endpoint_weights must have one entry per class")
 if bool((labels < 0).any()) or bool((labels >= logits.shape[1]).any()):
  raise ValueError("labels are outside the ordinal class range")
 probabilities = F.softmax(logits, dim=1)
 if not bool(torch.isfinite(probabilities).all()) or bool((probabilities < 0).any()):
  raise ValueError("Softmax probabilities must be finite and non-negative")
 is_lower = labels == 0
 is_upper = labels == logits.shape[1] - 1
 active = is_lower | is_upper
 true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
 adjacent_indices = torch.where(is_lower, torch.ones_like(labels), torch.full_like(labels, logits.shape[1] - 2))
 adjacent_probability = probabilities.gather(1, adjacent_indices[:, None]).squeeze(1)
 preferred_mass = true_probability + rho * adjacent_probability
 safety_floor = torch.finfo(probabilities.dtype).tiny
 per_example = torch.where(
  active,
  -preferred_mass.clamp_min(safety_floor).log() * endpoint_weights.to(logits.device)[labels],
  torch.zeros_like(preferred_mass),
 )
 return per_example.mean()


def endpoint_preference_rps_loss(
 logits: torch.Tensor, labels: torch.Tensor, endpoint_weights: torch.Tensor, lambda_: float, rho: float = 0.5
) -> torch.Tensor:
 """RPS plus fixed-rho true-endpoint-preference correction."""
 if lambda_ < 0:
  raise ValueError("lambda_ must be non-negative")
 return rps_loss(logits, labels) + lambda_ * endpoint_preference_loss(logits, labels, endpoint_weights, rho)


class CoralHead(nn.Module):
 def __init__(self,features:int,num_classes:int): super().__init__();self.weight=nn.Linear(features,1,bias=False);self.bias=nn.Parameter(torch.zeros(num_classes-1))
 def forward(self,x): return self.weight(x)+self.bias
