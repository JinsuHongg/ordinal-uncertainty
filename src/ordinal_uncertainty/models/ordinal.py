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


def adjacent_classes(label: int, num_classes: int) -> tuple[int, ...]:
 """Return ordinally adjacent valid class indices."""
 if not 0 <= label < num_classes or num_classes < 2:
  raise ValueError("label/num_classes are outside the ordinal range")
 return tuple(index for index in (label - 1, label + 1) if 0 <= index < num_classes)


def l1_bayes_risk(logits: torch.Tensor) -> torch.Tensor:
 """Differentiable L1 Bayes risk from categorical logits, one value per row."""
 if logits.ndim != 2 or logits.shape[1] < 2:
  raise ValueError("logits must be [batch, classes >= 2]")
 probabilities = F.softmax(logits, dim=1)
 classes = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
 actions = classes[None, :, None]
 outcomes = classes[None, None, :]
 expected = (probabilities[:, None, :] * (outcomes - actions).abs()).sum(dim=2)
 return expected.min(dim=1).values


def rg_acr_loss(
 logits: torch.Tensor,
 features: torch.Tensor,
 labels: torch.Tensor,
 margin: float = 0.05,
 risk_cap: float = 2.0,
 epsilon: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
 """Risk-gated adjacent-centroid ranking with detached L1 Bayes-risk weights.

 Centroids are batch-derived.  Each anchor uses a leave-one-out own-class
 centroid and every present adjacent-class centroid.  The returned diagnostics
 are detached so callers can accumulate participation statistics safely.
 """
 if logits.ndim != 2 or features.ndim != 2 or labels.ndim != 1:
  raise ValueError("logits/features/labels must be rank 2/2/1")
 if logits.shape[0] != features.shape[0] or labels.numel() != features.shape[0]:
  raise ValueError("batch dimensions must agree")
 if logits.shape[1] < 2 or margin < 0 or risk_cap <= 0 or epsilon <= 0:
  raise ValueError("invalid RG-ACR configuration")
 num_classes = logits.shape[1]
 if bool((labels < 0).any()) or bool((labels >= num_classes).any()):
  raise ValueError("labels are outside the ordinal class range")
 z = F.normalize(features, p=2, dim=1, eps=epsilon)
 counts = torch.bincount(labels, minlength=num_classes)
 class_sums = torch.zeros(num_classes, z.shape[1], dtype=z.dtype, device=z.device)
 class_sums.index_add_(0, labels, z)
 raw_risk = l1_bayes_risk(logits).detach()
 per_anchor: list[torch.Tensor] = []
 valid_indices: list[int] = []
 adjacent_terms = torch.zeros(num_classes, dtype=torch.long, device=z.device)
 for index in range(z.shape[0]):
  label = int(labels[index].item())
  if int(counts[label].item()) < 2:
   continue
  present = tuple(adjacent for adjacent in adjacent_classes(label, num_classes) if int(counts[adjacent].item()) >= 1)
  if not present:
   continue
  own = F.normalize((class_sums[label] - z[index]).unsqueeze(0), p=2, dim=1, eps=epsilon).squeeze(0)
  distances = []
  for adjacent in present:
   other = F.normalize(class_sums[adjacent].unsqueeze(0), p=2, dim=1, eps=epsilon).squeeze(0)
   distances.append(F.relu(margin + (1.0 - (z[index] * own).sum()) - (1.0 - (z[index] * other).sum())))
   adjacent_terms[label] += 1
  per_anchor.append(torch.stack(distances).mean())
  valid_indices.append(index)
 if not valid_indices:
  zero = features.sum() * 0.0
  return zero, {
   "valid_mask": torch.zeros(features.shape[0], dtype=torch.bool, device=features.device),
   "weights": torch.zeros(features.shape[0], dtype=features.dtype, device=features.device),
   "per_anchor": torch.zeros(features.shape[0], dtype=features.dtype, device=features.device),
   "class_counts": counts.detach(),
   "adjacent_terms": adjacent_terms.detach(),
  }
 valid = torch.tensor(valid_indices, dtype=torch.long, device=features.device)
 per_anchor_valid = torch.stack(per_anchor)
 # Phase 3.5 froze the denominator to the mean detached risk over the full batch.
 weights_valid = (raw_risk[valid] / (raw_risk.mean() + epsilon)).clamp(max=risk_cap)
 loss = (weights_valid * per_anchor_valid).sum() / (weights_valid.sum() + epsilon)
 valid_mask = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
 valid_mask[valid] = True
 weights = torch.zeros(features.shape[0], dtype=features.dtype, device=features.device)
 weights[valid] = weights_valid.detach()
 per_anchor_full = torch.zeros(features.shape[0], dtype=features.dtype, device=features.device)
 per_anchor_full[valid] = per_anchor_valid.detach()
 return loss, {
  "valid_mask": valid_mask.detach(),
  "weights": weights,
  "per_anchor": per_anchor_full,
  "class_counts": counts.detach(),
  "adjacent_terms": adjacent_terms.detach(),
 }


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
