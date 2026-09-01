"""Small, frozen-feature linear-head controls for representation audits.

These helpers intentionally do not own a backbone or any image preprocessing.
They operate only on pre-extracted training features, so a head intervention
cannot change the representation being audited.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ordinal_uncertainty.models.ordinal import rps_loss


def freeze_backbone(backbone: nn.Module) -> None:
    """Disable gradients for every backbone parameter."""
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)


def class_priors(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return strictly positive empirical class priors from training labels."""
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(values, minlength=num_classes)
    if values.size == 0 or counts.size != num_classes or np.any(counts == 0):
        raise ValueError("training labels must contain every class")
    return counts.astype(np.float64) / values.size


def inverse_frequency_weights(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return inverse-frequency class weights, normalized to mean one."""
    priors = class_priors(labels, num_classes)
    weights = 1.0 / priors
    return weights / weights.mean()


def balanced_ce_loss(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Training-only inverse-frequency weighted cross entropy."""
    return F.cross_entropy(logits, labels, weight=weights.to(logits.device, logits.dtype))


def logit_adjusted_ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, priors: torch.Tensor, tau: float = 1.0
) -> torch.Tensor:
    """Menon et al. training-time logit-adjusted CE: CE(z + tau log(pi), y)."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    priors = priors.to(logits.device, logits.dtype)
    if priors.ndim != 1 or priors.numel() != logits.shape[1] or bool((priors <= 0).any()):
        raise ValueError("priors must be positive and match the class dimension")
    return F.cross_entropy(logits + tau * priors.log(), labels)


def head_loss(
    objective: str,
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    priors: torch.Tensor | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """Return the declared training objective for a linear-head control."""
    if objective == "ce":
        return F.cross_entropy(logits, labels)
    if objective == "rps":
        return rps_loss(logits, labels)
    if objective == "balanced_ce":
        if class_weights is None:
            raise ValueError("balanced_ce requires class_weights")
        return balanced_ce_loss(logits, labels, class_weights)
    if objective == "logit_adjusted":
        if priors is None:
            raise ValueError("logit_adjusted requires training priors")
        return logit_adjusted_ce_loss(logits, labels, priors, tau)
    raise ValueError(f"unknown frozen-head objective: {objective}")


def feature_nearest_strata(true_labels: np.ndarray, nearest_classes: np.ndarray, true_class: int = 4) -> np.ndarray:
    """Return ``feature_nearest_true``/``feature_nearest_other`` for one class."""
    labels = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    nearest = np.asarray(nearest_classes, dtype=np.int64).reshape(-1)
    if labels.shape != nearest.shape:
        raise ValueError("labels and nearest_classes must align")
    strata = np.full(labels.shape, "not_target", dtype=object)
    target = labels == true_class
    strata[target & (nearest == true_class)] = "feature_nearest_true"
    strata[target & (nearest != true_class)] = "feature_nearest_other"
    return strata
