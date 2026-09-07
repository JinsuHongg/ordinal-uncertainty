"""Fixed-norm, fixed-bias linear classifier for causal head audits."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DirectionOnlyLinear(nn.Module):
    """Train classifier directions while preserving supplied row norms and biases."""
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, fixed_norms: torch.Tensor | None = None) -> None:
        super().__init__()
        if weight.ndim != 2 or bias.shape != (weight.shape[0],):
            raise ValueError("invalid linear-head shapes")
        self.direction = nn.Parameter(weight.detach().clone())
        norms = weight.detach().norm(dim=1) if fixed_norms is None else fixed_norms.detach().clone()
        if norms.shape != (weight.shape[0],) or (norms <= 0).any():
            raise ValueError("fixed_norms must be one positive norm per class")
        self.register_buffer("fixed_norms", norms)
        self.register_buffer("fixed_bias", bias.detach().clone())

    def effective_weight(self) -> torch.Tensor:
        return F.normalize(self.direction, dim=1) * self.fixed_norms[:, None]

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.linear(features, self.effective_weight(), self.fixed_bias)

    def max_norm_error(self) -> torch.Tensor:
        return (self.effective_weight().norm(dim=1) - self.fixed_norms).abs().max()
