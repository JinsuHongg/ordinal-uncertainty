"""SLACE (Nachmani et al., AAAI 2025) Softmax loss components."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def inverse_frequency_weights(counts: torch.Tensor) -> torch.Tensor:
    """Inverse training-frequency weights normalized to mean one."""
    counts = torch.as_tensor(counts, dtype=torch.float)
    if counts.ndim != 1 or bool((counts <= 0).any()): raise ValueError("counts must be positive 1-D")
    raw = counts.reciprocal()
    return raw / raw.mean()


def slace_matrices(counts: torch.Tensor, alpha: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return proximity, per-target soft labels, and dominance matrices from Eq. 7--11."""
    counts = torch.as_tensor(counts, dtype=torch.float)
    if alpha <= 0: raise ValueError("alpha must be positive")
    k, total = len(counts), counts.sum()
    proximity = torch.empty((k, k), dtype=torch.float)
    for predicted in range(k):
        for true in range(k):
            if predicted < true:
                mass = counts[predicted] / 2 + counts[predicted + 1:true + 1].sum()
            elif predicted > true:
                mass = counts[predicted] / 2 + counts[true:predicted].sum()
            else:
                mass = counts[predicted] / 2
            proximity[predicted, true] = -torch.log(mass / total)
    distance = proximity.max(dim=0).values[None, :] - proximity
    soft_labels = torch.softmax(-alpha * distance.T, dim=1)  # [true, predicted]
    dominance = proximity.T[:, :, None] <= proximity.T[:, None, :]  # [true, i, j]
    return proximity, soft_labels, dominance


def slace_loss(logits: torch.Tensor, labels: torch.Tensor, soft_labels: torch.Tensor, dominance: torch.Tensor) -> torch.Tensor:
    """Published SLACE: Eq. 11 over standard Softmax class probabilities."""
    probabilities = torch.softmax(logits, dim=1)
    selected_dominance = dominance.to(logits.device)[labels]
    accumulated = torch.bmm(selected_dominance.float(), probabilities.unsqueeze(2)).squeeze(2)
    targets = soft_labels.to(logits.device)[labels]
    return -(targets * accumulated.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(1).mean()


def weighted_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=weights.to(logits.device))
