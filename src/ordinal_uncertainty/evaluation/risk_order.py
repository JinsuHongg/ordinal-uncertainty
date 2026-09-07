"""Risk-order preservation primitives for frozen ordinal linear heads."""
from __future__ import annotations

import torch


def discrete_l1_action_risks(probabilities: torch.Tensor) -> torch.Tensor:
    """Return differentiable discrete L1 action risks, one column per action."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (batch, classes)")
    classes = torch.arange(probabilities.shape[1], device=probabilities.device, dtype=probabilities.dtype)
    return (probabilities[:, None, :] * (classes[None, None, :] - classes[None, :, None]).abs()).sum(dim=2)


def detached_l1_bayes_risk(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return L1 Bayes risk with a detached discrete minimizing action.

    The selected action is intentionally non-differentiable; gradients flow
    through the corresponding action-risk entry only.
    """
    action_risks = discrete_l1_action_risks(probabilities)
    action = action_risks.detach().argmin(dim=1)
    risk = action_risks.gather(1, action[:, None]).squeeze(1)
    return risk, action


def risk_order_preservation_loss(
    student_risk: torch.Tensor,
    teacher_risk: torch.Tensor,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Zero-margin, teacher-risk-weighted squared hinge on unordered pairs.

    Correctly ordered pairs receive exactly zero loss. Teacher ties are omitted.
    """
    if student_risk.ndim != 1 or teacher_risk.ndim != 1 or student_risk.shape != teacher_risk.shape:
        raise ValueError("student_risk and teacher_risk must be aligned vectors")
    if student_risk.numel() < 2:
        zero = student_risk.sum() * 0.0
        return zero, {"valid_pair_count": zero.detach(), "violation_fraction": zero.detach(), "weighted_violation_fraction": zero.detach(), "mean_absolute_teacher_difference": zero.detach(), "tied_pair_fraction": zero.detach()}
    i, j = torch.triu_indices(student_risk.numel(), student_risk.numel(), offset=1, device=student_risk.device)
    teacher_delta = teacher_risk[i] - teacher_risk[j]
    weight = teacher_delta.abs()
    valid = weight > 0
    total_pairs = weight.numel()
    if not bool(valid.any()):
        zero = student_risk.sum() * 0.0
        return zero, {"valid_pair_count": torch.zeros((), device=student_risk.device), "violation_fraction": torch.zeros((), device=student_risk.device), "weighted_violation_fraction": torch.zeros((), device=student_risk.device), "mean_absolute_teacher_difference": torch.zeros((), device=student_risk.device), "tied_pair_fraction": torch.ones((), device=student_risk.device)}
    signed_delta = teacher_delta[valid].sign() * (student_risk[i][valid] - student_risk[j][valid])
    violation = torch.relu(-signed_delta)
    valid_weight = weight[valid]
    loss = (valid_weight * violation.square()).sum() / (valid_weight.sum() + epsilon)
    violated = violation > 0
    return loss, {
        "valid_pair_count": valid.sum().to(student_risk.dtype),
        "violation_fraction": violated.to(student_risk.dtype).mean(),
        "weighted_violation_fraction": valid_weight[violated].sum() / (valid_weight.sum() + epsilon),
        "mean_absolute_teacher_difference": valid_weight.mean(),
        "tied_pair_fraction": torch.as_tensor(1.0 - valid.sum().item() / total_pairs, device=student_risk.device, dtype=student_risk.dtype),
    }
