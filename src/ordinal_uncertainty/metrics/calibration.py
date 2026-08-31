"""Validation-only scalar temperature scaling utilities."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def temperature_probabilities(logits: torch.Tensor, temperature: float | torch.Tensor) -> torch.Tensor:
    """Return Softmax probabilities after positive scalar temperature scaling."""
    temperature_tensor = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
    if temperature_tensor.numel() != 1 or not bool(torch.isfinite(temperature_tensor)) or float(temperature_tensor) <= 0:
        raise ValueError("temperature must be one finite positive scalar")
    return torch.softmax(logits / temperature_tensor, dim=1)


def fit_temperature(validation_logits: torch.Tensor, validation_labels: torch.Tensor, max_iter: int = 100) -> tuple[float, float, float]:
    """Fit a scalar temperature by minimizing validation NLL only.

    Returns ``(temperature, uncalibrated_validation_nll, calibrated_validation_nll)``.
    Test labels are intentionally not accepted by this interface.
    """
    logits = validation_logits.detach().float().cpu()
    labels = validation_labels.detach().long().reshape(-1).cpu()
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("validation logits and labels have incompatible shapes")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("validation logits must be finite")
    raw_nll = float(F.cross_entropy(logits, labels).item())
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.25, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().item())
    calibrated_nll = float(F.cross_entropy(logits / temperature, labels).item())
    if not (torch.isfinite(torch.tensor(temperature)) and temperature > 0 and torch.isfinite(torch.tensor(calibrated_nll))):
        raise RuntimeError("temperature optimization returned a non-finite result")
    return temperature, raw_nll, calibrated_nll
