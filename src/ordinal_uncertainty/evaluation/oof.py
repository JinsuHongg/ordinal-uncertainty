"""Deterministic training-only OOF split and batch-index helpers."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold


def stratified_five_fold_assignments(labels: np.ndarray, seed: int = 0) -> np.ndarray:
    """Assign every supplied training sample to exactly one stratified fold."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    assignments = np.full(labels.shape, -1, dtype=np.int64)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(labels.size), labels)):
        assignments[held_out] = fold
    if (assignments < 0).any():
        raise RuntimeError("incomplete OOF assignment")
    return assignments


def balanced_batch_indices(labels: torch.Tensor, batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Draw a replacement balanced batch using only fitting-fold labels."""
    classes = torch.unique(labels, sorted=True)
    chosen_classes = classes[torch.randint(len(classes), (batch_size,), generator=generator)]
    output = []
    for class_index in chosen_classes:
        candidates = torch.nonzero(labels == class_index, as_tuple=False).squeeze(1)
        output.append(candidates[torch.randint(len(candidates), (1,), generator=generator)])
    return torch.cat(output)


def natural_batch_indices(size: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    """One empirical-distribution epoch, shuffled without rebalancing."""
    return list(torch.randperm(size, generator=generator).split(batch_size))
