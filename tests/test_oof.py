import numpy as np
import torch

from ordinal_uncertainty.evaluation.oof import balanced_batch_indices, natural_batch_indices, stratified_five_fold_assignments


def test_five_fold_assignments_are_training_only_complete_and_stratified():
    labels = np.repeat(np.arange(5), [486, 128, 206, 194, 66])
    folds = stratified_five_fold_assignments(labels, seed=0)
    assert folds.shape == (1080,)
    assert set(folds) == set(range(5))
    assert np.all((folds >= 0) & (folds < 5))
    for class_index in range(5):
        counts = np.bincount(folds[labels == class_index], minlength=5)
        assert counts.max() - counts.min() <= 1


def test_balanced_and_natural_batches_are_separate_sampling_schemes():
    labels = torch.tensor([0] * 12 + [1] * 3 + [2] * 2)
    generator = torch.Generator().manual_seed(0)
    balanced = balanced_batch_indices(labels, 100, generator)
    natural = natural_batch_indices(len(labels), 8, generator)
    balanced_counts = torch.bincount(labels[balanced], minlength=3)
    assert balanced_counts.max() - balanced_counts.min() < 20
    assert torch.equal(torch.sort(torch.cat(natural)).values, torch.arange(len(labels)))
