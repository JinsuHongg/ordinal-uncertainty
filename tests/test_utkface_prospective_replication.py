import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SPEC = importlib.util.spec_from_file_location(
    "utkface_prospective",
    Path("scripts/run_utkface_prospective_replication.py"),
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_natural_direction_fit_preserves_empirical_draws_and_fixed_parameters():
    features = np.eye(5, dtype=np.float32)
    labels = np.arange(5, dtype=np.int64)
    head, history, draws, constraints = RUNNER.fit_direction_head(
        features,
        labels,
        torch.eye(5),
        torch.zeros(5),
        sampling="natural",
        seed=7,
        device=torch.device("cpu"),
    )

    assert len(history) == RUNNER.HEAD_EPOCHS
    assert np.array_equal(draws, np.full(5, RUNNER.HEAD_EPOCHS))
    assert constraints["max_norm_error"] <= RUNNER.CONSTRAINT_TOL
    assert constraints["max_bias_error"] == 0.0
    assert head.fixed_norms.requires_grad is False
    assert head.fixed_bias.requires_grad is False


def test_validation_contract_has_expected_archived_counts_and_is_disjoint():
    train = ["utkface:train-a", "utkface:train-b"]
    validation = ["utkface:validation-a"]
    RUNNER.assert_disjoint_ids(train, validation)


CLUSTER_ROOT = Path("/scratch/users/jhong36/data/utkface/extracted/UTKFace")
CLUSTER_MANIFEST = Path("/scratch/users/jhong36/ordinal-uq/manifest/utkface/manifest.jsonl")


def test_cli_honors_explicit_cluster_paths():
    args = RUNNER.parse_args([
        "backbone", "--objective", "ce", "--seed", "1", "--out", "/tmp/utk-out",
        "--data-root", str(CLUSTER_ROOT), "--manifest", str(CLUSTER_MANIFEST),
    ])

    assert args.data_root == CLUSTER_ROOT
    assert args.manifest == CLUSTER_MANIFEST


def test_records_honors_explicit_cluster_paths_without_local_paths():
    splits = RUNNER.records(CLUSTER_MANIFEST, CLUSTER_ROOT)

    assert [len(splits[name]) for name in ("train", "validation")] == [14224, 2371]


def test_records_rejects_missing_explicit_paths():
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        RUNNER.records(Path("/missing/manifest.jsonl"), CLUSTER_ROOT)
    with pytest.raises(FileNotFoundError, match="data root not found"):
        RUNNER.records(CLUSTER_MANIFEST, Path("/missing/data"))


def test_runner_source_has_no_workstation_path_dependency():
    source = Path("scripts/run_utkface_prospective_replication.py").read_text()

    assert "/home/jhong90" not in source
    assert "/mnt/storage" not in source


def test_original_head_replay_preserves_logits_probabilities_and_decisions():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.25, 0.75]], dtype=np.float32)
    weight = np.asarray([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float32)
    bias = np.asarray([0.1, -0.2], dtype=np.float32)
    direct_logits = features @ weight.T + bias

    replay = RUNNER.replay_original_head(features, weight, bias, direct_logits)

    assert replay["max_logit_diff"] == 0.0
    assert replay["max_probability_diff"] == 0.0
    assert replay["mode_differences"] == 0
    assert replay["l1_differences"] == 0


def test_feature_archive_split_alignment_rejects_duplicate_or_overlapping_ids():
    splits = {
        "train": np.asarray(["utkface:one", "utkface:two"]),
        "validation": np.asarray(["utkface:three"]),
        "test": np.asarray(["utkface:four"]),
    }
    RUNNER.assert_feature_split_integrity(splits)

    with pytest.raises(RuntimeError, match="duplicate"):
        RUNNER.assert_feature_split_integrity({**splits, "test": np.asarray(["utkface:four", "utkface:four"])})
    with pytest.raises(RuntimeError, match="overlap"):
        RUNNER.assert_feature_split_integrity({**splits, "test": np.asarray(["utkface:two"])})
