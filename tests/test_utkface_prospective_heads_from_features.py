import importlib.util
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location(
    "utkface_heads_from_features",
    Path("scripts/run_utkface_prospective_heads_from_features.py"),
)
HEADS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HEADS)


def test_exact_l1_condition_metrics_include_endpoint_macro_and_severe_error():
    labels = np.asarray([0, 1, 4, 4], dtype=np.int64)
    logits = np.asarray([
        [5.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 5.0, 0.0],
        [5.0, 0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    metrics, arrays = HEADS.condition_metrics(labels, logits)

    assert arrays["l1_bayes_decision"].tolist() == [0, 1, 3, 0]
    assert metrics["global_l1_mae"] == 1.25
    assert metrics["macro_l1_mae"] == np.mean([0.0, 0.0, 2.5])
    assert metrics["severe_error_rate_l1"] == 0.25
    assert metrics["endpoint_4"]["support"] == 2
    assert metrics["endpoint_4"]["routing"] == [1, 0, 0, 1, 0]


def test_constraint_audit_preserves_norms_biases_and_only_trains_directions():
    weight = torch.tensor([[3.0, 4.0], [4.0, 3.0]], dtype=torch.float32)
    bias = torch.tensor([0.1, -0.2], dtype=torch.float32)
    head = HEADS.DirectionOnlyLinear(weight, bias)

    audit = HEADS.constraint_audit(head, weight, bias)

    assert audit["max_norm_error"] <= 1e-6
    assert audit["max_bias_error"] == 0.0
    assert audit["trainable_parameter_names"] == ["direction"]


def test_feature_archive_contract_rejects_non_v100_or_unready_unit():
    metadata = {
        "feature_dimension": 512,
        "split_counts": {"train": 14224, "validation": 2371, "test": 2371},
        "test_class_4_support": 67,
        "replay": {"train": {"mode_differences": 0, "l1_differences": 0}, "validation": {"mode_differences": 0, "l1_differences": 0}, "test": {"mode_differences": 0, "l1_differences": 0}},
    }
    run = {"readiness": "READY FOR HEAD ADAPTATION", "execution": {"gpu": "Tesla V100-SXM2-32GB"}}

    HEADS.validate_feature_record(run, metadata)

    with np.testing.assert_raises_regex(RuntimeError, "V100"):
        HEADS.validate_feature_record({**run, "execution": {"gpu": "NVIDIA A30"}}, metadata)
