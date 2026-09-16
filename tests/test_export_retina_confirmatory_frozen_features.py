import importlib.util
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location(
    "export_retina_confirmatory_frozen_features",
    Path("scripts/export_retina_confirmatory_frozen_features.py"),
)
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


def test_exact_l1_uses_archived_decision_rule():
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.2, 0.2]], dtype=np.float64)
    assert EXPORTER.exact_l1(probabilities).shape == (1,)


def test_a_output_comparison_requires_ids_labels_logits_probabilities_and_l1():
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    bias = torch.zeros(2)
    logits = features @ weight.numpy().T
    probabilities = EXPORTER.softmax(logits)
    archived = {
        "sample_ids": np.asarray([0, 1]),
        "labels": np.asarray([0, 1]),
        "a_logits": logits,
        "a_probabilities": probabilities,
        "a_predictive_mean": probabilities @ np.arange(2),
        "a_l1": EXPORTER.exact_l1(probabilities),
    }
    result = EXPORTER.compare_a_outputs(features, archived["labels"], archived["sample_ids"], archived, {"weight": weight, "bias": bias})
    assert result["status"] == "PASS"
    assert result["exact_l1_match"]


def test_feature_archive_contract_preserves_natural_order_metadata():
    assert EXPORTER.EXPECTED_COUNTS.tolist() == [486, 128, 206, 194, 66]
    assert "features" in Path("scripts/export_retina_confirmatory_frozen_features.py").read_text()
    assert "split_roles" in Path("scripts/export_retina_confirmatory_frozen_features.py").read_text()
