import importlib.util
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_seed2_replacement_features",
    Path("scripts/export_solar_ce_seed2_replacement_features.py"),
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def test_replacement_a_replay_uses_checkpoint_head_and_preserves_float32():
    features = np.eye(5, dtype=np.float32)
    weight = torch.eye(5, dtype=torch.float32)
    bias = torch.zeros(5, dtype=torch.float32)
    logits = EXPORT.a_logits_from_features(features, weight, bias)
    assert logits.dtype == np.float32
    assert np.array_equal(logits, features)


def test_replacement_feature_contract_has_v100_and_namespace_guards():
    source = Path("scripts/export_solar_ce_seed2_replacement_features.py").read_text()
    assert "V100" in source
    assert "replacement_seed2/ce" in source
    assert "(len(loader.dataset), 512)" in source
