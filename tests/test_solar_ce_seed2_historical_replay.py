import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_seed2_historical_replay",
    Path("scripts/replay_solar_ce_seed2_historical_features.py"),
)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def test_identity_gate_requires_numeric_and_decision_agreement():
    archived_logits = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    archived_probabilities = REPLAY.probabilities(archived_logits)
    result = REPLAY.full_identity_gate(
        sample_ids=np.array([10, 11]),
        labels=np.array([0, 1]),
        logits=archived_logits,
        archived_ids=np.array([10, 11]),
        archived_labels=np.array([0, 1]),
        archived_logits=archived_logits,
        archived_probabilities=archived_probabilities,
        archived_l1=np.array([0, 1]),
    )
    assert result["status"] == "PASS"
    assert result["mode_difference_count"] == 0
    assert result["exact_l1_difference_count"] == 0


def test_v100_requirement_is_explicit_in_replay_source():
    source = Path("scripts/replay_solar_ce_seed2_historical_features.py").read_text()
    assert "V100" in source
    assert "nvidia-smi -L" in source
    assert "seed_2_historical_replay_tmp" in source
