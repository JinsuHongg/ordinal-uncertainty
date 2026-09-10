from pathlib import Path


def test_phase3_19_protocol_is_frozen_training_only_factorial():
    source = Path("scripts/run_phase3_19_sampling_objective.py").read_text()
    assert '"E_natural_ce"' in source
    assert '"F_balanced_rps"' in source
    assert '"G_natural_rps"' in source
    assert "DirectionOnlyLinear(weight, bias)" in source
    assert "natural_batch_indices" in source
    assert "balanced_batch_indices" in source
    assert "rps_loss" in source
    assert "validation_or_test_loaded\": False" in source
