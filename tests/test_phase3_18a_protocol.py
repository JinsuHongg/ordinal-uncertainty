from pathlib import Path


def test_phase3_18a_is_a_matched_frozen_ce_abc_protocol():
    source = Path("scripts/run_phase3_18a_retinamnist_ce_direction.py").read_text()
    assert "train_features" in source and "train_logits" in source
    assert "DirectionOnlyLinear(original_weight, original_bias)" in source
    assert "weight_decay=1e-4" in source
    assert "weight_decay=0.0" in source
    assert "balanced_batch_indices" in source
    assert "EPOCHS = 100" in source and "BATCH_SIZE = 64" in source
    assert "test_or_validation_loaded\": False" in source
    assert "controlled" not in source.lower()
    assert "l_rop" not in source.lower()
