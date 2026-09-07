"""Static guardrails for the predeclared Solar A/B/C/D confirmation."""
from pathlib import Path


def test_phase3_15_is_a_frozen_rps_four_condition_protocol() -> None:
    source = Path("scripts/run_phase3_15_solar_direction_scale.py").read_text()
    assert "phase3_9_mechanism_audit/features/rps" in source
    assert "selected_checkpoint.pt" in source
    assert "ALPHA = 0.50" in source
    assert "EPOCHS = 100" in source and "BATCH_SIZE = 64" in source
    assert "DirectionOnlyLinear(w, b, w.norm(dim=1))" in source
    assert "DirectionOnlyLinear(w, b, target_norms)" in source
    assert "(1 - ALPHA) * w.norm(dim=1) + ALPHA * balanced_weight.norm(dim=1)" in source
    assert "balanced_batch_indices" in source
    assert "CUDA is mandatory" in source


def test_phase3_15_protects_split_roles_and_no_overwrite() -> None:
    source = Path("scripts/run_phase3_15_solar_direction_scale.py").read_text()
    assert "refusing to overwrite" in source
    assert '"head_fit_split": "train only"' in source
    assert "single predeclared archived confirmatory evaluation only" in source
    assert '"validation_selection": "none"' in source
    assert '"rop": False' in source
    assert '"backbone_retrained": False' in source
