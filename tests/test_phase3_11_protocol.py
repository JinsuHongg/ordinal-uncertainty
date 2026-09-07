"""Guardrails for the one-shot frozen-candidate validation gate."""
from pathlib import Path


def test_phase3_11_keeps_fitting_and_validation_separate() -> None:
    source = (Path(__file__).parents[1] / "scripts/run_phase3_11_frozen_candidate_validation.py").read_text()
    assert 'choices=("fit", "evaluate")' in source
    assert "EPOCHS, BATCH, ALPHA, LAMBDA, SEED = 100, 64, .5, 1.0, 0" in source
    assert "if OUT.exists(): raise FileExistsError" in source
    assert "all full-training head states must exist before validation is loaded" in source
    assert "natural_batch_indices" in source
    assert "balanced_batch_indices" in source
    assert "risk_order_preservation_loss" in source
    assert "test_loaded\": False" in source
