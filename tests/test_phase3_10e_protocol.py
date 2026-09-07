"""Static guardrails for the single predeclared Phase 3.10E condition."""
from pathlib import Path

def test_phase3_10e_is_fixed_controlled_scale_rop_interaction() -> None:
    source = (Path(__file__).parents[1] / "scripts/run_phase3_10e_controlled_scale_rop.py").read_text()
    assert "ALPHA = 0.5" in source
    assert "LAMBDA = 1.0" in source
    assert "EPOCHS = 100" in source
    assert "weight_decay=0.0" in source
    assert "natural_batch_indices" in source
    assert "balanced_batch_indices" in source
    assert "risk_order_preservation_loss" in source
    assert "detached_l1_bayes_risk" in source
    assert ".5 * original_weight.norm(dim=1) + .5 * b_weight.norm(dim=1)" in source
