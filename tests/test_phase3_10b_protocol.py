"""Static guards that keep Phase 3.10B a training-only, read-only audit."""
from pathlib import Path


SCRIPT = Path("scripts/run_phase3_10b_head_bias_audit.py")


def test_phase3_10b_uses_saved_oof_artifacts_without_data_loaders_or_training():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "RetinaMNIST(" not in source
    assert "retinamnist_loaders" not in source
    assert "optimizer" not in source.lower()
    assert "backward(" not in source
    assert "fold_checkpoints" in source
    assert "train_rps_features.npz" in source


def test_phase3_10b_has_explicit_alignment_and_logit_identity_checks():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "features.shape != (1080, 512)" in source
    assert "np.array_equal(ids, np.arange(1080))" in source
    assert "assert np.allclose(check,logits[held])" in source
