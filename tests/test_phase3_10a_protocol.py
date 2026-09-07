"""Protocol-level leakage guards for the training-only Phase 3.10A script."""
from pathlib import Path


SCRIPT = Path("scripts/run_phase3_10a_rop_falsification.py")


def test_phase3_10a_has_no_dataset_loader_or_validation_test_input_path():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "retinamnist_loaders" not in source
    assert "RetinaMNIST(" not in source
    assert 'name.startswith(("val_", "test_"))' in source
    assert 'required = ("train_sample_id", "train_labels", "train_features", "train_logits", "train_probabilities")' in source


def test_phase3_10a_uses_only_train_feature_archive_and_canonical_checkpoint():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "phase3_3_representation_audit_replay_verified/rps/seed_0/features.npz" in source
    assert "phase2_model_comparison/rps/seed_0_artifact_complete/best_checkpoint.pt" in source
    assert "stratified_five_fold_assignments(labels, SEED)" in source
