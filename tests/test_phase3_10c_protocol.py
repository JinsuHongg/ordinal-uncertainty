from pathlib import Path


def test_direction_only_protocol_has_required_constraints_and_no_leakage():
    source = Path('scripts/run_phase3_10c_direction_only.py').read_text()
    assert 'DirectionOnlyLinear(ow,ob)' in source
    assert 'weight_decay=0.0' in source
    assert 'if error>1e-6:raise RuntimeError' in source
    assert 'RetinaMNIST(' not in source
    assert 'retinamnist_loaders' not in source
    assert 'train_rps_features.npz' in source
    assert 'balanced_batch_indices' in source
