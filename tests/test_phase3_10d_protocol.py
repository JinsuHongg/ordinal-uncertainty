from pathlib import Path


def test_controlled_scale_grid_and_constraints_are_frozen():
    source = Path('scripts/run_phase3_10d_controlled_scale.py').read_text()
    assert "ALPHAS=(.25,.5,.75)" in source
    assert "DirectionOnlyLinear(ow,ob,norm)" in source
    assert "weight_decay=0.0" in source
    assert "if er>1e-6:raise RuntimeError" in source
    assert "RetinaMNIST(" not in source
    assert "retinamnist_loaders" not in source
