from pathlib import Path
def test_phase3_20a_grid_and_constraints_are_fixed():
    source=Path('scripts/run_phase3_20a_imbalance_severity.py').read_text()
    assert 'N4S = (66, 50, 33, 16, 8)' in source
    assert 'SEEDS = range(5)' in source
    assert 'RandomHorizontalFlip' in source
    assert 'validation_or_test_in_training' in source
    assert 'class_weights' not in source
