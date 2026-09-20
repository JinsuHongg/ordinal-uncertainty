import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_natural_analysis",
    Path("scripts/analyze_solar_ce_natural_direction.py"),
)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


def test_followup_uses_replacement_seed2_and_frozen_four_seed_order():
    assert ANALYZER.SEEDS == (1, 2, 3, 4)
    assert ANALYZER.source_root(2) == ANALYZER.REPLACEMENT_ROOT
    assert ANALYZER.source_root(1) == ANALYZER.N_ROOT / "seed_1"
    assert ANALYZER.source_root(3) == ANALYZER.N_ROOT / "seed_3"
    assert ANALYZER.source_root(4) == ANALYZER.N_ROOT / "seed_4"


def test_metric_uses_exact_l1_error_and_endpoint_mass_per_true_class():
    y = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    decision = np.array([0, 1, 2, 3, 3], dtype=np.int64)
    probability = np.eye(5, dtype=np.float64)
    result = ANALYZER.metric(y, decision, probability)
    assert result["global_mae"] == 0.2
    assert result["mae_4"] == 1.0
    assert result["severe_4"] == 0.0
    assert result["p_end_4"] == 1.0
