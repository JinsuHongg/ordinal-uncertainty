import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_seed2_replacement_c",
    Path("scripts/run_solar_ce_seed2_replacement_c.py"),
)
C_RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C_RUN)


def test_c_metrics_use_exact_l1_endpoint_difference():
    labels = np.array([4, 4, 3], dtype=np.int64)
    a_l1 = np.array([2, 3, 3], dtype=np.int64)
    c_l1 = np.array([4, 3, 3], dtype=np.int64)
    result = C_RUN.endpoint_h1(labels, a_l1, c_l1)
    assert result["endpoint_mae_A"] == 1.5
    assert result["endpoint_mae_C"] == 0.5
    assert result["delta_H1_C_minus_A"] == -1.0


def test_c_source_requires_fixed_direction_only_protocol():
    source = Path("scripts/run_solar_ce_seed2_replacement_c.py").read_text()
    assert "fit_direction_head" in source
    assert "max_norm_error" in source
    assert "max_bias_error" in source


def test_c_step_count_matches_fixed_balanced_epoch_budget():
    assert C_RUN.head_step_count(45_047) == 70_400
