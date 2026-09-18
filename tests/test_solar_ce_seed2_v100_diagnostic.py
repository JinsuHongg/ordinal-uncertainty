import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_seed2_v100_diagnostic",
    Path("scripts/audit_solar_ce_seed2_v100_localization.py"),
)
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAGNOSTIC)


def test_stage_summary_localizes_first_nonzero_difference():
    reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    stages = {
        "normalized_input": reference.copy(),
        "features": reference + np.array([[0.0, 0.0], [0.0, 1e-4]], dtype=np.float32),
        "logits": reference + 1e-3,
    }
    result = DIAGNOSTIC.summarize_stages(reference, stages)
    assert result["earliest_nonzero_stage"] == "features"
    assert result["stages"]["normalized_input"]["max_abs_error"] == 0.0
    assert np.isclose(result["stages"]["features"]["max_abs_error"], 1e-4)
    assert np.isclose(result["stages"]["logits"]["max_abs_error"], 1e-3)
