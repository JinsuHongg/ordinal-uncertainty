import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_seed2_replacement_analysis",
    Path("scripts/analyze_solar_ce_seed2_replacement.py"),
)
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)


def test_metric_reports_exact_l1_global_macro_and_severe_error():
    labels = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    decisions = np.array([0, 1, 4, 3, 2], dtype=np.int64)
    probability = np.eye(5, dtype=float)
    result = ANALYSIS.metric(labels, decisions, probability)
    assert result["global_mae"] == 0.8
    assert result["macro_mae"] == 0.8
    assert result["global_severe"] == 0.4


def test_analysis_source_marks_missing_cross_seed_ce_n_as_unavailable():
    source = Path("scripts/analyze_solar_ce_seed2_replacement.py").read_text()
    assert "UNAVAILABLE" in source
    assert "replacement_seed2" in source
