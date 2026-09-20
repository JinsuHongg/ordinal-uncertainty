import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_seed2_replacement_n",
    Path("scripts/run_solar_ce_seed2_replacement_n.py"),
)
N_RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(N_RUN)


def test_natural_expected_counts_match_empirical_frequency_each_epoch():
    labels = np.array([0, 0, 1, 4], dtype=np.int64)
    assert np.array_equal(N_RUN.expected_draw_counts(labels), np.array([200, 100, 0, 0, 100]))


def test_n_source_has_no_balanced_sampler():
    source = Path("scripts/run_solar_ce_seed2_replacement_n.py").read_text()
    assert "natural_batch_indices" in source
    assert "balanced_batch_indices" not in source
