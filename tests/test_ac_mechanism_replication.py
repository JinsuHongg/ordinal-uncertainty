import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "run_ac_mechanism_replication",
    Path("scripts/run_ac_mechanism_replication.py"),
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_geometry_uses_fitting_centroids_and_frozen_margin_definitions():
    fit_x = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    fit_y = np.arange(5, dtype=np.int64)
    values = RUNNER.geometry(fit_x, fit_y, np.array([[3.8]], dtype=np.float32))
    assert values["nearest_centroid"].tolist() == [4]
    assert np.allclose(values["endpoint_adj_margin"], [0.6])  # d_3 - d_4
    assert np.allclose(values["generic_centroid_margin"], [0.6])  # d_(2) - d_(1)
