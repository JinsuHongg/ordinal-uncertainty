import numpy as np

from scripts.run_phase4_1_geometric_mechanism import analysis, unit_rows


def test_angular_analysis_tracks_recovered_margin_change() -> None:
    features = np.zeros((3, 512)); features[:, 0] = 1
    labels = np.array([4, 4, 0]); ids = np.array([1, 2, 3])
    a = np.zeros((5, 512)); c = np.zeros((5, 512))
    a[:, 2] = 1; c[:, 2] = 1
    a[4, 1] = 1; a[3, 0] = 1
    c[4, 0] = 1; c[3, 1] = 1
    summary, rows = analysis(features, labels, ids, np.array([3, 3, 0]), np.array([4, 3, 0]), unit_rows(a), unit_rows(c), "synthetic")
    assert summary["groups"]["recovered"]["count"] == 1
    assert rows[0]["delta_angular_margin"] > 0
    assert rows[1]["delta_angular_margin"] > 0
