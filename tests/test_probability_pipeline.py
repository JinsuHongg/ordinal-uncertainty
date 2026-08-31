import numpy as np

from ordinal_uncertainty.evaluation.probability_pipeline import finalize_probability_evaluation


def test_slace_probability_pipeline_finalizes_artifacts(tmp_path):
    labels = np.array([0, 1, 2, 3, 4])
    logits = np.array([
        [4, 0, 0, 0, 0],
        [2, 3, 0, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 3, 2, 0],
        [2, 0, 0, 0, 1],
    ], dtype=np.float32)
    messages = []
    result = finalize_probability_evaluation(labels, logits, tmp_path / "evaluation", messages.append)
    assert result["probabilities"].shape == (5, 5)
    assert np.allclose(result["probabilities"].sum(1), 1)
    for relative in ("predictions.csv", "metrics.json", "classwise_metrics.csv", "risk_coverage.csv"):
        assert (tmp_path / "evaluation" / relative).is_file()
    assert messages[-1] == "STAGE 12: artifact finalization complete"
