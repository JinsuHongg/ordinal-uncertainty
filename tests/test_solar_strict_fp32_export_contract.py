import importlib.util
from pathlib import Path

import numpy as np
import torch


SPEC = importlib.util.spec_from_file_location(
    "solar_export",
    Path("scripts/export_solar_confirmatory_frozen_features.py"),
)
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def test_a_identity_reports_full_decision_and_numeric_diagnostics(tmp_path):
    features = np.eye(5, dtype=np.float32)
    weight = torch.eye(5)
    bias = torch.zeros(5)
    logits = features.copy()
    probabilities = EXPORT.probabilities(logits)
    l1 = EXPORT.bayes_decisions(probabilities)["l1_bayes_decision"].astype(np.int64)
    path = tmp_path / "archived.npz"
    np.savez_compressed(
        path,
        sample_ids=np.arange(5, dtype=np.int64),
        labels=np.arange(5, dtype=np.int64),
        a_logits=logits,
        a_probabilities=probabilities,
        a_l1=l1,
        a_predictive_mean=probabilities @ np.arange(5),
    )
    result = EXPORT.compare_a_eval(
        {"features": features, "labels": np.arange(5), "sample_ids": np.arange(5)},
        path,
        weight,
        bias,
    )
    assert result["status"] == "PASS"
    assert result["mode_difference_count"] == 0
    assert result["exact_l1_difference_count"] == 0
    assert result["logit_mean_abs_error"] == 0.0


def test_strict_fp32_flags_are_explicit_in_export_source():
    source = Path("scripts/export_solar_confirmatory_frozen_features.py").read_text()
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in source
    assert "torch.backends.cudnn.allow_tf32 = False" in source
    assert 'torch.set_float32_matmul_precision("highest")' in source
