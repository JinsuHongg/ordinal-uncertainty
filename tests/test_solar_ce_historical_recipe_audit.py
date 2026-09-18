import importlib.util
from pathlib import Path

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "solar_ce_historical_recipe_audit",
    Path("scripts/audit_solar_ce_historical_recipe.py"),
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def test_identity_metrics_accepts_exact_archived_logits():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    probabilities = AUDIT.softmax(logits)
    l1 = np.array([0, 1], dtype=np.int64)
    result = AUDIT.identity_metrics(logits, logits, probabilities, l1)
    assert result["status"] == "PASS"
    assert result["logit_max_abs_error"] == 0.0
    assert result["probability_max_abs_error"] == 0.0
    assert result["exact_l1_difference_count"] == 0


def test_full_population_identity_requires_exact_alignment():
    logits = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    probabilities = AUDIT.softmax(logits)
    result = AUDIT.full_population_identity(
        sample_ids=np.array([10, 11]), labels=np.array([0, 1]), logits=logits,
        archived_ids=np.array([10, 11]), archived_labels=np.array([0, 1]),
        archived_logits=logits, archived_probabilities=probabilities,
        archived_l1=np.array([0, 1]),
    )
    assert result["status"] == "PASS"
    assert result["sample_ids_match"] is True
    assert result["labels_match"] is True
