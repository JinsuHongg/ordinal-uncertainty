from scripts.audit_retina_rps_replay import replay_status


def test_replay_status_requires_one_exact_transform():
    passed = {"ids_match": True, "labels_match": True, "max_abs_logit_error": 1e-4}
    failed = {"ids_match": True, "labels_match": True, "max_abs_logit_error": 1e-3}
    assert replay_status({"normalize_half": passed, "to_tensor": failed}) == "COMPATIBLE"
    assert replay_status({"normalize_half": passed, "to_tensor": passed}) == "AMBIGUOUS"
    assert replay_status({"normalize_half": failed, "to_tensor": failed}) == "INCOMPATIBLE"
