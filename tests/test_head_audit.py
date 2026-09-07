import numpy as np

from ordinal_uncertainty.evaluation.head_audit import common_norm_weights, cosine_alignment, linear_logits, margin_decomposition, swapped_parameters


def test_logit_and_margin_decomposition_identity():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    weight = np.array([[1.0, 0.0], [0.0, 1.0]])
    bias = np.array([.5, -.5])
    term, logits = linear_logits(features, weight, bias)
    feature_margin, bias_margin, margin = margin_decomposition(term, bias, 0, 1)
    assert np.allclose(logits, term + bias)
    assert np.allclose(margin, logits[:, 0] - logits[:, 1])
    assert bias_margin == 1.0
    assert np.allclose(feature_margin, [-1, -1])


def test_alignment_and_counterfactual_swaps():
    features = np.array([[1.0, 0.0]])
    original_w = np.array([[1.0, 0.0], [0.0, 1.0]])
    original_b = np.array([1.0, 2.0])
    balanced_w = np.array([[2.0, 0.0], [0.0, 3.0]])
    balanced_b = np.array([3.0, 4.0])
    assert np.allclose(cosine_alignment(features, original_w), [[1.0, 0.0]])
    weight, bias = swapped_parameters(original_w, original_b, balanced_w, balanced_b, "original_weights_balanced_biases")
    assert np.array_equal(weight, original_w) and np.array_equal(bias, balanced_b)
    weight, bias = swapped_parameters(original_w, original_b, balanced_w, balanced_b, "balanced_weights_original_biases")
    assert np.array_equal(weight, balanced_w) and np.array_equal(bias, original_b)
    assert np.allclose(np.linalg.norm(common_norm_weights(balanced_w, 2.5), axis=1), 2.5)
