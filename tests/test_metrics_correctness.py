import numpy as np
import torch

from engine import accuracy_topk, _to_prob_matrix


def test_accuracy_topk_counts_are_correct():
    """
    Construct logits and targets so that top-1 and top-2
    correctness can be checked exactly.
    """
    # 3 samples, 4 classes
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0],   # predict class 0
            [0.0, 5.0, 4.0, 0.0],    # predict class 1, second-best 2
            [1.0, 2.0, 3.0, 4.0],    # predict class 3, second-best 2
        ]
    )
    targets = torch.tensor([0, 2, 3])

    # Expected:
    #   top-1 correct: sample 0 (0), sample 2 (3) -> 2
    #   top-2 correct: all samples -> 3
    counts = accuracy_topk(logits, targets, topk=(1, 2))
    assert counts == [2, 3]


def test_to_prob_matrix_from_logits_matches_softmax():
    """
    When given logits, _to_prob_matrix should behave like softmax
    for well-behaved values.
    """
    logits = np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float64)

    # Manual softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    softmax = e / e.sum(axis=1, keepdims=True)

    probs = _to_prob_matrix(logits)

    assert probs.shape == softmax.shape
    np.testing.assert_allclose(
        probs,
        softmax,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        probs.sum(axis=1),
        np.ones(probs.shape[0]),
        atol=1e-6,
    )


def test_to_prob_matrix_preserves_valid_probabilities():
    """
    If we pass rows that already sum to 1, they should be preserved
    up to small numerical noise.
    """
    probs_in = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float64)
    probs = _to_prob_matrix(probs_in)

    assert probs.shape == probs_in.shape
    np.testing.assert_allclose(
        probs,
        probs_in,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        probs.sum(axis=1),
        np.ones(probs.shape[0]),
        atol=1e-8,
    )
