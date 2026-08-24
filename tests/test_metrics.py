"""Tests for retrieval metrics.

These run without Postgres or an API key so CI's fast path stays fast, and they
pin the exact metric conventions the regression gate depends on.
"""

import math

import pytest

from groundtruth.eval.metrics import (
    cohens_kappa,
    dcg,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    recall_at_k,
)


def test_dcg_uses_exponential_gain():
    # single item, relevance 3 -> (2^3 - 1) / log2(2) = 7.0
    assert dcg([3.0]) == pytest.approx(7.0)
    # second position is discounted by log2(3)
    assert dcg([0.0, 3.0]) == pytest.approx(7.0 / math.log2(3))


def test_ndcg_perfect_ranking_is_one():
    labels = {1: 3.0, 2: 2.0, 3: 1.0}
    assert ndcg_at_k([1, 2, 3], labels, k=3) == pytest.approx(1.0)


def test_ndcg_penalizes_inversion():
    labels = {1: 3.0, 2: 0.0}
    assert ndcg_at_k([2, 1], labels, k=2) < ndcg_at_k([1, 2], labels, k=2)


def test_ndcg_penalizes_missed_relevant_item():
    """A relevant chunk that stage 1 never retrieved must still cost us."""
    labels = {1: 3.0, 99: 3.0}  # 99 is relevant but absent from the ranking
    assert ndcg_at_k([1], labels, k=10) < 1.0


def test_ndcg_no_labels_is_zero():
    assert ndcg_at_k([1, 2], {}, k=10) == 0.0


def test_recall_at_k_counts_only_thresholded_relevants():
    labels = {1: 2.0, 2: 0.0, 3: 1.0}
    # relevant = {1, 3}; top-2 of [1, 2, 3] contains only 1
    assert recall_at_k([1, 2, 3], labels, k=2) == pytest.approx(0.5)
    assert recall_at_k([1, 2, 3], labels, k=3) == pytest.approx(1.0)


def test_mrr_and_hit_rate():
    labels = {5: 1.0}
    assert mrr([9, 8, 5], labels) == pytest.approx(1 / 3)
    assert mrr([9, 8], labels) == 0.0
    assert hit_rate_at_k([9, 8, 5], labels, k=3) == 1.0
    assert hit_rate_at_k([9, 8, 5], labels, k=2) == 0.0


def test_kappa_bounds():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # Perfect disagreement on balanced classes -> -1.0
    assert cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)


def test_kappa_degenerate_single_category():
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)


def test_kappa_rejects_mismatched_input():
    with pytest.raises(ValueError):
        cohens_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        cohens_kappa([], [])
