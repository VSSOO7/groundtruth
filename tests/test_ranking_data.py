"""Tests for learning-to-rank data handling.

The grouped split and group-size computation are the two places a ranking setup
silently breaks: leak a query across the split and validation nDCG becomes
meaningless; miscompute group sizes and XGBoost silently ranks across query
boundaries. Both failures produce plausible-looking numbers, so they get tests.

Only numpy is needed -- xgboost is imported lazily inside the module under test's
main path, so these run in CI's fast lane.
"""

import numpy as np
import pytest

from groundtruth.training.train_reranker import group_sizes, group_split


def test_group_split_never_leaks_a_query():
    qids = np.repeat(np.arange(20), 5)  # 20 queries, 5 candidates each
    train_idx, val_idx = group_split(qids, val_fraction=0.25, seed=1)

    train_qids = set(qids[train_idx].tolist())
    val_qids = set(qids[val_idx].tolist())
    assert train_qids.isdisjoint(val_qids)


def test_group_split_covers_every_row():
    qids = np.repeat(np.arange(12), 4)
    train_idx, val_idx = group_split(qids, val_fraction=0.25, seed=3)
    assert len(train_idx) + len(val_idx) == len(qids)
    assert set(train_idx.tolist()) | set(val_idx.tolist()) == set(range(len(qids)))


def test_group_split_is_deterministic_for_a_seed():
    qids = np.repeat(np.arange(30), 3)
    a = group_split(qids, val_fraction=0.2, seed=7)
    b = group_split(qids, val_fraction=0.2, seed=7)
    assert np.array_equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_group_split_seed_changes_partition():
    qids = np.repeat(np.arange(40), 3)
    a = group_split(qids, val_fraction=0.25, seed=1)[1]
    b = group_split(qids, val_fraction=0.25, seed=2)[1]
    assert not np.array_equal(a, b)


def test_group_split_always_yields_a_nonempty_val_set():
    """Tiny datasets must still produce at least one validation query."""
    qids = np.repeat(np.arange(2), 3)
    _, val_idx = group_split(qids, val_fraction=0.01, seed=1)
    assert len(val_idx) > 0


def test_group_sizes_matches_contiguous_runs():
    qids = np.array([0, 0, 0, 1, 1, 2])
    assert group_sizes(qids) == [3, 2, 1]


def test_group_sizes_sums_to_row_count():
    qids = np.repeat(np.arange(9), 4)
    assert sum(group_sizes(qids)) == len(qids)


def test_group_sizes_handles_single_group():
    assert group_sizes(np.array([5, 5, 5])) == [3]


def test_group_sizes_empty():
    assert group_sizes(np.array([], dtype=int)) == []


@pytest.mark.parametrize("fraction", [0.1, 0.2, 0.5])
def test_group_split_respects_requested_fraction(fraction):
    qids = np.repeat(np.arange(100), 2)
    _, val_idx = group_split(qids, val_fraction=fraction, seed=11)
    val_query_count = len(set(qids[val_idx].tolist()))
    assert val_query_count == pytest.approx(100 * fraction, rel=0.2)
