"""Tests for the judge/human agreement report.

The `report()` function is pure, so the interesting cases are testable offline:
perfect agreement, systematic bias (a labeler that always grades one notch high),
and the binary collapse that decides whether a chunk counts as relevant at all.
"""

import pytest

from groundtruth.eval.agreement import interpret, report


def _pairs(machine, human):
    return [
        {"query_id": 1, "chunk_id": i, "machine": m, "human": h}
        for i, (m, h) in enumerate(zip(machine, human, strict=True))
    ]


def test_perfect_agreement_gives_kappa_one():
    grades = [0, 1, 2, 3, 0, 3, 2, 1]
    stats = report(_pairs(grades, grades))
    assert stats["kappa_graded"] == pytest.approx(1.0)
    assert stats["exact_agreement"] == pytest.approx(1.0)
    assert stats["severe_disagreement_rate"] == pytest.approx(0.0)


def test_off_by_one_bias_is_counted_separately_from_severe():
    """A labeler grading consistently one notch high: no exact hits, no severe misses."""
    human = [0, 1, 2, 0, 1, 2]
    machine = [h + 1 for h in human]
    stats = report(_pairs(machine, human))
    assert stats["exact_agreement"] == pytest.approx(0.0)
    assert stats["off_by_one_rate"] == pytest.approx(1.0)
    assert stats["severe_disagreement_rate"] == pytest.approx(0.0)


def test_severe_disagreement_is_flagged():
    human = [0, 0, 3, 3]
    machine = [3, 3, 0, 0]
    stats = report(_pairs(machine, human))
    assert stats["severe_disagreement_rate"] == pytest.approx(1.0)
    assert stats["kappa_graded"] < 0.0  # worse than chance


def test_binary_view_forgives_grade_granularity():
    """2-vs-3 disagreements vanish under the relevant/not-relevant collapse."""
    human = [3, 3, 2, 0, 0]
    machine = [2, 2, 3, 0, 0]
    stats = report(_pairs(machine, human))
    assert stats["kappa_binary_relevant"] == pytest.approx(1.0)
    assert stats["kappa_graded"] < 1.0


def test_binary_view_still_catches_relevance_flips():
    human = [3, 3, 0, 0]
    machine = [0, 0, 3, 3]
    stats = report(_pairs(machine, human))
    assert stats["kappa_binary_relevant"] < 0.0


def test_n_pairs_is_reported():
    stats = report(_pairs([0, 1, 2], [0, 1, 2]))
    assert stats["n_pairs"] == 3


def test_rates_sum_to_one():
    """Every pair is exactly one of: exact, off-by-one, or severe."""
    human = [0, 1, 2, 3, 0, 1]
    machine = [0, 2, 2, 1, 3, 1]
    stats = report(_pairs(machine, human))
    total = stats["exact_agreement"] + stats["off_by_one_rate"] + stats["severe_disagreement_rate"]
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kappa,expected_fragment",
    [
        (0.05, "poor"),
        (0.30, "fair"),
        (0.50, "moderate"),
        (0.70, "substantial"),
        (0.90, "near-perfect"),
    ],
)
def test_interpret_bands(kappa, expected_fragment):
    assert expected_fragment in interpret(kappa)
