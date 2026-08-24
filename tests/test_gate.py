"""Tests for the CI regression gate.

The gate is the only thing standing between a quality regression and `main`, so
its failure modes get direct coverage. The cases that matter are the boundary
(exactly at tolerance), the asymmetry (improvements must never fail), and the
degenerate baselines that would otherwise divide by zero.
"""

import json
import os
import tempfile

import pytest

from groundtruth.eval.gate import _load, compare


def _metrics(overall, human=None):
    out = {"overall": {"ndcg@10": overall}}
    if human is not None:
        out["human_verified"] = {"ndcg@10": human}
    return out


def test_identical_runs_pass():
    passed, _ = compare(_metrics(0.70), _metrics(0.70), tolerance=0.02)
    assert passed


def test_improvement_passes():
    passed, lines = compare(_metrics(0.70), _metrics(0.85), tolerance=0.02)
    assert passed
    assert any("REGRESSION" not in line for line in lines)


def test_large_improvement_never_fails():
    """The gate is one-sided by design -- a big jump is good news, not an anomaly."""
    passed, _ = compare(_metrics(0.40), _metrics(0.95), tolerance=0.02)
    assert passed


def test_regression_beyond_tolerance_fails():
    # 0.70 -> 0.65 is a 7.1% drop, well past a 2% tolerance.
    passed, lines = compare(_metrics(0.70), _metrics(0.65), tolerance=0.02)
    assert not passed
    assert any("REGRESSION" in line for line in lines)


def test_regression_within_tolerance_passes():
    # 0.70 -> 0.693 is a 1% drop: noise, not a regression.
    passed, _ = compare(_metrics(0.70), _metrics(0.693), tolerance=0.02)
    assert passed


def test_regression_exactly_at_tolerance_passes():
    """Boundary: a drop of exactly the tolerance is tolerated, not failed."""
    baseline = 0.70
    candidate = baseline * (1 - 0.02)
    passed, _ = compare(_metrics(baseline), _metrics(candidate), tolerance=0.02)
    assert passed


def test_human_slice_regression_fails_even_when_overall_is_flat():
    """The case the gate exists for: bootstrapped labels look fine, trusted slice drops."""
    passed, lines = compare(
        _metrics(0.70, human=0.80),
        _metrics(0.70, human=0.60),
        tolerance=0.02,
    )
    assert not passed
    assert any("human_verified" in line and "REGRESSION" in line for line in lines)


def test_missing_human_slice_is_skipped_not_failed():
    passed, lines = compare(_metrics(0.70, human=0.80), _metrics(0.70), tolerance=0.02)
    assert passed
    assert any("skipped" in line for line in lines)


def test_zero_baseline_does_not_divide_by_zero():
    """A fresh baseline of 0.0 must not crash or report an infinite percentage."""
    passed, lines = compare(_metrics(0.0), _metrics(0.30), tolerance=0.02)
    assert passed
    assert all("inf" not in line.lower() for line in lines)


def test_zero_candidate_against_real_baseline_fails():
    passed, _ = compare(_metrics(0.70), _metrics(0.0), tolerance=0.02)
    assert not passed


def test_accepts_full_run_envelope_shape():
    """run_eval writes {run_id, config, metrics}; the gate must read either shape."""
    envelope = {"run_id": 7, "config": {}, "metrics": _metrics(0.70)}
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(envelope, fh)
        assert _load(path) == _metrics(0.70)
    finally:
        os.unlink(path)


@pytest.mark.parametrize("tolerance", [0.0, 0.05, 0.10])
def test_tolerance_is_respected(tolerance):
    baseline = 0.80
    just_inside = baseline * (1 - tolerance)
    passed, _ = compare(_metrics(baseline), _metrics(just_inside), tolerance=tolerance)
    assert passed

    just_outside = baseline * (1 - tolerance) - 0.01
    passed, _ = compare(_metrics(baseline), _metrics(just_outside), tolerance=tolerance)
    assert not passed
