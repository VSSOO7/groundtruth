"""Tests for the grounded-answer schema and its grounding contract.

Pure Pydantic -- no SDK, no network -- so these run in CI's fast path and pin the
invariant that matters most: the system may never emit an uncited claim.
"""

import pytest
from pydantic import ValidationError

from groundtruth.generation.schema import Claim, GroundedAnswer


def test_valid_grounded_answer_exposes_cited_ids():
    a = GroundedAnswer(
        refused=False,
        answer="Revenue was $5M.",
        claims=[Claim(text="Revenue was $5M.", chunk_ids=[1, 2])],
        confidence=0.9,
    )
    assert a.cited_chunk_ids() == {1, 2}


def test_refusal_with_no_claims_is_valid():
    a = GroundedAnswer(refused=True, answer="Not in context.", claims=[], confidence=0.0)
    assert a.refused
    assert a.claims == []


def test_non_refusal_without_claims_is_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(refused=False, answer="Revenue was $5M.", claims=[], confidence=0.9)


def test_claim_without_citations_is_rejected():
    with pytest.raises(ValidationError):
        GroundedAnswer(
            refused=False,
            answer="x",
            claims=[Claim(text="x", chunk_ids=[])],
            confidence=0.5,
        )


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.1])
def test_confidence_bounds_enforced(bad_confidence):
    with pytest.raises(ValidationError):
        GroundedAnswer(refused=True, answer="x", claims=[], confidence=bad_confidence)
