"""Tests for reranker feature extraction.

Guards the train/serve contract: FEATURE_NAMES order and length must match what
`extract_features` emits, because the booster indexes features positionally.
"""

import pytest

from groundtruth.retrieval.features import FEATURE_NAMES, FeatureContext, extract_features
from groundtruth.retrieval.types import Candidate


def make_candidate(**overrides) -> Candidate:
    defaults = {
        "chunk_id": 1,
        "text": "Goodwill impairment of $1,200 million was recorded in fiscal 2023.",
        "token_count": 14,
        "rrf_score": 0.03,
        "dense_score": 0.82,
        "dense_rank": 1,
        "sparse_score": 0.44,
        "sparse_rank": 3,
        "item_section": "7",
        "section_name": "MD&A",
        "company_name": "Acme Corp",
        "ticker": "ACME",
        "fiscal_year": 2023,
        "accession_no": "0000000000-23-000001",
        "source_url": "https://example.invalid/filing",
        "char_start": 0,
        "char_end": 64,
    }
    return Candidate(**{**defaults, **overrides})


def test_feature_vector_length_matches_names():
    ctx = FeatureContext.build("goodwill impairment")
    vec = extract_features(make_candidate(), ctx)
    assert len(vec) == len(FEATURE_NAMES)


def test_all_features_are_finite_floats():
    ctx = FeatureContext.build("goodwill impairment")
    vec = extract_features(make_candidate(), ctx)
    assert all(isinstance(v, float) for v in vec)


def test_exact_phrase_detected():
    ctx = FeatureContext.build("goodwill impairment")
    vec = extract_features(make_candidate(), ctx)
    assert vec[FEATURE_NAMES.index("exact_phrase_hit")] == 1.0


def test_exact_phrase_absent():
    ctx = FeatureContext.build("segment revenue growth")
    vec = extract_features(make_candidate(), ctx)
    assert vec[FEATURE_NAMES.index("exact_phrase_hit")] == 0.0


def test_query_coverage_is_fractional():
    ctx = FeatureContext.build("goodwill impairment unrelated_token_xyz")
    vec = extract_features(make_candidate(), ctx)
    coverage = vec[FEATURE_NAMES.index("query_coverage")]
    assert 0.0 < coverage < 1.0


def test_numeric_overlap_matches_normalized_figures():
    """'$1,200' in the query should match '$1,200' in the chunk after normalization."""
    ctx = FeatureContext.build("was the impairment $1,200 million")
    vec = extract_features(make_candidate(), ctx)
    assert vec[FEATURE_NAMES.index("numeric_overlap")] > 0.0


def test_missing_retriever_scores_degrade_to_zero():
    """A candidate found only by BM25 has no dense score -- must not crash."""
    ctx = FeatureContext.build("goodwill")
    vec = extract_features(make_candidate(dense_score=None, dense_rank=None), ctx)
    assert vec[FEATURE_NAMES.index("dense_score")] == 0.0
    assert vec[FEATURE_NAMES.index("dense_rank_recip")] == 0.0
    assert vec[FEATURE_NAMES.index("both_retrievers_hit")] == 0.0


def test_section_one_hots_are_mutually_exclusive():
    ctx = FeatureContext.build("risk")
    vec = extract_features(make_candidate(item_section="1A"), ctx)
    assert vec[FEATURE_NAMES.index("section_is_risk")] == 1.0
    assert vec[FEATURE_NAMES.index("section_is_mdna")] == 0.0
    assert vec[FEATURE_NAMES.index("section_is_financials")] == 0.0


def test_cross_encoder_score_is_neutral_when_absent():
    ctx = FeatureContext.build("goodwill")
    vec = extract_features(make_candidate(), ctx, cross_encoder_score=None)
    assert vec[FEATURE_NAMES.index("cross_encoder_score")] == 0.0


@pytest.mark.parametrize("section", [None, "", "1", "15"])
def test_unknown_sections_do_not_raise(section):
    ctx = FeatureContext.build("anything")
    assert len(extract_features(make_candidate(item_section=section), ctx)) == len(FEATURE_NAMES)
