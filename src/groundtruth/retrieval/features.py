"""Feature extraction for the learned reranker.

The whole thesis of this project: a Data Scientist's instinct -- turn a ranking
problem into features + a gradient-boosted model -- beats reaching for a bigger
neural reranker by default. These features are deliberately cheap (no extra GPU
pass except the optional cross-encoder) and interpretable, so `xgboost` feature
importances double as an ablation story in the README.

Keep this module pure and deterministic: it is imported by both the online path
and the offline training script, and the two MUST compute identical features or
the model sees a train/serve skew. Every feature added here must be added in
exactly one place -- this function -- and nowhere else.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from groundtruth.retrieval.types import Candidate

# Ordered feature names. Persisted alongside the model so the serving path can
# assert the vector layout matches what was trained. Order is the contract.
FEATURE_NAMES: tuple[str, ...] = (
    "dense_score",
    "dense_rank_recip",
    "sparse_score",
    "sparse_rank_recip",
    "rrf_score",
    "both_retrievers_hit",
    "cross_encoder_score",
    "token_count_log",
    "query_term_overlap",
    "query_coverage",
    "exact_phrase_hit",
    "numeric_overlap",
    "section_is_risk",
    "section_is_mdna",
    "section_is_financials",
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUM_RE = re.compile(r"\$?\d[\d,]*\.?\d*")


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _numbers(text: str) -> set[str]:
    # Normalize "$1,234.0" and "1234" toward comparability.
    return {n.replace("$", "").replace(",", "").rstrip(".0") or "0" for n in _NUM_RE.findall(text)}


@dataclass(slots=True)
class FeatureContext:
    """Per-query state shared across all candidates for that query."""

    query_text: str
    query_tokens: set[str]
    query_numbers: set[str]

    @classmethod
    def build(cls, query_text: str) -> FeatureContext:
        toks = _tokens(query_text)
        return cls(
            query_text=query_text.lower(),
            query_tokens=set(toks),
            query_numbers=_numbers(query_text),
        )


def extract_features(
    cand: Candidate,
    ctx: FeatureContext,
    *,
    cross_encoder_score: float | None = None,
) -> list[float]:
    """Map one (query, candidate) pair to the fixed-order feature vector.

    `cross_encoder_score` is optional: when a cross-encoder baseline is wired in
    we feed its score as a *feature* (stacking) rather than using it as the final
    ranker. When absent we pass a neutral 0.0 -- XGBoost handles the constant fine
    and the ablation table reports both regimes.
    """
    cand_tokens = _tokens(cand.text)
    cand_token_set = set(cand_tokens)
    cand_numbers = _numbers(cand.text)

    overlap = ctx.query_tokens & cand_token_set
    coverage = len(overlap) / max(len(ctx.query_tokens), 1)

    query_str = ctx.query_text.strip()
    exact_phrase = 1.0 if query_str and query_str in cand.text.lower() else 0.0

    numeric_overlap = len(ctx.query_numbers & cand_numbers) / max(len(ctx.query_numbers), 1)

    section = (cand.item_section or "").upper()

    return [
        cand.dense_score if cand.dense_score is not None else 0.0,
        1.0 / cand.dense_rank if cand.dense_rank else 0.0,
        cand.sparse_score if cand.sparse_score is not None else 0.0,
        1.0 / cand.sparse_rank if cand.sparse_rank else 0.0,
        cand.rrf_score,
        1.0 if (cand.dense_rank and cand.sparse_rank) else 0.0,
        cross_encoder_score if cross_encoder_score is not None else 0.0,
        math.log1p(cand.token_count),
        float(len(overlap)),
        coverage,
        exact_phrase,
        numeric_overlap,
        1.0 if section == "1A" else 0.0,  # Risk Factors
        1.0 if section == "7" else 0.0,  # MD&A
        1.0 if section == "8" else 0.0,  # Financial Statements
    ]


def feature_matrix(
    candidates: list[Candidate],
    ctx: FeatureContext,
    *,
    cross_encoder_scores: list[float] | None = None,
) -> list[list[float]]:
    """Vectorize a full candidate list. Row order mirrors `candidates`."""
    ce = cross_encoder_scores or [None] * len(candidates)  # type: ignore[list-item]
    return [
        extract_features(c, ctx, cross_encoder_score=s) for c, s in zip(candidates, ce, strict=True)
    ]
