"""Retrieval metrics, implemented directly rather than pulled from a RAG framework.

Two reasons this is hand-written: (1) the CI gate depends on these numbers, so
their definitions must be auditable in-repo, and (2) framework implementations
disagree on details that move the number by several points -- notably whether
nDCG uses exponential or linear gain, and whether it truncates the ideal ranking
to k. The conventions used here are stated explicitly per function.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dcg(relevances: Sequence[float]) -> float:
    """Discounted cumulative gain with *exponential* gain: (2^rel - 1) / log2(i+2).

    Exponential gain is the standard for graded relevance (TREC, LambdaMART's
    internal objective) and is what `xgboost`'s rank:ndcg optimizes, so training
    and evaluation agree.
    """
    return float(sum((2.0**rel - 1.0) / math.log2(i + 2) for i, rel in enumerate(relevances)))


def ndcg_at_k(
    ranked_ids: Sequence[int],
    labels: Mapping[int, float],
    k: int = 10,
) -> float:
    """nDCG@k. Unlabeled retrieved items count as relevance 0.

    The ideal ranking is the top-k of *all labeled* items, not just retrieved
    ones -- so failing to retrieve a known-relevant chunk is correctly penalized.
    """
    gains = [labels.get(cid, 0.0) for cid in ranked_ids[:k]]
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked_ids: Sequence[int],
    labels: Mapping[int, float],
    k: int = 50,
    *,
    threshold: float = 1.0,
) -> float:
    """Fraction of relevant items (relevance >= threshold) present in the top-k.

    This is the metric that bounds the whole system: the reranker can only
    reorder what stage 1 retrieved. Recall@candidate_k is therefore the ceiling
    on final answer quality, and worth tracking separately from nDCG.
    """
    relevant = {cid for cid, rel in labels.items() if rel >= threshold}
    if not relevant:
        return 0.0
    hit = sum(1 for cid in ranked_ids[:k] if cid in relevant)
    return hit / len(relevant)


def mrr(
    ranked_ids: Sequence[int],
    labels: Mapping[int, float],
    *,
    threshold: float = 1.0,
) -> float:
    """Reciprocal rank of the first relevant item; 0.0 if none found."""
    for i, cid in enumerate(ranked_ids):
        if labels.get(cid, 0.0) >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(
    ranked_ids: Sequence[int],
    labels: Mapping[int, float],
    k: int = 10,
    *,
    threshold: float = 1.0,
) -> float:
    """1.0 if any relevant item appears in the top-k."""
    return 1.0 if any(labels.get(c, 0.0) >= threshold for c in ranked_ids[:k]) else 0.0


def cohens_kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Inter-rater agreement between two label sets over the same items.

    Used to report how well the LLM judge agrees with the human-verified slice.
    Publishing this is the point: an unvalidated LLM-as-judge score is a vibe,
    not a metric. Rule of thumb -- below ~0.6 the judge is not trustworthy and
    the rubric needs work before any downstream number means anything.
    """
    if len(a) != len(b) or not a:
        raise ValueError("label sequences must be non-empty and equal length")

    categories = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(categories)}
    n = len(a)

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    count_a = [0] * len(categories)
    count_b = [0] * len(categories)
    for x, y in zip(a, b, strict=True):
        count_a[idx[x]] += 1
        count_b[idx[y]] += 1
    expected = sum((count_a[i] / n) * (count_b[i] / n) for i in range(len(categories)))

    if expected == 1.0:  # degenerate: both raters used a single identical category
        return 1.0
    return (observed - expected) / (1.0 - expected)
