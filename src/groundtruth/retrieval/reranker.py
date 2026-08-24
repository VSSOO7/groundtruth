"""The learned reranker: stage 2 of retrieval.

An XGBoost ranker (`rank:ndcg`, i.e. LambdaMART) scores each candidate from the
features in `features.py` and reorders them. This is the project's differentiator
-- most RAG systems either skip reranking or drop in an off-the-shelf
cross-encoder. Here the reranker is *trained on our own relevance labels*, so it
learns filing-specific signal (exact-phrase and numeric-overlap matter more in
10-Ks than in web text) that a generic model cannot.

Graceful degradation is a production requirement, not a nicety: if the model
artifact is missing (fresh clone, first boot before training) we fall back to the
RRF order and log it, so the API still serves. The reranker improves ranking; it
is never a hard dependency for liveness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import structlog
import xgboost as xgb

from groundtruth.retrieval.features import FEATURE_NAMES, FeatureContext, feature_matrix
from groundtruth.retrieval.types import Candidate

log = structlog.get_logger(__name__)


@dataclass(slots=True)
class RerankedCandidate:
    candidate: Candidate
    rerank_score: float
    prior_rrf_rank: int  # position before rerank -- lets us measure movement


class Reranker:
    """Loads a trained booster and reorders candidates. Stateless per-query."""

    def __init__(self, booster: xgb.Booster | None, feature_names: tuple[str, ...]):
        self._booster = booster
        self._feature_names = feature_names

    @property
    def is_active(self) -> bool:
        return self._booster is not None

    @classmethod
    def load(cls, path: str | Path) -> Reranker:
        """Load `<path>` (UBJSON booster) and its `<path>.features.json` sidecar.

        The sidecar pins feature order. If the trained layout no longer matches
        the code's FEATURE_NAMES, we refuse to load rather than silently serve a
        skewed model -- a wrong-but-confident ranker is worse than the fallback.
        """
        path = Path(path)
        if not path.exists():
            log.warning("reranker.artifact_missing", path=str(path), effect="fallback_to_rrf")
            return cls(None, FEATURE_NAMES)

        booster = xgb.Booster()
        booster.load_model(str(path))

        sidecar = path.with_suffix(path.suffix + ".features.json")
        trained_names = (
            tuple(json.loads(sidecar.read_text())) if sidecar.exists() else FEATURE_NAMES
        )
        if trained_names != FEATURE_NAMES:
            raise RuntimeError(
                "Reranker feature schema mismatch: model trained on "
                f"{trained_names} but code expects {FEATURE_NAMES}. Retrain."
            )
        log.info("reranker.loaded", path=str(path), n_features=len(trained_names))
        return cls(booster, trained_names)

    def rerank(
        self,
        query_text: str,
        candidates: list[Candidate],
        *,
        cross_encoder_scores: list[float] | None = None,
        top_k: int | None = None,
    ) -> list[RerankedCandidate]:
        if not candidates:
            return []

        if self._booster is None:
            # Fallback: preserve incoming (RRF) order, expose neutral scores.
            ranked = [
                RerankedCandidate(candidate=c, rerank_score=c.rrf_score, prior_rrf_rank=i)
                for i, c in enumerate(candidates)
            ]
            return ranked[: top_k or len(ranked)]

        ctx = FeatureContext.build(query_text)
        matrix = feature_matrix(candidates, ctx, cross_encoder_scores=cross_encoder_scores)
        dmat = xgb.DMatrix(
            np.asarray(matrix, dtype=np.float32), feature_names=list(self._feature_names)
        )
        scores = self._booster.predict(dmat)

        ranked = [
            RerankedCandidate(candidate=c, rerank_score=float(s), prior_rrf_rank=i)
            for i, (c, s) in enumerate(zip(candidates, scores, strict=True))
        ]
        ranked.sort(key=lambda r: r.rerank_score, reverse=True)
        return ranked[: top_k or len(ranked)]


def dedupe_by_document(
    ranked: list[RerankedCandidate], *, max_per_doc: int = 3
) -> list[RerankedCandidate]:
    """Diversity guard: stop one filing from monopolizing the context window.

    Applied post-rerank so we drop a filing's *weakest* chunks, keeping its
    best-ranked ones. Returns copies with `prior_rrf_rank` preserved.
    """
    seen: dict[str, int] = {}
    kept: list[RerankedCandidate] = []
    for r in ranked:
        key = r.candidate.accession_no
        if seen.get(key, 0) >= max_per_doc:
            continue
        seen[key] = seen.get(key, 0) + 1
        kept.append(replace(r))
    return kept
