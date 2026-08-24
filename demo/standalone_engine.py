"""Standalone demo retrieval & ranking engine for Groundtruth.

Runs directly over `eval/fixture_corpus.json` or live filings without needing an
active external PostgreSQL service. Supports dense embeddings, BM25 sparse matching,
RRF fusion, and XGBoost learned reranking.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from groundtruth.embedding import Embedder, SentenceTransformerEmbedder
from groundtruth.retrieval.features import (
    FEATURE_NAMES,
    FeatureContext,
    extract_features,
)
from groundtruth.retrieval.reranker import RerankedCandidate, Reranker, dedupe_by_document
from groundtruth.retrieval.types import Candidate, Filters

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class DemoCorpusItem:
    chunk_id: int
    passage_id: str
    accession_no: str
    cik: str
    company_name: str
    ticker: str
    form_type: str
    fiscal_year: int
    item_section: str
    section_name: str
    text: str
    token_count: int
    vector: np.ndarray | None = None


class BM25Index:
    """In-memory BM25 index over candidate passages."""

    def __init__(self, corpus: list[DemoCorpusItem], k1: float = 1.2, b: float = 0.75) -> None:
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_lens = [len(_tokenize(doc.text)) for doc in corpus]
        self.avg_doc_len = float(np.mean(self.doc_lens)) if self.doc_lens else 1.0
        self.doc_freqs: dict[str, int] = Counter()
        self.doc_term_freqs: list[dict[str, int]] = []

        for doc in corpus:
            tf = Counter(_tokenize(doc.text))
            self.doc_term_freqs.append(dict(tf))
            for term in tf:
                self.doc_freqs[term] += 1

        self.n_docs = len(corpus)
        self.idf: dict[str, float] = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> list[float]:
        query_terms = _tokenize(query)
        scores = [0.0] * self.n_docs
        for i, tf_dict in enumerate(self.doc_term_freqs):
            doc_len = self.doc_lens[i]
            score = 0.0
            for term in query_terms:
                if term not in tf_dict:
                    continue
                tf = tf_dict[term]
                idf = self.idf.get(term, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
            scores[i] = score
        return scores


class StandaloneDemoEngine:
    """In-memory demonstration engine mirroring the production QueryPipeline."""

    def __init__(
        self,
        fixture_path: Path | str = "eval/fixture_corpus.json",
        model_path: Path | str = "models/reranker.ubj",
    ) -> None:
        self.fixture_path = Path(fixture_path)
        self.model_path = Path(model_path)
        self.corpus: list[DemoCorpusItem] = []
        self._load_corpus()
        self.bm25 = BM25Index(self.corpus)
        self.embedder: Embedder | None = None
        self.vectors: np.ndarray | None = None
        self.reranker = Reranker.load(self.model_path)

    def _load_corpus(self) -> None:
        if not self.fixture_path.exists():
            return
        data = json.loads(self.fixture_path.read_text())
        chunk_id = 1
        for doc in data.get("documents", []):
            for passage in doc.get("passages", []):
                text = passage["text"]
                self.corpus.append(
                    DemoCorpusItem(
                        chunk_id=chunk_id,
                        passage_id=passage.get("id", f"p-{chunk_id}"),
                        accession_no=doc["accession_no"],
                        cik=doc["cik"],
                        company_name=doc["company_name"],
                        ticker=doc["ticker"],
                        form_type=doc.get("form_type", "10-K"),
                        fiscal_year=doc["fiscal_year"],
                        item_section=passage.get("item_section", "7"),
                        section_name=passage.get("section_name", "MD&A"),
                        text=text,
                        token_count=len(text.split()),
                    )
                )
                chunk_id += 1

    def ensure_embeddings(self) -> None:
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", 384, device="cpu")
            texts = [item.text for item in self.corpus]
            vecs = self.embedder.embed_passages(texts)
            self.vectors = np.asarray(vecs, dtype=np.float32)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
            self.vectors = self.vectors / np.maximum(norms, 1e-9)

    def query(
        self,
        query_text: str,
        filters: Filters | None = None,
        top_k: int = 8,
        rrf_k: int = 60,
    ) -> tuple[list[RerankedCandidate], dict[str, float]]:
        t_start = time.perf_counter()
        self.ensure_embeddings()
        assert self.embedder is not None
        assert self.vectors is not None

        t_embed_start = time.perf_counter()
        qvec = np.asarray(self.embedder.embed_query(query_text), dtype=np.float32)
        qvec = qvec / max(float(np.linalg.norm(qvec)), 1e-9)
        embed_ms = (time.perf_counter() - t_embed_start) * 1000

        t_retrieve_start = time.perf_counter()
        dense_sims = np.dot(self.vectors, qvec).tolist()
        sparse_scores = self.bm25.score(query_text)

        # Rank indices
        dense_order = np.argsort(dense_sims)[::-1].tolist()
        sparse_order = np.argsort(sparse_scores)[::-1].tolist()

        dense_ranks = {idx: rank + 1 for rank, idx in enumerate(dense_order)}
        sparse_ranks = {idx: rank + 1 for rank, idx in enumerate(sparse_order)}

        candidates: list[Candidate] = []
        for idx, item in enumerate(self.corpus):
            # Apply filters
            if filters:
                if filters.cik and item.cik != filters.cik:
                    continue
                if filters.fiscal_year and item.fiscal_year != filters.fiscal_year:
                    continue
                if filters.sections and item.item_section not in filters.sections:
                    continue

            d_rank = dense_ranks[idx]
            s_rank = sparse_ranks[idx]
            d_score = float(dense_sims[idx])
            s_score = float(sparse_scores[idx])
            rrf = (1.0 / (rrf_k + d_rank)) + (1.0 / (rrf_k + s_rank))

            cand = Candidate(
                chunk_id=item.chunk_id,
                text=item.text,
                token_count=item.token_count,
                rrf_score=rrf,
                dense_score=d_score,
                dense_rank=d_rank,
                sparse_score=s_score,
                sparse_rank=s_rank,
                item_section=item.item_section,
                section_name=item.section_name,
                company_name=item.company_name,
                ticker=item.ticker,
                fiscal_year=item.fiscal_year,
                accession_no=item.accession_no,
                source_url=f"https://www.sec.gov/edgar/data/{item.cik}/{item.accession_no}.txt",
                char_start=0,
                char_end=len(item.text),
                form_type=item.form_type,
            )
            candidates.append(cand)

        # Sort by RRF
        candidates.sort(key=lambda c: c.rrf_score, reverse=True)
        retrieve_ms = (time.perf_counter() - t_retrieve_start) * 1000

        # Rerank
        t_rerank_start = time.perf_counter()
        reranked = self.reranker.rerank(query_text, candidates, top_k=None)
        ranked = dedupe_by_document(reranked, max_per_doc=2)[:top_k]
        rerank_ms = (time.perf_counter() - t_rerank_start) * 1000

        timings = {
            "embed_ms": embed_ms,
            "retrieve_ms": retrieve_ms,
            "rerank_ms": rerank_ms,
            "total_ms": (time.perf_counter() - t_start) * 1000,
        }
        return ranked, timings

    def get_feature_contributions(self, query_text: str, candidate: Candidate) -> dict[str, float]:
        ctx = FeatureContext.build(query_text)
        feats = extract_features(candidate, ctx)
        return dict(zip(FEATURE_NAMES, feats, strict=True))
