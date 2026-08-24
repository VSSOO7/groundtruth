"""Embedding model wrapper.

Two details that materially affect retrieval quality and signal domain fluency:

1. **Asymmetric prefixes.** bge-* models are trained with an instruction prefix on
   the *query* side only ("Represent this sentence for searching relevant
   passages:"). Passages are embedded raw. Applying the query prefix to passages
   -- or forgetting it on queries -- measurably drops recall. We encode the two
   sides through different methods so the call site cannot mix them up.

2. **Normalized embeddings + cosine.** We L2-normalize so the pgvector cosine
   operator (`<=>`) and an inner product agree, and so scores are comparable
   across queries.

The `Embedder` protocol lets the online path, the ingest job, and the tests share
one interface; tests inject a deterministic fake instead of loading 400MB of
weights.
"""

from __future__ import annotations

from typing import Protocol, cast

# bge-base-en-v1.5's documented retrieval instruction. Do not change without
# rebuilding the index -- query and passage embeddings must come from the same
# model and convention or similarity is meaningless.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    """Minimal surface the retrieval and ingestion code depends on."""

    dim: int

    def embed_query(self, text: str) -> list[float]: ...

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Production embedder backed by sentence-transformers.

    Imported lazily so that importing this module (and thus the whole retrieval
    package) does not drag in torch. Only constructing the class loads weights.
    """

    def __init__(self, model_name: str, expected_dim: int, *, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # lazy: avoids torch at import

        self._model = SentenceTransformer(model_name, device=device)
        self.dim = self._model.get_sentence_embedding_dimension()
        if self.dim != expected_dim:
            # A dim mismatch means the DB's vector(N) column and the model
            # disagree -- every insert would fail. Fail loudly at startup instead.
            raise ValueError(
                f"Embedding dim mismatch: {model_name} produces {self.dim}, "
                f"schema expects {expected_dim}. Update EMBEDDING_DIM and the "
                f"chunks.embedding column together."
            )

    def embed_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            BGE_QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
        )
        return cast(list[float], vec.tolist())

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [cast(list[float], v.tolist()) for v in vecs]
