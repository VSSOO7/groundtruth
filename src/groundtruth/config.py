"""Typed settings loaded from environment / .env.

Every knob that can change retrieval quality lives here and gets serialized into
`eval_runs.config`, so any metric in the leaderboard is reproducible from its row.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database ---
    database_url: str = "postgresql://groundtruth:groundtruth@localhost:5432/groundtruth"
    db_pool_min: int = 2
    db_pool_max: int = 10

    # --- Anthropic ---
    # Model IDs are complete as written -- never append a date suffix.
    anthropic_api_key: str = ""
    generation_model: str = "claude-opus-5"
    judge_model: str = "claude-opus-5"  # eval judge: quality matters most
    cheap_model: str = "claude-haiku-4-5"  # bulk label bootstrapping
    max_output_tokens: int = 16000
    effort: str = "high"  # low | medium | high | xhigh | max

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    # --- Retrieval ---
    candidate_k: int = Field(default=100, description="candidates per retriever before fusion")
    final_k: int = Field(default=8, description="chunks passed to the generator")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    hnsw_ef_search: int = Field(default=100, description="higher = better recall, slower")
    reranker_path: str = "models/reranker.ubj"

    # --- Ingestion ---
    sec_user_agent: str = "groundtruth/0.1 (set-SEC_USER_AGENT-in-env)"
    chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    chunker_version: str = "section-aware-v1"

    # --- Observability ---
    log_level: str = "INFO"
    otel_exporter_otlp_endpoint: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
