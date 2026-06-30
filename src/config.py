"""Application-wide configuration via pydantic-settings.

All values are read from environment variables or .env file.
Import the singleton `settings` — do not instantiate Settings() elsewhere
unless you need isolated values in tests.

    from src.config import settings
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API keys
    openai_api_key: str = Field(default="", description="OpenAI API key")

    # Embedding
    embedding_model: str = Field(default="text-embedding-3-small")

    # Vector store
    chroma_persist_dir: Path = Field(default=Path("./data/chroma"))

    # Retrieval
    dense_top_k: int = Field(default=10, ge=1)
    sparse_top_k: int = Field(default=10, ge=1)
    rerank_top_n: int = Field(default=5, ge=1)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    sparse_weight: float = Field(default=0.3, ge=0.0, le=1.0)

    # Data directories
    raw_data_dir: Path = Field(default=Path("./data/raw"))
    processed_data_dir: Path = Field(default=Path("./data/processed"))

    # Deduplication
    dedup_threshold: float = Field(default=0.95, ge=0.0, le=1.0)

    # Chunking
    chunk_strategy: Literal["fixed_size", "recursive_header", "semantic"] = Field(
        default="fixed_size"
    )
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    semantic_breakpoint_percentile: float = Field(default=95.0, ge=0.0, le=100.0)

    # Logging
    log_level: str = Field(default="INFO")

    @model_validator(mode="after")
    def chunk_overlap_less_than_size(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @property
    def chroma_persist_dir_str(self) -> str:
        """Return chroma_persist_dir as str (ChromaDB requires str, not Path)."""
        return str(self.chroma_persist_dir)


settings = Settings()
