from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, Field

ChunkingStrategy = Literal["fixed_size", "recursive_header", "semantic"]


class ProcessedDocument(BaseModel):
    doc_id: str
    source_path: str
    source_format: Literal["markdown", "text", "html", "pdf"]
    title: str
    section_heading: str | None
    page_number: int | None
    text: str
    processed_at: str = Field(description="ISO-8601 UTC timestamp")


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    source_path: str
    source_format: Literal["markdown", "text", "html", "pdf"]
    title: str
    section_heading: str | None
    page_number: int | None
    text: str
    chunk_index: int = Field(ge=0)
    strategy: ChunkingStrategy
    processed_at: str = Field(description="ISO-8601 UTC timestamp")


def chunk_id(doc_id: str, text: str) -> str:
    """Deterministic content-addressed ID for a chunk."""
    return hashlib.sha256(f"{doc_id}:{text}".encode()).hexdigest()
