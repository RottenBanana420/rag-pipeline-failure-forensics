from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProcessedDocument(BaseModel):
    doc_id: str
    source_path: str
    source_format: Literal["markdown", "text", "html", "pdf"]
    title: str
    section_heading: str | None
    page_number: int | None
    text: str
    processed_at: str = Field(description="ISO-8601 UTC timestamp")
