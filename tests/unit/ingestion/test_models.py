from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.ingestion.models import Chunk, ProcessedDocument, chunk_id


def _valid_kwargs(**overrides) -> dict:
    base = {
        "doc_id": "abc123",
        "source_path": "data/raw/sample.md",
        "source_format": "markdown",
        "title": "Sample",
        "section_heading": None,
        "page_number": None,
        "text": "Hello world.",
        "processed_at": "2026-06-28T00:00:00+00:00",
    }
    base.update(overrides)
    return base


class TestProcessedDocumentValidation:
    def test_valid_markdown(self):
        doc = ProcessedDocument(**_valid_kwargs())
        assert doc.source_format == "markdown"
        assert doc.section_heading is None
        assert doc.page_number is None

    def test_valid_pdf_with_page(self):
        doc = ProcessedDocument(**_valid_kwargs(source_format="pdf", page_number=1))
        assert doc.page_number == 1

    def test_valid_html_with_section_heading(self):
        doc = ProcessedDocument(
            **_valid_kwargs(source_format="html", section_heading="Overview")
        )
        assert doc.section_heading == "Overview"

    def test_invalid_source_format_rejected(self):
        with pytest.raises(ValidationError):
            ProcessedDocument(**_valid_kwargs(source_format="docx"))

    def test_all_formats_accepted(self):
        for fmt in ("markdown", "text", "html", "pdf"):
            doc = ProcessedDocument(**_valid_kwargs(source_format=fmt))
            assert doc.source_format == fmt

    def test_round_trip_json(self):
        doc = ProcessedDocument(**_valid_kwargs(section_heading="Intro"))
        restored = ProcessedDocument.model_validate_json(doc.model_dump_json())
        assert restored == doc


def _valid_chunk_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "chunk_id": "aabbcc",
        "doc_id": "abc123",
        "source_path": "data/raw/sample.md",
        "source_format": "markdown",
        "title": "Sample",
        "section_heading": None,
        "page_number": None,
        "text": "Hello world.",
        "chunk_index": 0,
        "strategy": "fixed_size",
        "processed_at": "2026-06-28T00:00:00+00:00",
    }
    base.update(overrides)
    return base


class TestChunkValidation:
    def test_valid_fixed_size(self):
        c = Chunk(**_valid_chunk_kwargs())
        assert c.strategy == "fixed_size"
        assert c.chunk_index == 0

    def test_valid_recursive_header(self):
        c = Chunk(**_valid_chunk_kwargs(strategy="recursive_header"))
        assert c.strategy == "recursive_header"

    def test_valid_semantic(self):
        c = Chunk(**_valid_chunk_kwargs(strategy="semantic"))
        assert c.strategy == "semantic"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValidationError):
            Chunk(**_valid_chunk_kwargs(strategy="unknown"))

    def test_negative_chunk_index_rejected(self):
        with pytest.raises(ValidationError):
            Chunk(**_valid_chunk_kwargs(chunk_index=-1))

    def test_round_trip_json(self):
        c = Chunk(**_valid_chunk_kwargs(section_heading="Intro"))
        assert Chunk.model_validate_json(c.model_dump_json()) == c


class TestChunkId:
    def test_deterministic(self):
        assert chunk_id("doc1", "hello") == chunk_id("doc1", "hello")

    def test_different_docs_differ(self):
        assert chunk_id("doc1", "hello") != chunk_id("doc2", "hello")

    def test_different_text_differs(self):
        assert chunk_id("doc1", "hello") != chunk_id("doc1", "world")

    def test_returns_hex_string(self):
        result = chunk_id("x", "y")
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
