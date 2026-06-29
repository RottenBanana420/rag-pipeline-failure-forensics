from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.ingestion.models import ProcessedDocument


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
        doc = ProcessedDocument(**_valid_kwargs(source_format="html", section_heading="Overview"))
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
