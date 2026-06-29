from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.ingestion.chunker import Chunker
from src.ingestion.models import Chunk, ProcessedDocument


def _make_doc(**overrides: object) -> ProcessedDocument:
    base: dict[str, object] = {
        "doc_id": "doc123",
        "source_path": "data/raw/sample.md",
        "source_format": "markdown",
        "title": "Sample",
        "section_heading": "Intro",
        "page_number": None,
        "text": (
            "This is the first paragraph of the document. "
            "It contains multiple sentences and should be split. "
            "Here is another sentence to make it longer. "
            "And yet another sentence to ensure the chunk size is exceeded. "
            "Final sentence of the first paragraph."
        ),
        "processed_at": "2026-06-28T00:00:00+00:00",
    }
    base.update(overrides)
    return ProcessedDocument(**base)


class TestFixedSizeChunker:
    def test_returns_chunk_objects(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        chunks = chunker.chunk([_make_doc()])
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_strategy_field_is_fixed_size(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        chunks = chunker.chunk([_make_doc()])
        assert all(c.strategy == "fixed_size" for c in chunks)

    def test_metadata_preserved(self) -> None:
        doc = _make_doc()
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        for c in chunker.chunk([doc]):
            assert c.doc_id == doc.doc_id
            assert c.source_path == doc.source_path
            assert c.source_format == doc.source_format
            assert c.title == doc.title
            assert c.section_heading == doc.section_heading
            assert c.page_number == doc.page_number

    def test_chunk_indices_sequential(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        chunks = chunker.chunk([_make_doc()])
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunk_id_deterministic(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        assert (
            [c.chunk_id for c in chunker.chunk([_make_doc()])]
            == [c.chunk_id for c in chunker.chunk([_make_doc()])]
        )

    def test_all_text_covered(self) -> None:
        doc = _make_doc()
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        combined = " ".join(c.text for c in chunker.chunk([doc]))
        for word in doc.text.split():
            assert word in combined

    def test_empty_doc_produces_no_chunks(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        assert chunker.chunk([_make_doc(text="")]) == []

    def test_small_doc_is_single_chunk(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=10000, chunk_overlap=0))
        chunks = chunker.chunk([_make_doc(text="Short text.")])
        assert len(chunks) == 1
        assert chunks[0].text == "Short text."

    def test_multiple_docs_indexed_continuously(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="fixed_size", chunk_size=100, chunk_overlap=20))
        chunks = chunker.chunk([_make_doc(doc_id="d1"), _make_doc(doc_id="d2")])
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
