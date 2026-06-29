from __future__ import annotations

from unittest.mock import MagicMock, patch

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


class TestRecursiveHeaderChunker:
    def test_returns_chunks(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="recursive_header", chunk_size=100, chunk_overlap=20))
        assert len(chunker.chunk([_make_doc()])) >= 1

    def test_strategy_field_is_recursive_header(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="recursive_header", chunk_size=100, chunk_overlap=20))
        assert all(c.strategy == "recursive_header" for c in chunker.chunk([_make_doc()]))

    def test_metadata_preserved(self) -> None:
        doc = _make_doc()
        chunker = Chunker(Settings(chunk_strategy="recursive_header", chunk_size=100, chunk_overlap=20))
        for c in chunker.chunk([doc]):
            assert c.doc_id == doc.doc_id
            assert c.section_heading == doc.section_heading

    def test_keeps_paragraph_content(self) -> None:
        text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird one."
        chunker = Chunker(Settings(chunk_strategy="recursive_header", chunk_size=500, chunk_overlap=0))
        combined = " ".join(c.text for c in chunker.chunk([_make_doc(text=text)]))
        assert "First paragraph" in combined
        assert "Second paragraph" in combined
        assert "Third one" in combined

    def test_chunk_indices_sequential(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="recursive_header", chunk_size=100, chunk_overlap=20))
        chunks = chunker.chunk([_make_doc()])
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_empty_doc_produces_no_chunks(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="recursive_header", chunk_size=100, chunk_overlap=0))
        assert chunker.chunk([_make_doc(text="")]) == []


class TestSemanticChunker:
    def _mock_embeddings(self, vectors: list[list[float]]) -> MagicMock:
        resp = MagicMock()
        resp.data = [MagicMock(embedding=v) for v in vectors]
        return resp

    def test_splits_at_topic_boundary(self) -> None:
        """First sentence orthogonal to last two → two chunks."""
        chunker = Chunker(Settings(
            chunk_strategy="semantic",
            chunk_size=1000,
            chunk_overlap=0,
            semantic_breakpoint_percentile=50.0,
        ))
        doc = _make_doc(
            text="Machine learning is powerful. The weather is sunny today. Rain is expected tomorrow."
        )
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]]  # ML vs weather+weather

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.embeddings.create.return_value = self._mock_embeddings(vecs)
            chunks = chunker.chunk([doc])

        assert len(chunks) == 2
        assert all(c.strategy == "semantic" for c in chunks)

    def test_keeps_similar_sentences_together(self) -> None:
        """Identical embeddings → similarity=1, distance=0 → no breakpoints → one chunk."""
        chunker = Chunker(Settings(
            chunk_strategy="semantic",
            chunk_size=1000,
            chunk_overlap=0,
            semantic_breakpoint_percentile=50.0,
        ))
        doc = _make_doc(
            text="Dogs are friendly pets. Cats are also great companions. Fish are nice too."
        )
        vecs = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.embeddings.create.return_value = self._mock_embeddings(vecs)
            chunks = chunker.chunk([doc])

        assert len(chunks) == 1

    def test_single_sentence_skips_api(self) -> None:
        """Single-sentence docs skip embeddings API entirely."""
        chunker = Chunker(Settings(chunk_strategy="semantic", chunk_size=1000, chunk_overlap=0))
        doc = _make_doc(text="Only one sentence here.")

        with patch("openai.OpenAI") as MockClient:
            chunks = chunker.chunk([doc])
            MockClient.return_value.embeddings.create.assert_not_called()

        assert len(chunks) == 1
        assert chunks[0].strategy == "semantic"

    def test_metadata_preserved(self) -> None:
        chunker = Chunker(Settings(
            chunk_strategy="semantic",
            chunk_size=1000,
            chunk_overlap=0,
            semantic_breakpoint_percentile=50.0,
        ))
        doc = _make_doc(
            text="Machine learning is powerful. The weather is sunny today. Rain is expected tomorrow."
        )
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]]

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.embeddings.create.return_value = self._mock_embeddings(vecs)
            chunks = chunker.chunk([doc])

        for c in chunks:
            assert c.doc_id == doc.doc_id
            assert c.source_path == doc.source_path
            assert c.section_heading == doc.section_heading

    def test_empty_doc_produces_no_chunks(self) -> None:
        chunker = Chunker(Settings(chunk_strategy="semantic", chunk_size=1000, chunk_overlap=0))
        with patch("openai.OpenAI"):
            assert chunker.chunk([_make_doc(text="")]) == []

    def test_chunk_indices_sequential(self) -> None:
        chunker = Chunker(Settings(
            chunk_strategy="semantic",
            chunk_size=1000,
            chunk_overlap=0,
            semantic_breakpoint_percentile=50.0,
        ))
        doc = _make_doc(
            text="Machine learning is powerful. The weather is sunny today. Rain is expected tomorrow."
        )
        vecs = [[1.0, 0.0], [0.0, 1.0], [0.1, 0.9]]

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.embeddings.create.return_value = self._mock_embeddings(vecs)
            chunks = chunker.chunk([doc])

        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
