"""Tests for ChromaVectorStore startup dimension guard (TDD — written before implementation)."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.embedder import EmbedderProtocol


def _make_embedder(provider_id: str, dimensions: int) -> EmbedderProtocol:
    """Return a mock that satisfies EmbedderProtocol with given provider_id and dimensions."""
    mock = MagicMock(spec=EmbedderProtocol)
    mock.provider_id = provider_id
    mock.dimensions = dimensions
    mock.embed.return_value = []
    return mock  # type: ignore[return-value]


class TestDimensionGuardNewCollection:
    def test_new_collection_writes_provider_metadata(self, settings, tmp_path, monkeypatch):
        """Creating a new collection stores embedding_provider and embedding_dimensions."""
        import chromadb

        from src.retrieval.vector_store import COLLECTION_NAME, ChromaVectorStore

        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_guard_new"))
        from src.config import Settings
        settings2 = Settings()

        embedder = _make_embedder("openai/text-embedding-3-small", 1536)
        ChromaVectorStore(settings2, embedder=embedder)

        client = chromadb.PersistentClient(path=settings2.chroma_persist_dir_str)
        col = client.get_collection(COLLECTION_NAME)
        meta = col.metadata

        assert meta is not None
        assert meta["embedding_provider"] == "openai/text-embedding-3-small"
        assert meta["embedding_dimensions"] == 1536

    def test_embedder_is_required(self, settings):
        """Constructing without an embedder must fail, not silently skip the guard."""
        from src.retrieval.vector_store import ChromaVectorStore

        with pytest.raises(TypeError):
            ChromaVectorStore(settings)  # type: ignore[call-arg]


class TestDimensionGuardExistingCollection:
    def test_matching_provider_passes_silently(self, settings, tmp_path, monkeypatch):
        """Re-opening a collection with the same provider raises no error."""
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_guard_match"))
        from src.config import Settings
        from src.retrieval.vector_store import ChromaVectorStore

        settings2 = Settings()
        embedder = _make_embedder("openai/text-embedding-3-small", 1536)

        # First open — creates collection and writes metadata
        ChromaVectorStore(settings2, embedder=embedder)

        # Second open — reads metadata, same provider → no error
        ChromaVectorStore(settings2, embedder=embedder)

    def test_different_provider_raises_value_error(self, tmp_path, monkeypatch):
        """Re-opening with a different provider raises ValueError with correct message."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_guard_mismatch"))
        from src.config import Settings
        from src.retrieval.vector_store import ChromaVectorStore

        settings2 = Settings()
        embedder_a = _make_embedder("openai/text-embedding-3-small", 1536)
        embedder_b = _make_embedder("sentence-transformers/all-minilm-l6-v2", 384)

        # First open — creates collection with provider A
        ChromaVectorStore(settings2, embedder=embedder_a)

        # Second open — different provider B should raise
        with pytest.raises(ValueError):
            ChromaVectorStore(settings2, embedder=embedder_b)

    def test_different_provider_error_message_exact_format(self, tmp_path, monkeypatch):
        """ValueError message matches the exact required format."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma_guard_msg"))
        from src.config import Settings
        from src.retrieval.vector_store import ChromaVectorStore

        settings2 = Settings()
        embedder_a = _make_embedder("openai/text-embedding-3-small", 1536)
        embedder_b = _make_embedder("sentence-transformers/all-minilm-l6-v2", 384)

        ChromaVectorStore(settings2, embedder=embedder_a)

        expected_msg = (
            "Collection was indexed with 'openai/text-embedding-3-small' (1536 dims). "
            "Current config is 'sentence-transformers/all-minilm-l6-v2' (384 dims). "
            "Delete 'data/chroma/' and re-index to switch providers."
        )

        with pytest.raises(ValueError) as exc_info:
            ChromaVectorStore(settings2, embedder=embedder_b)

        assert str(exc_info.value) == expected_msg
