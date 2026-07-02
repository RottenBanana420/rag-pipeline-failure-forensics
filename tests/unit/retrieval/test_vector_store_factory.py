"""Unit tests for make_vector_store factory — TDD (written before implementation)."""

from unittest.mock import MagicMock

import pytest

from src.retrieval.embedder import EmbedderProtocol


def _make_mock_embedder(
    provider_id: str = "openai/text-embedding-3-small",
    dimensions: int = 1536,
) -> EmbedderProtocol:
    mock = MagicMock(spec=EmbedderProtocol)
    mock.provider_id = provider_id
    mock.dimensions = dimensions
    mock.embed.return_value = []
    return mock  # type: ignore[return-value]


@pytest.fixture
def mock_embedder() -> EmbedderProtocol:
    return _make_mock_embedder()


@pytest.fixture
def chroma_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "chroma")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


@pytest.fixture
def qdrant_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("VECTOR_STORE_PROVIDER", "qdrant")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestMakeVectorStore:
    def test_importable(self):
        from src.retrieval.vector_store import make_vector_store  # noqa: F401

    def test_chroma_provider_returns_chroma_vector_store(self, chroma_settings, mock_embedder):
        from src.retrieval.vector_store import ChromaVectorStore, make_vector_store

        result = make_vector_store(chroma_settings, mock_embedder)

        assert isinstance(result, ChromaVectorStore)

    def test_qdrant_provider_raises_not_implemented(self, qdrant_settings, mock_embedder):
        from src.retrieval.vector_store import make_vector_store

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            make_vector_store(qdrant_settings, mock_embedder)

    def test_unknown_provider_raises_value_error(self, monkeypatch, tmp_path, mock_embedder):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.vector_store import make_vector_store

        s = Settings()
        # Force an invalid provider by bypassing the Literal validator
        object.__setattr__(s, "vector_store_provider", "unsupported_provider")

        with pytest.raises(ValueError, match="unsupported_provider"):
            make_vector_store(s, mock_embedder)

    def test_unknown_provider_error_lists_valid_providers(self, monkeypatch, tmp_path, mock_embedder):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.vector_store import make_vector_store

        s = Settings()
        object.__setattr__(s, "vector_store_provider", "bogus")

        with pytest.raises(ValueError) as exc_info:
            make_vector_store(s, mock_embedder)

        msg = str(exc_info.value)
        assert "chroma" in msg
        assert "qdrant" in msg

    def test_result_satisfies_vector_store_protocol(self, chroma_settings, mock_embedder):
        from src.retrieval.vector_store import VectorStoreProtocol, make_vector_store

        result = make_vector_store(chroma_settings, mock_embedder)

        assert isinstance(result, VectorStoreProtocol)

    def test_vector_store_alias_still_works(self, tmp_path, monkeypatch, mock_embedder):
        """VectorStore backward-compat alias still resolves to ChromaVectorStore."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.vector_store import ChromaVectorStore, VectorStore

        assert VectorStore is ChromaVectorStore

        s = Settings()
        vs = VectorStore(s, mock_embedder)
        assert isinstance(vs, ChromaVectorStore)
