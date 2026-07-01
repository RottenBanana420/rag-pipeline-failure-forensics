"""Unit tests for make_vector_store factory — TDD (written before implementation)."""

import pytest


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

    def test_chroma_provider_returns_chroma_vector_store(self, chroma_settings):
        from src.retrieval.vector_store import ChromaVectorStore, make_vector_store

        result = make_vector_store(chroma_settings)

        assert isinstance(result, ChromaVectorStore)

    def test_qdrant_provider_raises_not_implemented(self, qdrant_settings):
        from src.retrieval.vector_store import make_vector_store

        with pytest.raises(NotImplementedError, match="not yet implemented"):
            make_vector_store(qdrant_settings)

    def test_unknown_provider_raises_value_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.vector_store import make_vector_store

        s = Settings()
        # Force an invalid provider by bypassing the Literal validator
        object.__setattr__(s, "vector_store_provider", "unsupported_provider")

        with pytest.raises(ValueError, match="unsupported_provider"):
            make_vector_store(s)

    def test_unknown_provider_error_lists_valid_providers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.vector_store import make_vector_store

        s = Settings()
        object.__setattr__(s, "vector_store_provider", "bogus")

        with pytest.raises(ValueError) as exc_info:
            make_vector_store(s)

        msg = str(exc_info.value)
        assert "chroma" in msg
        assert "qdrant" in msg

    def test_result_satisfies_vector_store_protocol(self, chroma_settings):
        from src.retrieval.vector_store import VectorStoreProtocol, make_vector_store

        result = make_vector_store(chroma_settings)

        assert isinstance(result, VectorStoreProtocol)

    def test_vector_store_alias_still_works(self, tmp_path, monkeypatch):
        """VectorStore backward-compat alias still resolves to ChromaVectorStore."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.vector_store import ChromaVectorStore, VectorStore

        assert VectorStore is ChromaVectorStore

        s = Settings()
        vs = VectorStore(s)
        assert isinstance(vs, ChromaVectorStore)
