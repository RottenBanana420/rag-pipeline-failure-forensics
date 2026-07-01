"""Unit tests for OpenAIEmbedder — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(n: int) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[float(i)] * 4) for i in range(n)]
    return resp


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestOpenAIEmbedder:
    def test_importable(self):
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder  # noqa: F401

    def test_embed_returns_vector_per_text(self, settings):
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.embeddings.create.return_value = _mock_response(3)
            result = OpenAIEmbedder(settings).embed(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.0, 0.0, 0.0, 0.0]

    def test_embed_empty_input_returns_empty(self, settings):
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            result = OpenAIEmbedder(settings).embed([])

        assert result == []

    def test_embed_batches_large_input(self, settings):
        from src.retrieval.providers.embedder_openai import BATCH_SIZE, OpenAIEmbedder

        texts = ["text"] * (BATCH_SIZE + 1)

        with patch("src.retrieval.providers.embedder_openai.OpenAI") as MockOpenAI:
            mock_create = MockOpenAI.return_value.embeddings.create
            mock_create.side_effect = [_mock_response(BATCH_SIZE), _mock_response(1)]
            result = OpenAIEmbedder(settings).embed(texts)

        assert mock_create.call_count == 2
        assert len(result) == BATCH_SIZE + 1

    def test_embed_passes_model_from_settings(self, settings):
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI") as MockOpenAI:
            mock_create = MockOpenAI.return_value.embeddings.create
            mock_create.return_value = _mock_response(1)
            OpenAIEmbedder(settings).embed(["hello"])

        assert mock_create.call_args.kwargs["model"] == settings.embedding_model

    def test_provider_id_includes_model_name(self, settings):
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            embedder = OpenAIEmbedder(settings)

        assert embedder.provider_id == f"openai/{settings.embedding_model}"

    def test_dimensions_text_embedding_3_small(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        from src.config import Settings

        s = Settings()
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            embedder = OpenAIEmbedder(s)

        assert embedder.dimensions == 1536

    def test_dimensions_text_embedding_3_large(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        from src.config import Settings

        s = Settings()
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            embedder = OpenAIEmbedder(s)

        assert embedder.dimensions == 3072

    def test_dimensions_unknown_model_fallback(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        from src.config import Settings

        s = Settings()
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            embedder = OpenAIEmbedder(s)
            # Override model to simulate unknown
            embedder._model = "unknown-model"

        assert embedder.dimensions == 1536

    def test_satisfies_embedder_protocol(self, settings):
        from src.retrieval.embedder import EmbedderProtocol
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            embedder = OpenAIEmbedder(settings)

        assert isinstance(embedder, EmbedderProtocol)

    def test_openai_imported_lazily(self):
        """OpenAI should not be imported at module top-level in the provider file."""
        import importlib
        import sys

        # Remove openai from sys.modules to test lazy import
        openai_mod = sys.modules.pop("openai", None)
        try:
            # Re-import the provider module — this should NOT trigger openai import
            if "src.retrieval.providers.embedder_openai" in sys.modules:
                del sys.modules["src.retrieval.providers.embedder_openai"]
            import src.retrieval.providers.embedder_openai  # noqa: F401

            # openai should still not be in sys.modules (lazy)
            assert "openai" not in sys.modules
        finally:
            # Restore openai module
            if openai_mod is not None:
                sys.modules["openai"] = openai_mod
