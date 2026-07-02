"""Unit tests for GeminiEmbedder — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_result(n: int) -> MagicMock:
    result = MagicMock()
    result.embeddings = [MagicMock(values=[float(i)] * 4) for i in range(n)]
    return result


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "gemini-embedding-001")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestGeminiEmbedder:
    def test_importable(self):
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder  # noqa: F401

    def test_embed_returns_vector_per_text(self, settings):
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client") as MockClient:
            MockClient.return_value.models.embed_content.return_value = _mock_result(3)
            result = GeminiEmbedder(settings).embed(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.0, 0.0, 0.0, 0.0]

    def test_embed_empty_input_returns_empty(self, settings):
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client"):
            result = GeminiEmbedder(settings).embed([])

        assert result == []

    def test_embed_batches_large_input(self, settings):
        from src.retrieval.providers.embedder_gemini import BATCH_SIZE, GeminiEmbedder

        texts = ["text"] * (BATCH_SIZE + 1)

        with patch("google.genai.Client") as MockClient:
            mock_embed_content = MockClient.return_value.models.embed_content
            mock_embed_content.side_effect = [_mock_result(BATCH_SIZE), _mock_result(1)]
            result = GeminiEmbedder(settings).embed(texts)

        assert mock_embed_content.call_count == 2
        assert len(result) == BATCH_SIZE + 1

    def test_embed_passes_model_from_settings(self, settings):
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client") as MockClient:
            mock_embed_content = MockClient.return_value.models.embed_content
            mock_embed_content.return_value = _mock_result(1)
            GeminiEmbedder(settings).embed(["hello"])

        assert mock_embed_content.call_args.kwargs["model"] == settings.embedding_model

    def test_provider_id_includes_model_name(self, settings):
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client"):
            embedder = GeminiEmbedder(settings)

        assert embedder.provider_id == f"gemini/{settings.embedding_model}"

    def test_dimensions_gemini_embedding_001(self, settings):
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client"):
            embedder = GeminiEmbedder(settings)

        assert embedder.dimensions == 3072

    def test_dimensions_unknown_model_fallback(self, settings):
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client"):
            embedder = GeminiEmbedder(settings)
            embedder._model = "unknown-model"

        assert embedder.dimensions == 3072

    def test_satisfies_embedder_protocol(self, settings):
        from src.retrieval.embedder import EmbedderProtocol
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client"):
            embedder = GeminiEmbedder(settings)

        assert isinstance(embedder, EmbedderProtocol)

    def test_genai_imported_lazily(self):
        """google.genai should not be imported at module top-level in the provider file."""
        import sys

        genai_mod = sys.modules.pop("google.genai", None)
        try:
            if "src.retrieval.providers.embedder_gemini" in sys.modules:
                del sys.modules["src.retrieval.providers.embedder_gemini"]
            import src.retrieval.providers.embedder_gemini  # noqa: F401

            assert "google.genai" not in sys.modules
        finally:
            if genai_mod is not None:
                sys.modules["google.genai"] = genai_mod
