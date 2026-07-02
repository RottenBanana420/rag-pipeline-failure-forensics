"""Unit tests for CohereEmbedder — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_response(n: int) -> MagicMock:
    response = MagicMock()
    response.embeddings.float_ = [[float(i)] * 4 for i in range(n)]
    return response


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "embed-v4.0")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestCohereEmbedder:
    def test_importable(self):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder  # noqa: F401

    def test_embed_returns_vector_per_text(self, settings):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2") as MockClient:
            MockClient.return_value.v2.embed.return_value = _mock_response(3)
            result = CohereEmbedder(settings).embed(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.0, 0.0, 0.0, 0.0]

    def test_embed_empty_input_returns_empty(self, settings):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2"):
            result = CohereEmbedder(settings).embed([])

        assert result == []

    def test_embed_batches_large_input(self, settings):
        from src.retrieval.providers.embedder_cohere import BATCH_SIZE, CohereEmbedder

        texts = ["text"] * (BATCH_SIZE + 1)

        with patch("cohere.ClientV2") as MockClient:
            mock_embed = MockClient.return_value.v2.embed
            mock_embed.side_effect = [_mock_response(BATCH_SIZE), _mock_response(1)]
            result = CohereEmbedder(settings).embed(texts)

        assert mock_embed.call_count == 2
        assert len(result) == BATCH_SIZE + 1

    def test_embed_passes_model_from_settings(self, settings):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2") as MockClient:
            mock_embed = MockClient.return_value.v2.embed
            mock_embed.return_value = _mock_response(1)
            CohereEmbedder(settings).embed(["hello"])

        assert mock_embed.call_args.kwargs["model"] == settings.embedding_model

    def test_embed_passes_input_type_search_document(self, settings):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2") as MockClient:
            mock_embed = MockClient.return_value.v2.embed
            mock_embed.return_value = _mock_response(1)
            CohereEmbedder(settings).embed(["hello"])

        assert mock_embed.call_args.kwargs["input_type"] == "search_document"

    def test_provider_id_includes_model_name(self, settings):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2"):
            embedder = CohereEmbedder(settings)

        assert embedder.provider_id == f"cohere/{settings.embedding_model}"

    def test_dimensions_embed_v4(self, settings):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2"):
            embedder = CohereEmbedder(settings)

        assert embedder.dimensions == 1024

    def test_dimensions_embed_english_light_v3(self, monkeypatch, tmp_path):
        monkeypatch.setenv("COHERE_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("EMBEDDING_MODEL", "embed-english-light-v3.0")
        from src.config import Settings

        s = Settings()
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2"):
            embedder = CohereEmbedder(s)

        assert embedder.dimensions == 384

    def test_dimensions_unknown_model_fallback(self, settings):
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2"):
            embedder = CohereEmbedder(settings)
            embedder._model = "unknown-model"

        assert embedder.dimensions == 1024

    def test_satisfies_embedder_protocol(self, settings):
        from src.retrieval.embedder import EmbedderProtocol
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2"):
            embedder = CohereEmbedder(settings)

        assert isinstance(embedder, EmbedderProtocol)

    def test_cohere_imported_lazily(self):
        """cohere should not be imported at module top-level in the provider file."""
        import sys

        cohere_mod = sys.modules.pop("cohere", None)
        try:
            if "src.retrieval.providers.embedder_cohere" in sys.modules:
                del sys.modules["src.retrieval.providers.embedder_cohere"]
            import src.retrieval.providers.embedder_cohere  # noqa: F401

            assert "cohere" not in sys.modules
        finally:
            if cohere_mod is not None:
                sys.modules["cohere"] = cohere_mod
