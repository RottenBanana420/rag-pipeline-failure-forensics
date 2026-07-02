"""Unit tests for VoyageEmbedder — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


def _mock_result(n: int) -> MagicMock:
    result = MagicMock()
    result.embeddings = [[float(i)] * 4 for i in range(n)]
    return result


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "voyage-3.5")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestVoyageEmbedder:
    def test_importable(self):
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder  # noqa: F401

    def test_embed_returns_vector_per_text(self, settings):
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.embed.return_value = _mock_result(3)
            result = VoyageEmbedder(settings).embed(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.0, 0.0, 0.0, 0.0]

    def test_embed_empty_input_returns_empty(self, settings):
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client"):
            result = VoyageEmbedder(settings).embed([])

        assert result == []

    def test_embed_batches_large_input(self, settings):
        from src.retrieval.providers.embedder_voyage import BATCH_SIZE, VoyageEmbedder

        texts = ["text"] * (BATCH_SIZE + 1)

        with patch("voyageai.Client") as MockClient:
            mock_embed = MockClient.return_value.embed
            mock_embed.side_effect = [_mock_result(BATCH_SIZE), _mock_result(1)]
            result = VoyageEmbedder(settings).embed(texts)

        assert mock_embed.call_count == 2
        assert len(result) == BATCH_SIZE + 1

    def test_embed_passes_model_from_settings(self, settings):
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client") as MockClient:
            mock_embed = MockClient.return_value.embed
            mock_embed.return_value = _mock_result(1)
            VoyageEmbedder(settings).embed(["hello"])

        assert mock_embed.call_args.kwargs["model"] == settings.embedding_model

    def test_provider_id_includes_model_name(self, settings):
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client"):
            embedder = VoyageEmbedder(settings)

        assert embedder.provider_id == f"voyage/{settings.embedding_model}"

    def test_dimensions_voyage_3_5(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("EMBEDDING_MODEL", "voyage-3.5")
        from src.config import Settings

        s = Settings()
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client"):
            embedder = VoyageEmbedder(s)

        assert embedder.dimensions == 1024

    def test_dimensions_voyage_large_2_legacy(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        monkeypatch.setenv("EMBEDDING_MODEL", "voyage-large-2")
        from src.config import Settings

        s = Settings()
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client"):
            embedder = VoyageEmbedder(s)

        assert embedder.dimensions == 1536

    def test_dimensions_unknown_model_fallback(self, settings):
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client"):
            embedder = VoyageEmbedder(settings)
            embedder._model = "unknown-model"

        assert embedder.dimensions == 1024

    def test_satisfies_embedder_protocol(self, settings):
        from src.retrieval.embedder import EmbedderProtocol
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client"):
            embedder = VoyageEmbedder(settings)

        assert isinstance(embedder, EmbedderProtocol)

    def test_voyageai_imported_lazily(self):
        """voyageai should not be imported at module top-level in the provider file."""
        import sys

        voyageai_mod = sys.modules.pop("voyageai", None)
        try:
            if "src.retrieval.providers.embedder_voyage" in sys.modules:
                del sys.modules["src.retrieval.providers.embedder_voyage"]
            import src.retrieval.providers.embedder_voyage  # noqa: F401

            assert "voyageai" not in sys.modules
        finally:
            if voyageai_mod is not None:
                sys.modules["voyageai"] = voyageai_mod
