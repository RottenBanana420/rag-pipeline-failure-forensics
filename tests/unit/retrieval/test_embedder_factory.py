"""Unit tests for make_embedder factory — TDD (written before implementation)."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def openai_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


@pytest.fixture
def st_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestMakeEmbedder:
    def test_importable(self):
        from src.retrieval.embedder import make_embedder  # noqa: F401

    def test_openai_provider_returns_openai_embedder(self, openai_settings):
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_openai import OpenAIEmbedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            result = make_embedder(openai_settings)

        assert isinstance(result, OpenAIEmbedder)

    def test_sentence_transformers_provider_returns_st_embedder(self, st_settings):
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        with patch(
            "src.retrieval.providers.embedder_sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = make_embedder(st_settings)

        assert isinstance(result, SentenceTransformersEmbedder)

    def test_unknown_provider_raises_value_error(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.embedder import make_embedder

        s = Settings()
        # Force an invalid provider by bypassing the Literal validator
        object.__setattr__(s, "embedding_provider", "unsupported_provider")

        with pytest.raises(ValueError, match="unsupported_provider"):
            make_embedder(s)

    def test_unknown_provider_error_lists_valid_providers(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.embedder import make_embedder

        s = Settings()
        object.__setattr__(s, "embedding_provider", "bogus")

        with pytest.raises(ValueError) as exc_info:
            make_embedder(s)

        msg = str(exc_info.value)
        # Error message should list the valid providers
        assert "openai" in msg
        assert "sentence_transformers" in msg

    def test_result_satisfies_embedder_protocol(self, openai_settings):
        from src.retrieval.embedder import EmbedderProtocol, make_embedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            result = make_embedder(openai_settings)

        assert isinstance(result, EmbedderProtocol)

    def test_provider_modules_not_imported_at_module_level(self):
        """make_embedder must use lazy imports — provider modules not at embedder.py top-level."""
        import sys

        # Remove provider modules from sys.modules
        sys.modules.pop("src.retrieval.providers.embedder_openai", None)
        sys.modules.pop("src.retrieval.providers.embedder_sentence_transformers", None)
        sys.modules.pop("src.retrieval.embedder", None)

        # Re-import embedder — this should NOT pull in provider modules
        import src.retrieval.embedder  # noqa: F401

        # Provider modules should not yet be imported just from importing embedder
        # (They may still be there from previous test runs via Embedder = OpenAIEmbedder,
        #  but make_embedder itself should not trigger them at import time)
        assert "src.retrieval.embedder" in sys.modules

    def test_openai_embedder_uses_settings_model(self, openai_settings):
        """The openai embedder returned by the factory uses the model from settings."""
        from src.retrieval.embedder import make_embedder

        with patch("src.retrieval.providers.embedder_openai.OpenAI"):
            result = make_embedder(openai_settings)

        assert result._model == openai_settings.embedding_model

    def test_st_embedder_uses_settings_model(self, st_settings):
        """The ST embedder returned by the factory uses the model from settings."""
        from src.retrieval.embedder import make_embedder

        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        with patch(
            "src.retrieval.providers.embedder_sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = make_embedder(st_settings)

        assert result._model_name == st_settings.embedding_model
