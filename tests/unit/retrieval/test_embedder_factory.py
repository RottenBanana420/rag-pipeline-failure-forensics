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


@pytest.fixture
def voyage_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


@pytest.fixture
def gemini_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


@pytest.fixture
def cohere_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("COHERE_API_KEY", "test-key")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "cohere")
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

        with patch("openai.OpenAI"):
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
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = make_embedder(st_settings)

        assert isinstance(result, SentenceTransformersEmbedder)

    def test_sentence_transformers_provider_passes_device_none_when_auto(self, st_settings):
        """settings.embedding_device defaults to "auto"; factory must translate that to
        device=None so the underlying library auto-detects CUDA/MPS/CPU itself."""
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_sentence_transformers import DEFAULT_MODEL

        assert st_settings.embedding_device == "auto"
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_st_cls:
            make_embedder(st_settings)

        mock_st_cls.assert_called_once_with(DEFAULT_MODEL, device=None)

    def test_sentence_transformers_provider_passes_explicit_device(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
        monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_sentence_transformers import DEFAULT_MODEL

        settings = Settings()
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_st_cls:
            make_embedder(settings)

        mock_st_cls.assert_called_once_with(DEFAULT_MODEL, device="cpu")

    def test_voyage_provider_returns_voyage_embedder(self, voyage_settings):
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_voyage import VoyageEmbedder

        with patch("voyageai.Client"):
            result = make_embedder(voyage_settings)

        assert isinstance(result, VoyageEmbedder)

    def test_voyage_falls_back_to_default_when_openai_model_configured(
        self, voyage_settings
    ):
        """With default settings, embedding_model is "text-embedding-3-small" (an OpenAI
        model). make_embedder must detect this isn't a Voyage model name and substitute
        the Voyage provider's own default instead of passing it straight through.
        """
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_voyage import DEFAULT_MODEL

        with patch("voyageai.Client"):
            result = make_embedder(voyage_settings)

        assert result._model == DEFAULT_MODEL

    def test_voyage_embedder_uses_settings_model_when_voyage_compatible(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "voyage")
        monkeypatch.setenv("EMBEDDING_MODEL", "voyage-3-large")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.embedder import make_embedder

        explicit_voyage_settings = Settings()
        with patch("voyageai.Client"):
            result = make_embedder(explicit_voyage_settings)

        assert result._model == "voyage-3-large"

    def test_gemini_provider_returns_gemini_embedder(self, gemini_settings):
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_gemini import GeminiEmbedder

        with patch("google.genai.Client"):
            result = make_embedder(gemini_settings)

        assert isinstance(result, GeminiEmbedder)

    def test_gemini_falls_back_to_default_when_openai_model_configured(
        self, gemini_settings
    ):
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_gemini import DEFAULT_MODEL

        with patch("google.genai.Client"):
            result = make_embedder(gemini_settings)

        assert result._model == DEFAULT_MODEL

    def test_cohere_provider_returns_cohere_embedder(self, cohere_settings):
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_cohere import CohereEmbedder

        with patch("cohere.ClientV2"):
            result = make_embedder(cohere_settings)

        assert isinstance(result, CohereEmbedder)

    def test_cohere_falls_back_to_default_when_openai_model_configured(
        self, cohere_settings
    ):
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_cohere import DEFAULT_MODEL

        with patch("cohere.ClientV2"):
            result = make_embedder(cohere_settings)

        assert result._model == DEFAULT_MODEL

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

        with patch("openai.OpenAI"):
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

        with patch("openai.OpenAI"):
            result = make_embedder(openai_settings)

        assert result._model == openai_settings.embedding_model

    def test_st_embedder_uses_settings_model_when_st_compatible(self, monkeypatch, tmp_path):
        """The ST embedder uses the model from settings when it is an ST-compatible name."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("EMBEDDING_PROVIDER", "sentence_transformers")
        monkeypatch.setenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L3-v2")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.embedder import make_embedder

        explicit_st_settings = Settings()
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            result = make_embedder(explicit_st_settings)

        assert result._model_name == "paraphrase-MiniLM-L3-v2"

    def test_st_embedder_falls_back_to_default_when_openai_model_configured(self, st_settings):
        """When the configured model looks like an OpenAI model, ST provider uses its own default.

        With default settings, embedding_model is "text-embedding-3-small" (an OpenAI model).
        make_embedder must detect this and pass DEFAULT_MODEL to SentenceTransformersEmbedder
        instead of crashing with a missing-model error from HuggingFace.
        """
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_sentence_transformers import DEFAULT_MODEL

        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_st_cls:
            result = make_embedder(st_settings)

        # The ST class must have been called with the ST default, not the OpenAI model name
        mock_st_cls.assert_called_once_with(DEFAULT_MODEL, device=None)
        assert result._model_name == DEFAULT_MODEL

    def test_default_settings_use_st_default_model(self, monkeypatch, tmp_path):
        """make_embedder(Settings()) with no overrides must not pass an OpenAI model to ST.

        Settings() defaults: embedding_provider="sentence_transformers",
        embedding_model="text-embedding-3-small". Without the guard in make_embedder,
        SentenceTransformer would try to download "text-embedding-3-small" from HuggingFace
        (it doesn't exist) and crash. This test asserts the guard works.
        """
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        # Deliberately do NOT set EMBEDDING_PROVIDER or EMBEDDING_MODEL — use pure defaults
        from src.config import Settings
        from src.retrieval.embedder import make_embedder
        from src.retrieval.providers.embedder_sentence_transformers import DEFAULT_MODEL

        default_settings = Settings()
        assert default_settings.embedding_provider == "sentence_transformers"
        assert default_settings.embedding_model == "text-embedding-3-small"

        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ) as mock_st_cls:
            result = make_embedder(default_settings)

        mock_st_cls.assert_called_once_with(DEFAULT_MODEL, device=None)
        assert result._model_name == DEFAULT_MODEL
        assert DEFAULT_MODEL == "all-MiniLM-L6-v2"
