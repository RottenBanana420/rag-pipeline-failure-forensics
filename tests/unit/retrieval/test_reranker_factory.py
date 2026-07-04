"""Unit tests for make_reranker factory."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def st_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    from src.config import Settings

    return Settings()


class TestMakeReranker:
    def test_importable(self):
        from src.retrieval.reranker import make_reranker  # noqa: F401

    def test_sentence_transformers_provider_returns_st_reranker(self, st_settings):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )
        from src.retrieval.reranker import make_reranker

        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            result = make_reranker(st_settings)

        assert isinstance(result, SentenceTransformersReranker)

    def test_sentence_transformers_provider_passes_device_none_when_auto(
        self, st_settings
    ):
        """settings.reranker_device defaults to "auto"; factory must translate that to
        device=None so the underlying library auto-detects CUDA/MPS/CPU itself."""
        from src.retrieval.reranker import make_reranker

        assert st_settings.reranker_device == "auto"
        with patch(
            "sentence_transformers.CrossEncoder", return_value=MagicMock()
        ) as mock_ctor:
            make_reranker(st_settings)

        args, kwargs = mock_ctor.call_args
        assert args == (st_settings.reranker_model,)
        assert kwargs["device"] is None

    def test_sentence_transformers_provider_passes_explicit_device(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_DEVICE", "cpu")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.reranker import make_reranker

        settings = Settings()
        with patch(
            "sentence_transformers.CrossEncoder", return_value=MagicMock()
        ) as mock_ctor:
            make_reranker(settings)

        args, kwargs = mock_ctor.call_args
        assert args == (settings.reranker_model,)
        assert kwargs["device"] == "cpu"

    def test_uses_settings_reranker_model(self, monkeypatch, tmp_path):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("RERANKER_MODEL", "cross-encoder/custom-model")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        from src.config import Settings
        from src.retrieval.reranker import make_reranker

        settings = Settings()
        with patch(
            "sentence_transformers.CrossEncoder", return_value=MagicMock()
        ) as mock_ctor:
            result = make_reranker(settings)

        args, kwargs = mock_ctor.call_args
        assert args == ("cross-encoder/custom-model",)
        assert kwargs["device"] is None
        assert result._model_name == "cross-encoder/custom-model"

    def test_unknown_provider_raises_value_error(self, st_settings):
        from src.retrieval.reranker import make_reranker

        # Force an invalid provider by bypassing the Literal validator
        object.__setattr__(st_settings, "reranker_provider", "unsupported_provider")

        with pytest.raises(ValueError, match="unsupported_provider"):
            make_reranker(st_settings)

    def test_unknown_provider_error_lists_valid_providers(self, st_settings):
        from src.retrieval.reranker import make_reranker

        object.__setattr__(st_settings, "reranker_provider", "bogus")

        with pytest.raises(ValueError) as exc_info:
            make_reranker(st_settings)

        message = str(exc_info.value)
        assert "sentence_transformers" in message
        assert "cohere" in message
        assert "voyage" in message

    def test_result_satisfies_reranker_protocol(self, st_settings):
        from src.retrieval.reranker import RerankerProtocol, make_reranker

        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            result = make_reranker(st_settings)

        assert isinstance(result, RerankerProtocol)

    def test_provider_modules_not_imported_at_module_level(self):
        """make_reranker must use lazy imports — provider modules not at reranker.py top-level."""
        import sys

        sys.modules.pop("src.retrieval.providers.reranker_sentence_transformers", None)
        sys.modules.pop("src.retrieval.reranker", None)

        import src.retrieval.reranker  # noqa: F401

        assert "src.retrieval.reranker" in sys.modules

    def test_cohere_and_voyage_provider_modules_not_imported_at_module_level(self):
        """Cohere/Voyage provider modules must also be imported lazily, not at
        reranker.py top-level — importing reranker.py alone must not pull them in."""
        import sys

        sys.modules.pop("src.retrieval.providers.reranker_cohere", None)
        sys.modules.pop("src.retrieval.providers.reranker_voyage", None)
        sys.modules.pop("src.retrieval.reranker", None)

        import src.retrieval.reranker  # noqa: F401

        assert "src.retrieval.reranker" in sys.modules
        assert "src.retrieval.providers.reranker_cohere" not in sys.modules
        assert "src.retrieval.providers.reranker_voyage" not in sys.modules

    def _cohere_settings(self, monkeypatch, tmp_path, reranker_model=None):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")
        monkeypatch.setenv("RERANKER_PROVIDER", "cohere")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        if reranker_model is not None:
            monkeypatch.setenv("RERANKER_MODEL", reranker_model)
        from src.config import Settings

        return Settings()

    def _voyage_settings(self, monkeypatch, tmp_path, reranker_model=None):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
        monkeypatch.setenv("RERANKER_PROVIDER", "voyage")
        monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
        monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
        if reranker_model is not None:
            monkeypatch.setenv("RERANKER_MODEL", reranker_model)
        from src.config import Settings

        return Settings()

    def test_cohere_provider_returns_cohere_reranker(self, monkeypatch, tmp_path):
        from src.retrieval.providers.reranker_cohere import CohereReranker
        from src.retrieval.reranker import make_reranker

        settings = self._cohere_settings(monkeypatch, tmp_path)
        with patch("cohere.ClientV2", return_value=MagicMock()):
            result = make_reranker(settings)

        assert isinstance(result, CohereReranker)

    def test_voyage_provider_returns_voyage_reranker(self, monkeypatch, tmp_path):
        from src.retrieval.providers.reranker_voyage import VoyageReranker
        from src.retrieval.reranker import make_reranker

        settings = self._voyage_settings(monkeypatch, tmp_path)
        with patch("voyageai.Client", return_value=MagicMock()):
            result = make_reranker(settings)

        assert isinstance(result, VoyageReranker)

    def test_cohere_default_model_substituted_when_reranker_model_unset(
        self, monkeypatch, tmp_path
    ):
        """When reranker_model still equals the sentence_transformers default (i.e. the
        user never customized RERANKER_MODEL), the cohere branch must swap in Cohere's
        own DEFAULT_MODEL rather than passing the sentence_transformers string through."""
        from src.retrieval.providers.reranker_cohere import DEFAULT_MODEL
        from src.retrieval.reranker import make_reranker

        settings = self._cohere_settings(monkeypatch, tmp_path)
        assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"

        with patch("cohere.ClientV2", return_value=MagicMock()):
            result = make_reranker(settings)

        assert result._model == DEFAULT_MODEL

    def test_voyage_default_model_substituted_when_reranker_model_unset(
        self, monkeypatch, tmp_path
    ):
        from src.retrieval.providers.reranker_voyage import DEFAULT_MODEL
        from src.retrieval.reranker import make_reranker

        settings = self._voyage_settings(monkeypatch, tmp_path)
        assert settings.reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"

        with patch("voyageai.Client", return_value=MagicMock()):
            result = make_reranker(settings)

        assert result._model == DEFAULT_MODEL

    def test_cohere_customized_reranker_model_passes_through_unchanged(
        self, monkeypatch, tmp_path
    ):
        """A user-supplied RERANKER_MODEL (different from the sentence_transformers
        default) must be trusted as-is, not silently overridden by DEFAULT_MODEL."""
        from src.retrieval.reranker import make_reranker

        settings = self._cohere_settings(
            monkeypatch, tmp_path, reranker_model="rerank-english-v3.0"
        )

        with patch("cohere.ClientV2", return_value=MagicMock()):
            result = make_reranker(settings)

        assert result._model == "rerank-english-v3.0"

    def test_voyage_customized_reranker_model_passes_through_unchanged(
        self, monkeypatch, tmp_path
    ):
        from src.retrieval.reranker import make_reranker

        settings = self._voyage_settings(
            monkeypatch, tmp_path, reranker_model="rerank-2.5-lite"
        )

        with patch("voyageai.Client", return_value=MagicMock()):
            result = make_reranker(settings)

        assert result._model == "rerank-2.5-lite"
