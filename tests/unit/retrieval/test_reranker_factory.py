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

    def test_sentence_transformers_provider_passes_device_none_when_auto(self, st_settings):
        """settings.reranker_device defaults to "auto"; factory must translate that to
        device=None so the underlying library auto-detects CUDA/MPS/CPU itself."""
        from src.retrieval.reranker import make_reranker

        assert st_settings.reranker_device == "auto"
        with patch(
            "sentence_transformers.CrossEncoder", return_value=MagicMock()
        ) as mock_ctor:
            make_reranker(st_settings)

        mock_ctor.assert_called_once_with(st_settings.reranker_model, device=None)

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

        mock_ctor.assert_called_once_with(settings.reranker_model, device="cpu")

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

        mock_ctor.assert_called_once_with("cross-encoder/custom-model", device=None)
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

        assert "sentence_transformers" in str(exc_info.value)

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
