"""Unit tests for src/config.py.

Tests instantiate Settings() directly so monkeypatch env changes take effect
(the module-level `settings` singleton is already frozen at import time).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


class TestSettingsDefaults:
    def test_embedding_model_default(self, clean_env):
        from src.config import Settings

        assert Settings().embedding_model == "text-embedding-3-small"

    def test_dense_top_k_default(self, clean_env):
        from src.config import Settings

        assert Settings().dense_top_k == 10

    def test_sparse_top_k_default(self, clean_env):
        from src.config import Settings

        assert Settings().sparse_top_k == 10

    def test_rerank_top_n_default(self, clean_env):
        from src.config import Settings

        assert Settings().rerank_top_n == 5

    def test_dense_weight_default(self, clean_env):
        from src.config import Settings

        assert Settings().dense_weight == pytest.approx(0.7)

    def test_sparse_weight_default(self, clean_env):
        from src.config import Settings

        assert Settings().sparse_weight == pytest.approx(0.3)

    def test_dedup_threshold_default(self, clean_env):
        from src.config import Settings

        assert Settings().dedup_threshold == pytest.approx(0.95)

    def test_log_level_default(self, clean_env):
        from src.config import Settings

        assert Settings().log_level == "INFO"

    def test_chroma_persist_dir_default(self, clean_env):
        from src.config import Settings

        assert Settings().chroma_persist_dir == Path("./data/chroma")

    def test_openai_api_key_default_empty(self, clean_env):
        from src.config import Settings

        assert Settings().openai_api_key == ""

    def test_embedding_device_default(self, clean_env):
        from src.config import Settings

        assert Settings().embedding_device == "auto"

    def test_rerank_candidate_pool_default(self, clean_env):
        from src.config import Settings

        assert Settings().rerank_candidate_pool == 20

    def test_reranking_enabled_default(self, clean_env):
        from src.config import Settings

        assert Settings().reranking_enabled is True

    def test_reranker_provider_default(self, clean_env):
        from src.config import Settings

        assert Settings().reranker_provider == "sentence_transformers"

    def test_reranker_model_default(self, clean_env):
        from src.config import Settings

        assert Settings().reranker_model == "cross-encoder/ms-marco-MiniLM-L6-v2"

    def test_reranker_device_default(self, clean_env):
        from src.config import Settings

        assert Settings().reranker_device == "auto"


class TestSettingsOverrides:
    def test_log_level_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        from src.config import Settings

        assert Settings().log_level == "DEBUG"

    def test_dense_top_k_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("DENSE_TOP_K", "20")
        from src.config import Settings

        assert Settings().dense_top_k == 20

    def test_openai_api_key_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-abc123")
        from src.config import Settings

        assert Settings().openai_api_key == "sk-test-abc123"

    def test_chroma_persist_dir_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("CHROMA_PERSIST_DIR", "/tmp/test-chroma")
        from src.config import Settings

        assert Settings().chroma_persist_dir == Path("/tmp/test-chroma")

    def test_embedding_model_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-large")
        from src.config import Settings

        assert Settings().embedding_model == "text-embedding-3-large"

    def test_embedding_device_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
        from src.config import Settings

        assert Settings().embedding_device == "cpu"

    def test_rerank_candidate_pool_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("RERANK_CANDIDATE_POOL", "40")
        from src.config import Settings

        assert Settings().rerank_candidate_pool == 40

    def test_reranking_enabled_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("RERANKING_ENABLED", "false")
        from src.config import Settings

        assert Settings().reranking_enabled is False

    def test_reranker_device_env_override(self, monkeypatch, clean_env):
        monkeypatch.setenv("RERANKER_DEVICE", "cpu")
        from src.config import Settings

        assert Settings().reranker_device == "cpu"


class TestSettingsValidation:
    def test_invalid_log_level_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        from src.config import Settings

        with pytest.raises(ValidationError, match="log_level"):
            Settings()

    def test_dense_top_k_zero_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("DENSE_TOP_K", "0")
        from src.config import Settings

        with pytest.raises(ValidationError):
            Settings()

    def test_dense_weight_above_one_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("DENSE_WEIGHT", "1.5")
        from src.config import Settings

        with pytest.raises(ValidationError):
            Settings()

    def test_dedup_threshold_negative_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("DEDUP_THRESHOLD", "-0.1")
        from src.config import Settings

        with pytest.raises(ValidationError):
            Settings()

    def test_log_level_case_insensitive(self, monkeypatch, clean_env):
        monkeypatch.setenv("LOG_LEVEL", "warning")
        from src.config import Settings

        assert Settings().log_level == "WARNING"

    def test_invalid_embedding_device_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("EMBEDDING_DEVICE", "tpu")
        from src.config import Settings

        with pytest.raises(ValidationError, match="embedding_device"):
            Settings()

    def test_rerank_candidate_pool_zero_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("RERANK_CANDIDATE_POOL", "0")
        from src.config import Settings

        with pytest.raises(ValidationError):
            Settings()

    def test_invalid_reranker_device_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("RERANKER_DEVICE", "tpu")
        from src.config import Settings

        with pytest.raises(ValidationError, match="reranker_device"):
            Settings()

    def test_rerank_top_n_exceeds_candidate_pool_raises(self, monkeypatch, clean_env):
        monkeypatch.setenv("RERANK_TOP_N", "25")
        monkeypatch.setenv("RERANK_CANDIDATE_POOL", "20")
        from src.config import Settings

        with pytest.raises(ValidationError, match="rerank_top_n"):
            Settings()


class TestSettingsProperties:
    def test_chroma_persist_dir_str_is_string(self, clean_env):
        from src.config import Settings

        s = Settings()
        assert isinstance(s.chroma_persist_dir_str, str)
        # Path normalizes "./data/chroma" → "data/chroma" on str conversion
        assert s.chroma_persist_dir_str == str(Path("./data/chroma"))


class TestSettingsSingleton:
    def test_singleton_importable(self):
        from src.config import settings

        assert settings is not None

    def test_singleton_is_settings_instance(self):
        from src.config import Settings, settings

        assert isinstance(settings, Settings)


class TestChunkingSettingsDefaults:
    def test_chunk_strategy_default(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().chunk_strategy == "fixed_size"

    def test_chunk_size_default(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().chunk_size == 1000

    def test_chunk_overlap_default(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().chunk_overlap == 200

    def test_semantic_breakpoint_percentile_default(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().semantic_breakpoint_percentile == 95.0


class TestCitationVerificationSettingsDefaults:
    def test_citation_judge_provider_default(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().citation_judge_provider == "anthropic"

    def test_citation_judge_model_default(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().citation_judge_model == "claude-sonnet-4-5"

    def test_citation_judge_temperature_default(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().citation_judge_temperature == pytest.approx(0.0)

    def test_anthropic_api_key_default_empty(self, clean_env: None) -> None:
        from src.config import Settings

        assert Settings().anthropic_api_key == ""


class TestCitationVerificationSettingsOverrides:
    def test_anthropic_api_key_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import Settings

        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-abc123")
        assert Settings().anthropic_api_key == "sk-ant-test-abc123"

    def test_citation_judge_provider_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_PROVIDER", "openai")
        assert Settings().citation_judge_provider == "openai"

    def test_citation_judge_model_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_MODEL", "gpt-4-turbo")
        assert Settings().citation_judge_model == "gpt-4-turbo"

    def test_citation_judge_temperature_env_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_TEMPERATURE", "0.5")
        assert Settings().citation_judge_temperature == pytest.approx(0.5)


class TestCitationVerificationSettingsValidation:
    def test_citation_judge_provider_invalid_raises(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_PROVIDER", "gemini")
        with pytest.raises(ValidationError, match="citation_judge_provider"):
            Settings()

    def test_citation_judge_temperature_below_zero_raises(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_TEMPERATURE", "-0.1")
        with pytest.raises(ValidationError):
            Settings()

    def test_citation_judge_temperature_above_one_raises(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_TEMPERATURE", "1.5")
        with pytest.raises(ValidationError):
            Settings()

    def test_citation_judge_temperature_boundary_zero(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_TEMPERATURE", "0.0")
        assert Settings().citation_judge_temperature == pytest.approx(0.0)

    def test_citation_judge_temperature_boundary_one(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import Settings

        monkeypatch.setenv("CITATION_JUDGE_TEMPERATURE", "1.0")
        assert Settings().citation_judge_temperature == pytest.approx(1.0)


class TestChunkingSettingsValidation:
    def test_invalid_chunk_strategy_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("CHUNK_STRATEGY", "sliding_window")
        with pytest.raises(ValidationError):
            Settings()

    def test_chunk_size_below_minimum_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("CHUNK_SIZE", "50")
        with pytest.raises(ValidationError):
            Settings()

    def test_chunk_overlap_negative_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("CHUNK_OVERLAP", "-1")
        with pytest.raises(ValidationError):
            Settings()

    def test_semantic_percentile_above_100_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("SEMANTIC_BREAKPOINT_PERCENTILE", "101.0")
        with pytest.raises(ValidationError):
            Settings()

    def test_chunk_strategy_env_override(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.config import Settings

        monkeypatch.setenv("CHUNK_STRATEGY", "semantic")
        assert Settings().chunk_strategy == "semantic"

    def test_chunk_overlap_gte_chunk_size_rejected(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pydantic import ValidationError

        from src.config import Settings

        monkeypatch.setenv("CHUNK_SIZE", "100")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        with pytest.raises(ValidationError):
            Settings()
