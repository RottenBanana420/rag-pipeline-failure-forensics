"""Tests for EmbedderProtocol and VectorStoreProtocol runtime_checkable protocols."""

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion import Chunk
from src.retrieval.models import VectorStoreHit


# ---------------------------------------------------------------------------
# EmbedderProtocol tests
# ---------------------------------------------------------------------------


class TestEmbedderProtocol:
    def test_protocol_is_importable(self):
        from src.retrieval.embedder import EmbedderProtocol

        assert EmbedderProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        from src.retrieval.embedder import EmbedderProtocol

        # This call must not raise TypeError
        isinstance(object(), EmbedderProtocol)  # type: ignore[arg-type]

    def test_object_without_embed_fails_isinstance(self):
        from src.retrieval.embedder import EmbedderProtocol

        class BadEmbedder:
            pass

        assert not isinstance(BadEmbedder(), EmbedderProtocol)  # type: ignore[arg-type]

    def test_object_with_embed_method_passes_isinstance(self):
        from src.retrieval.embedder import EmbedderProtocol

        class MinimalEmbedder:
            def embed(self, texts: list[str]) -> list[list[float]]:
                return []

            @property
            def dimensions(self) -> int:
                return 1536

            @property
            def provider_id(self) -> str:
                return "openai"

        assert isinstance(MinimalEmbedder(), EmbedderProtocol)  # type: ignore[arg-type]

    def test_concrete_embedder_satisfies_protocol(self):
        from src.retrieval.embedder import Embedder, EmbedderProtocol

        with patch("openai.OpenAI"):
            from src.config import Settings

            with patch.dict(
                __import__("os").environ,
                {"OPENAI_API_KEY": "test-key", "CHUNK_STRATEGY": "fixed_size"},
            ):
                settings = Settings()
            embedder = Embedder(settings)

        assert isinstance(embedder, EmbedderProtocol)  # type: ignore[arg-type]

    def test_protocol_has_embed_method(self):
        from src.retrieval.embedder import EmbedderProtocol

        assert hasattr(EmbedderProtocol, "embed")

    def test_protocol_has_dimensions_property(self):
        from src.retrieval.embedder import EmbedderProtocol

        assert hasattr(EmbedderProtocol, "dimensions")

    def test_protocol_has_provider_id_property(self):
        from src.retrieval.embedder import EmbedderProtocol

        assert hasattr(EmbedderProtocol, "provider_id")


# ---------------------------------------------------------------------------
# VectorStoreProtocol tests
# ---------------------------------------------------------------------------


class TestVectorStoreProtocol:
    def test_protocol_is_importable(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        assert VectorStoreProtocol is not None

    def test_protocol_is_runtime_checkable(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        isinstance(object(), VectorStoreProtocol)  # type: ignore[arg-type]

    def test_object_without_methods_fails_isinstance(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        class BadStore:
            pass

        assert not isinstance(BadStore(), VectorStoreProtocol)  # type: ignore[arg-type]

    def test_object_with_all_methods_passes_isinstance(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        class MinimalVectorStore:
            def filter_duplicates(
                self,
                chunks: list[Chunk],
                embeddings: list[list[float]],
            ) -> tuple[list[Chunk], list[list[float]]]:
                return chunks, embeddings

            def upsert(
                self, chunks: list[Chunk], embeddings: list[list[float]]
            ) -> list[str]:
                return []

            def query(self, embedding: list[float], k: int = 10) -> list[VectorStoreHit]:
                return []

            def get_by_ids(self, ids: list[str]) -> list[VectorStoreHit]:
                return []

            def count(self) -> int:
                return 0

        assert isinstance(MinimalVectorStore(), VectorStoreProtocol)  # type: ignore[arg-type]

    def test_concrete_vector_store_satisfies_protocol(self, settings):
        from src.retrieval.vector_store import VectorStore, VectorStoreProtocol

        vs = VectorStore(settings)
        assert isinstance(vs, VectorStoreProtocol)  # type: ignore[arg-type]

    def test_protocol_has_filter_duplicates(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        assert hasattr(VectorStoreProtocol, "filter_duplicates")

    def test_protocol_has_upsert(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        assert hasattr(VectorStoreProtocol, "upsert")

    def test_protocol_has_query(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        assert hasattr(VectorStoreProtocol, "query")

    def test_protocol_has_get_by_ids(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        assert hasattr(VectorStoreProtocol, "get_by_ids")

    def test_protocol_has_count(self):
        from src.retrieval.vector_store import VectorStoreProtocol

        assert hasattr(VectorStoreProtocol, "count")

    def test_partial_implementation_fails_isinstance(self):
        """An object with only some methods should not satisfy the protocol."""
        from src.retrieval.vector_store import VectorStoreProtocol

        class PartialStore:
            def upsert(
                self, chunks: list[Chunk], embeddings: list[list[float]]
            ) -> list[str]:
                return []

            def count(self) -> int:
                return 0

        # Missing filter_duplicates, query, get_by_ids → should fail
        assert not isinstance(PartialStore(), VectorStoreProtocol)  # type: ignore[arg-type]
