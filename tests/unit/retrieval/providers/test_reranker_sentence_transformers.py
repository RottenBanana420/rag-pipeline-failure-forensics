"""Unit tests for RerankerProtocol conformance and SentenceTransformersReranker.

Uses mocking to avoid downloading the cross-encoder model in CI.
"""

from unittest.mock import MagicMock, patch

from src.retrieval.models import VectorStoreHit


def _hit(chunk_id: str, text: str = "text", similarity: float = 0.5) -> VectorStoreHit:
    return VectorStoreHit(
        chunk_id=chunk_id,
        text=text,
        doc_id="doc1",
        source_path="/p",
        title="T",
        section_heading=None,
        chunk_index=0,
        strategy="fixed_size",
        similarity=similarity,
    )


class TestSentenceTransformersReranker:
    def test_satisfies_reranker_protocol(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )
        from src.retrieval.reranker import RerankerProtocol

        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            reranker = SentenceTransformersReranker()

        assert isinstance(reranker, RerankerProtocol)

    def test_rerank_empty_hits_returns_empty_without_calling_predict(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        mock_model = MagicMock()
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = SentenceTransformersReranker()
            result = reranker.rerank("q", [], top_n=5)

        assert result == []
        mock_model.predict.assert_not_called()

    def test_rerank_returns_hits_sorted_by_score_descending(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        hits = [_hit("a"), _hit("b"), _hit("c")]
        mock_model = MagicMock()
        # Deliberately out of input order: "b" scores highest, then "c", then "a"
        mock_model.predict.return_value = [0.1, 0.9, 0.5]
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = SentenceTransformersReranker()
            result = reranker.rerank("q", hits, top_n=3)

        assert [h.chunk_id for h in result] == ["b", "c", "a"]

    def test_rerank_limits_to_top_n(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        hits = [_hit(f"c{i}") for i in range(5)]
        mock_model = MagicMock()
        mock_model.predict.return_value = [float(i) for i in range(5)]
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = SentenceTransformersReranker()
            result = reranker.rerank("q", hits, top_n=2)

        assert len(result) == 2

    def test_rerank_fewer_hits_than_top_n_returns_all(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        hits = [_hit("a"), _hit("b"), _hit("c")]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.2, 0.3]
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = SentenceTransformersReranker()
            result = reranker.rerank("q", hits, top_n=5)

        assert len(result) == 3

    def test_rerank_stamps_similarity_with_reranker_score(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        hits = [_hit("a", text="hello", similarity=0.5)]
        mock_model = MagicMock()
        mock_model.predict.return_value = [8.6]
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = SentenceTransformersReranker()
            result = reranker.rerank("q", hits, top_n=1)

        assert result[0].similarity == 8.6
        # Other fields preserved from the original hit
        assert result[0].chunk_id == "a"
        assert result[0].text == "hello"

    def test_rerank_calls_predict_with_query_text_pairs(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        hits = [_hit("a", text="alpha"), _hit("b", text="beta")]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.1, 0.2]
        with patch("sentence_transformers.CrossEncoder", return_value=mock_model):
            reranker = SentenceTransformersReranker()
            reranker.rerank("my query", hits, top_n=2)

        mock_model.predict.assert_called_once_with(
            [("my query", "alpha"), ("my query", "beta")]
        )

    def test_provider_id_includes_model_name(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            reranker = SentenceTransformersReranker(model_name="custom-cross-encoder")

        assert reranker.provider_id == "sentence_transformers/custom-cross-encoder"

    def test_default_model_name(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            DEFAULT_MODEL,
            SentenceTransformersReranker,
        )

        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            reranker = SentenceTransformersReranker()

        assert reranker._model_name == DEFAULT_MODEL
        assert DEFAULT_MODEL == "cross-encoder/ms-marco-MiniLM-L6-v2"

    def test_device_none_by_default(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            DEFAULT_MODEL,
            SentenceTransformersReranker,
        )

        with patch(
            "sentence_transformers.CrossEncoder", return_value=MagicMock()
        ) as mock_ctor:
            SentenceTransformersReranker()

        mock_ctor.assert_called_once_with(DEFAULT_MODEL, device=None)

    def test_device_passed_through(self):
        from src.retrieval.providers.reranker_sentence_transformers import (
            DEFAULT_MODEL,
            SentenceTransformersReranker,
        )

        with patch(
            "sentence_transformers.CrossEncoder", return_value=MagicMock()
        ) as mock_ctor:
            SentenceTransformersReranker(device="cpu")

        mock_ctor.assert_called_once_with(DEFAULT_MODEL, device="cpu")

    def test_logs_resolved_device(self, caplog):
        import logging

        from src.retrieval.providers.reranker_sentence_transformers import (
            SentenceTransformersReranker,
        )

        mock_model = MagicMock()
        mock_model.device = "cpu"
        with (
            patch("sentence_transformers.CrossEncoder", return_value=mock_model),
            caplog.at_level(logging.INFO),
        ):
            SentenceTransformersReranker()

        assert any(
            "cross-encoder/ms-marco-MiniLM-L6-v2" in record.message and "cpu" in record.message
            for record in caplog.records
        )

    def test_cross_encoder_imported_lazily(self):
        """CrossEncoder should not be imported at module top-level."""
        import sys

        st_mod = sys.modules.pop("sentence_transformers", None)
        try:
            sys.modules.pop(
                "src.retrieval.providers.reranker_sentence_transformers", None
            )
            import src.retrieval.providers.reranker_sentence_transformers  # noqa: F401

            assert "sentence_transformers" not in sys.modules
        finally:
            if st_mod is not None:
                sys.modules["sentence_transformers"] = st_mod
