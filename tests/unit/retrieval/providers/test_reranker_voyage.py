"""Unit tests for VoyageReranker.

Uses mocking to avoid making real network calls to the Voyage API.
"""

from unittest.mock import MagicMock, patch

import pytest

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


def _result(index: int, relevance_score: float) -> MagicMock:
    result = MagicMock()
    result.index = index
    result.relevance_score = relevance_score
    return result


def _mock_response(results: list[MagicMock]) -> MagicMock:
    response = MagicMock()
    response.results = results
    return response


@pytest.fixture
def settings(monkeypatch, tmp_path):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("CHUNK_STRATEGY", "fixed_size")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("RERANKER_MODEL", "rerank-2.5")
    from src.config import Settings

    return Settings()


class TestVoyageReranker:
    def test_importable(self):
        from src.retrieval.providers.reranker_voyage import VoyageReranker  # noqa: F401

    def test_satisfies_reranker_protocol(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker
        from src.retrieval.reranker import RerankerProtocol

        with patch("voyageai.Client"):
            reranker = VoyageReranker(settings)

        assert isinstance(reranker, RerankerProtocol)

    def test_rerank_empty_hits_returns_empty_without_calling_rerank(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker

        with patch("voyageai.Client") as MockClient:
            reranker = VoyageReranker(settings)
            result = reranker.rerank("q", [], top_n=5)

        assert result == []
        MockClient.return_value.rerank.assert_not_called()

    def test_rerank_maps_index_and_relevance_score(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker

        hits = [_hit("a"), _hit("b"), _hit("c")]
        # API returns results out of input order, already sorted descending:
        # "b" (index 1) scores highest, then "c" (index 2), then "a" (index 0)
        mock_results = [_result(1, 0.9), _result(2, 0.5), _result(0, 0.1)]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(mock_results)
            reranker = VoyageReranker(settings)
            result = reranker.rerank("q", hits, top_n=3)

        assert [h.chunk_id for h in result] == ["b", "c", "a"]
        assert [h.similarity for h in result] == [0.9, 0.5, 0.1]

    def test_rerank_preserves_other_hit_fields(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker

        hits = [_hit("a", text="hello", similarity=0.5)]
        mock_results = [_result(0, 8.6)]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(mock_results)
            reranker = VoyageReranker(settings)
            result = reranker.rerank("q", hits, top_n=1)

        assert result[0].similarity == 8.6
        assert result[0].chunk_id == "a"
        assert result[0].text == "hello"

    def test_rerank_does_not_mutate_input_hits(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker

        original = _hit("a", similarity=0.5)
        hits = [original]
        mock_results = [_result(0, 0.99)]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(mock_results)
            reranker = VoyageReranker(settings)
            reranker.rerank("q", hits, top_n=1)

        assert original.similarity == 0.5
        assert hits[0].similarity == 0.5

    def test_rerank_calls_api_with_query_documents_model_top_k(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker

        hits = [_hit("a", text="alpha"), _hit("b", text="beta")]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(
                [_result(0, 0.1), _result(1, 0.2)]
            )
            reranker = VoyageReranker(settings)
            reranker.rerank("my query", hits, top_n=2)

        MockClient.return_value.rerank.assert_called_once_with(
            "my query",
            ["alpha", "beta"],
            model=settings.reranker_model,
            top_k=2,
        )

    def test_rerank_top_n_smaller_than_hits_caps_output_length(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker

        hits = [_hit("a"), _hit("b"), _hit("c"), _hit("d"), _hit("e")]
        # API honors top_k and only returns the top 2 results.
        mock_results = [_result(3, 0.9), _result(1, 0.7)]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(mock_results)
            reranker = VoyageReranker(settings)
            result = reranker.rerank("q", hits, top_n=2)

        assert len(result) == 2
        assert [h.chunk_id for h in result] == ["d", "b"]

    def test_provider_id_includes_model_name(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker

        with patch("voyageai.Client"):
            reranker = VoyageReranker(settings)

        assert reranker.provider_id == f"voyage/{settings.reranker_model}"

    def test_default_model(self):
        from src.retrieval.providers.reranker_voyage import DEFAULT_MODEL

        assert DEFAULT_MODEL == "rerank-2.5"

    def test_voyageai_imported_lazily(self):
        """voyageai should not be imported at module top-level in the provider file."""
        import sys

        voyageai_mod = sys.modules.pop("voyageai", None)
        try:
            if "src.retrieval.providers.reranker_voyage" in sys.modules:
                del sys.modules["src.retrieval.providers.reranker_voyage"]
            import src.retrieval.providers.reranker_voyage  # noqa: F401

            assert "voyageai" not in sys.modules
        finally:
            if voyageai_mod is not None:
                sys.modules["voyageai"] = voyageai_mod


class TestVoyageRerankerTracing:
    def test_rerank_records_ranking_span(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker
        from src.tracing.context import collect_spans

        hits = [_hit("a")]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(
                [_result(0, 0.9)]
            )
            reranker = VoyageReranker(settings)
            with collect_spans() as spans:
                reranker.rerank("q", hits, top_n=1)

        assert len(spans) == 1
        assert spans[0].step == "ranking"
        assert spans[0].error is None

    def test_rerank_sets_confidence_score_from_mean_similarity(self, settings):
        from src.retrieval.providers.reranker_voyage import VoyageReranker
        from src.tracing.context import collect_spans

        hits = [_hit("a")]
        with patch("voyageai.Client") as MockClient:
            MockClient.return_value.rerank.return_value = _mock_response(
                [_result(0, 1.0)]
            )
            reranker = VoyageReranker(settings)
            with collect_spans() as spans:
                reranker.rerank("q", hits, top_n=1)

        assert spans[0].confidence_score == 5
