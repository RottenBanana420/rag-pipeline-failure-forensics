from unittest.mock import MagicMock

from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.embedder import Embedder
from src.retrieval.models import VectorStoreHit
from src.retrieval.vector_store import VectorStore


def _hit(**kwargs) -> VectorStoreHit:
    defaults = dict(
        chunk_id="c-000",
        text="sample",
        doc_id="d-000",
        source_path="/doc.md",
        title="Doc",
        section_heading=None,
        chunk_index=0,
        strategy="fixed_size",
        similarity=0.9,
    )
    return VectorStoreHit(**{**defaults, **kwargs})


class TestDenseRetriever:
    def test_retrieve_embeds_query(self):
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        vs = MagicMock(spec=VectorStore)
        vs.query.return_value = []

        DenseRetriever(embedder, vs).retrieve("what is X?")

        embedder.embed.assert_called_once_with(["what is X?"])

    def test_retrieve_passes_embedding_to_query(self):
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[0.5, 0.5, 0.0]]
        vs = MagicMock(spec=VectorStore)
        vs.query.return_value = []

        DenseRetriever(embedder, vs).retrieve("q")

        vs.query.assert_called_once_with([0.5, 0.5, 0.0], 10)

    def test_retrieve_passes_k_to_query(self):
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        vs = MagicMock(spec=VectorStore)
        vs.query.return_value = []

        DenseRetriever(embedder, vs).retrieve("q", k=5)

        vs.query.assert_called_once_with([1.0, 0.0, 0.0], 5)

    def test_retrieve_returns_query_results(self):
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        vs = MagicMock(spec=VectorStore)
        expected = [_hit(chunk_id="c-000"), _hit(chunk_id="c-001", similarity=0.7)]
        vs.query.return_value = expected

        result = DenseRetriever(embedder, vs).retrieve("q")

        assert result == expected

    def test_retrieve_default_k_is_10(self):
        embedder = MagicMock(spec=Embedder)
        embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        vs = MagicMock(spec=VectorStore)
        vs.query.return_value = []

        DenseRetriever(embedder, vs).retrieve("q")

        _, called_k = vs.query.call_args.args
        assert called_k == 10
