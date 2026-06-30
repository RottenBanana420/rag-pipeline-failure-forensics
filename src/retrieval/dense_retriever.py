from src.retrieval.embedder import Embedder
from src.retrieval.models import VectorStoreHit
from src.retrieval.vector_store import VectorStore

_DEFAULT_K = 10


class DenseRetriever:
    def __init__(self, embedder: Embedder, vector_store: VectorStore) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, k: int = _DEFAULT_K) -> list[VectorStoreHit]:
        (embedding,) = self._embedder.embed([query])
        return self._vector_store.query(embedding, k)
