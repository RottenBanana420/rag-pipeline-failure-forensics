from src.retrieval.embedder import EmbedderProtocol
from src.retrieval.models import VectorStoreHit, mean_similarity_confidence
from src.retrieval.vector_store import VectorStoreProtocol
from src.tracing.instrumentation import traced

_DEFAULT_K = 10


class DenseRetriever:
    def __init__(
        self, embedder: EmbedderProtocol, vector_store: VectorStoreProtocol
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    @traced("retrieval", confidence_fn=mean_similarity_confidence)
    def retrieve(self, query: str, k: int = _DEFAULT_K) -> list[VectorStoreHit]:
        (embedding,) = self._embedder.embed([query])
        return self._vector_store.query(embedding, k)
