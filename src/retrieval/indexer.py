import logging
from concurrent.futures import ThreadPoolExecutor

from src.config import Settings
from src.ingestion import Chunk
from src.retrieval.bm25_store import BM25Store
from src.retrieval.embedder import Embedder
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Indexer:
    def __init__(
        self,
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        bm25_store: BM25Store | None = None,
    ) -> None:
        self._embedder = embedder or Embedder(settings)
        self._vector_store = vector_store or VectorStore(settings)
        self._bm25_store = bm25_store or BM25Store(settings)
        self._bm25_store.load()

    def index(self, chunks: list[Chunk]) -> list[str]:
        if not chunks:
            return []

        embeddings = self._embedder.embed([c.text for c in chunks])
        accepted_chunks, accepted_embeddings = self._vector_store.filter_duplicates(
            chunks, embeddings
        )

        skipped = len(chunks) - len(accepted_chunks)
        if skipped:
            logger.info(
                "Indexer: %d/%d chunks accepted, %d duplicate(s) skipped",
                len(accepted_chunks),
                len(chunks),
                skipped,
            )

        if not accepted_chunks:
            return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            chroma_future = executor.submit(
                self._vector_store.upsert, accepted_chunks, accepted_embeddings
            )
            bm25_future = executor.submit(self._bm25_store.add, accepted_chunks)

        stored_ids = chroma_future.result()
        bm25_future.result()

        self._bm25_store.save()
        return stored_ids
