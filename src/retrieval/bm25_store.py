import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import Settings
from src.ingestion import Chunk


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Store:
    def __init__(self, settings: Settings) -> None:
        self._index_path = Path(settings.chroma_persist_dir).parent / "bm25_index.pkl"
        self._chunk_ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._chunk_ids.append(chunk.chunk_id)
            self._tokenized_corpus.append(_tokenize(chunk.text))
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._index_path, "wb") as fh:
            pickle.dump({"chunk_ids": self._chunk_ids, "corpus": self._tokenized_corpus}, fh)

    def load(self) -> None:
        if not self._index_path.exists():
            return
        with open(self._index_path, "rb") as fh:
            data = pickle.load(fh)
        self._chunk_ids = data["chunk_ids"]
        self._tokenized_corpus = data["corpus"]
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    def get_scores(self, query: str) -> list[tuple[str, float]]:
        if self._bm25 is None or not self._chunk_ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))

        if len(set(scores)) == 1:
            query_tokens = set(_tokenize(query))
            scores = [
                float(sum(1 for token in tokens if token in query_tokens))
                for tokens in self._tokenized_corpus
            ]
        else:
            scores = scores.tolist()

        return list(zip(self._chunk_ids, scores, strict=True))

    def count(self) -> int:
        return len(self._chunk_ids)
