from openai import OpenAI

from src.config import Settings

BATCH_SIZE = 200


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            response = self._client.embeddings.create(input=batch, model=self._model)
            vectors.extend(item.embedding for item in response.data)
        return vectors
