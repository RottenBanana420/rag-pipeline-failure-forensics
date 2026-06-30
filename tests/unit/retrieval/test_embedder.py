from unittest.mock import MagicMock, patch


def _mock_response(n: int) -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(embedding=[float(i)] * 4) for i in range(n)]
    return resp


class TestEmbedder:
    def test_embed_returns_vector_per_text(self, settings):
        from src.retrieval.embedder import Embedder

        with patch("src.retrieval.embedder.OpenAI") as MockOpenAI:
            MockOpenAI.return_value.embeddings.create.return_value = _mock_response(3)
            result = Embedder(settings).embed(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == [0.0, 0.0, 0.0, 0.0]

    def test_embed_empty_input_returns_empty(self, settings):
        from src.retrieval.embedder import Embedder

        with patch("src.retrieval.embedder.OpenAI"):
            result = Embedder(settings).embed([])

        assert result == []

    def test_embed_batches_large_input(self, settings):
        from src.retrieval.embedder import BATCH_SIZE, Embedder

        texts = ["text"] * (BATCH_SIZE + 1)

        with patch("src.retrieval.embedder.OpenAI") as MockOpenAI:
            mock_create = MockOpenAI.return_value.embeddings.create
            mock_create.side_effect = [_mock_response(BATCH_SIZE), _mock_response(1)]
            result = Embedder(settings).embed(texts)

        assert mock_create.call_count == 2
        assert len(result) == BATCH_SIZE + 1

    def test_embed_passes_model_from_settings(self, settings):
        from src.retrieval.embedder import Embedder

        with patch("src.retrieval.embedder.OpenAI") as MockOpenAI:
            mock_create = MockOpenAI.return_value.embeddings.create
            mock_create.return_value = _mock_response(1)
            Embedder(settings).embed(["hello"])

        assert mock_create.call_args.kwargs["model"] == settings.embedding_model
