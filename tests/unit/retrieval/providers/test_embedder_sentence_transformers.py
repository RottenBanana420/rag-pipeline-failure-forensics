"""Unit tests for SentenceTransformersEmbedder — TDD (written before implementation).

Uses mocking to avoid downloading the model in CI.
"""

from unittest.mock import MagicMock, patch

def _make_mock_st_model(dim: int = 384) -> MagicMock:
    """Return a mock SentenceTransformer instance."""
    mock_model = MagicMock()
    mock_model.get_embedding_dimension.return_value = dim
    # encode returns a list of numpy-like arrays; we simulate with lists of floats
    mock_model.encode.return_value = [[0.1] * dim, [0.2] * dim]
    return mock_model


class TestSentenceTransformersEmbedder:
    def test_importable(self):
        from src.retrieval.providers.embedder_sentence_transformers import (  # noqa: F401
            SentenceTransformersEmbedder,
        )

    def test_default_model_name(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=_make_mock_st_model(),
        ):
            embedder = SentenceTransformersEmbedder()

        assert embedder._model_name == "all-MiniLM-L6-v2"

    def test_custom_model_name(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=_make_mock_st_model(768),
        ):
            embedder = SentenceTransformersEmbedder(model_name="bert-base-nli-mean-tokens")

        assert embedder._model_name == "bert-base-nli-mean-tokens"

    def test_dimensions_from_model(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        mock_model = _make_mock_st_model(dim=384)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            embedder = SentenceTransformersEmbedder()

        assert embedder.dimensions == 384

    def test_dimensions_custom_model(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        mock_model = _make_mock_st_model(dim=768)
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            embedder = SentenceTransformersEmbedder(model_name="some-768-model")

        assert embedder.dimensions == 768

    def test_provider_id_format(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=_make_mock_st_model(),
        ):
            embedder = SentenceTransformersEmbedder()

        assert embedder.provider_id == "sentence_transformers/all-MiniLM-L6-v2"

    def test_provider_id_custom_model(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=_make_mock_st_model(768),
        ):
            embedder = SentenceTransformersEmbedder(model_name="custom-model")

        assert embedder.provider_id == "sentence_transformers/custom-model"

    def test_embed_returns_list_of_lists(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        dim = 384
        mock_model = _make_mock_st_model(dim)
        import numpy as np

        mock_model.encode.return_value = np.array([[0.1] * dim, [0.2] * dim])

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            embedder = SentenceTransformersEmbedder()
            result = embedder.embed(["hello", "world"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)
        assert len(result[0]) == dim

    def test_embed_empty_returns_empty(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        mock_model = _make_mock_st_model()
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            embedder = SentenceTransformersEmbedder()
            result = embedder.embed([])

        assert result == []

    def test_embed_calls_encode(self):
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        import numpy as np

        dim = 384
        mock_model = _make_mock_st_model(dim)
        mock_model.encode.return_value = np.array([[0.1] * dim])

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_model,
        ):
            embedder = SentenceTransformersEmbedder()
            embedder.embed(["test text"])

        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        # First positional arg should be the texts list
        assert call_args[0][0] == ["test text"]

    def test_satisfies_embedder_protocol(self):
        from src.retrieval.embedder import EmbedderProtocol
        from src.retrieval.providers.embedder_sentence_transformers import (
            SentenceTransformersEmbedder,
        )

        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=_make_mock_st_model(),
        ):
            embedder = SentenceTransformersEmbedder()

        assert isinstance(embedder, EmbedderProtocol)

    def test_sentence_transformers_imported_lazily(self):
        """SentenceTransformer should not be imported at module top-level."""
        import sys

        st_mod = sys.modules.pop("sentence_transformers", None)
        try:
            if "src.retrieval.providers.embedder_sentence_transformers" in sys.modules:
                del sys.modules["src.retrieval.providers.embedder_sentence_transformers"]
            import src.retrieval.providers.embedder_sentence_transformers  # noqa: F401

            assert "sentence_transformers" not in sys.modules
        finally:
            if st_mod is not None:
                sys.modules["sentence_transformers"] = st_mod
