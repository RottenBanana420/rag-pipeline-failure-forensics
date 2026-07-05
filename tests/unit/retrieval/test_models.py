from src.retrieval.models import VectorStoreHit, mean_similarity_confidence


def _hit(similarity: float) -> VectorStoreHit:
    return VectorStoreHit(
        chunk_id="c1",
        text="text",
        doc_id="doc1",
        source_path="/p",
        title="T",
        section_heading=None,
        chunk_index=0,
        strategy="fixed_size",
        similarity=similarity,
    )


class TestMeanSimilarityConfidence:
    def test_empty_hits_returns_none(self):
        assert mean_similarity_confidence([]) is None

    def test_single_hit_maps_similarity_to_confidence(self):
        assert mean_similarity_confidence([_hit(1.0)]) == 5

    def test_averages_similarity_across_hits(self):
        assert mean_similarity_confidence([_hit(1.0), _hit(0.0)]) == 3
