"""Unit tests for the citation parser."""

import dataclasses

import pytest

from src.generation.citation_parser import Citation, parse_citations


class TestCitation:
    def test_is_frozen_dataclass(self):
        citation = Citation(claim_text="Paris is the capital", chunk_indices=[1])

        with pytest.raises(dataclasses.FrozenInstanceError):
            citation.claim_text = "changed"  # type: ignore[misc]

    def test_is_frozen_dataclass_for_chunk_indices(self):
        citation = Citation(claim_text="Paris is the capital", chunk_indices=[1])

        with pytest.raises(dataclasses.FrozenInstanceError):
            citation.chunk_indices = [2]  # type: ignore[misc]

    def test_has_claim_text_field(self):
        citation = Citation(claim_text="Test claim", chunk_indices=[1])

        assert citation.claim_text == "Test claim"

    def test_has_chunk_indices_field(self):
        citation = Citation(claim_text="Test claim", chunk_indices=[1, 3])

        assert citation.chunk_indices == [1, 3]


class TestParseCitations:
    def test_single_citation(self):
        text = "Paris is the capital of France [1]."

        result = parse_citations(text)

        assert len(result) == 1
        assert result[0].claim_text == "Paris is the capital of France"
        assert result[0].chunk_indices == [1]

    def test_multi_bracket_citation(self):
        text = "The information is drawn from two sources [1][3]."

        result = parse_citations(text)

        assert len(result) == 1
        assert result[0].claim_text == "The information is drawn from two sources"
        assert result[0].chunk_indices == [1, 3]

    def test_multiple_citations_in_sequence(self):
        text = "First claim [1]. Second claim [2]."

        result = parse_citations(text)

        assert len(result) == 2
        assert result[0].claim_text == "First claim"
        assert result[0].chunk_indices == [1]
        assert result[1].claim_text == ". Second claim"
        assert result[1].chunk_indices == [2]

    def test_no_citations_returns_empty_list(self):
        text = "This text has no citations at all."

        result = parse_citations(text)

        assert result == []

    def test_empty_string_returns_empty_list(self):
        result = parse_citations("")

        assert result == []

    def test_malformed_brackets_not_treated_as_citations(self):
        text = "Text with [a] and [1 malformed brackets."

        result = parse_citations(text)

        assert result == []

    def test_mixed_valid_and_malformed_brackets(self):
        text = "Valid [1] citation, but also [a] malformed [2]."

        result = parse_citations(text)

        assert len(result) == 2
        assert result[0].claim_text == "Valid"
        assert result[0].chunk_indices == [1]
        assert result[1].claim_text == "citation, but also [a] malformed"
        assert result[1].chunk_indices == [2]

    def test_claims_are_stripped(self):
        text = "  Claim with whitespace  [1]."

        result = parse_citations(text)

        assert len(result) == 1
        assert result[0].claim_text == "Claim with whitespace"

    def test_citation_with_large_numbers(self):
        text = "From a large context pool [999]."

        result = parse_citations(text)

        assert len(result) == 1
        assert result[0].chunk_indices == [999]

    def test_multiple_citations_with_complex_run(self):
        text = "Multi-source claim [1][2][3]."

        result = parse_citations(text)

        assert len(result) == 1
        assert result[0].chunk_indices == [1, 2, 3]

    def test_claim_starts_at_beginning_of_string(self):
        text = "First claim [1]."

        result = parse_citations(text)

        assert result[0].claim_text == "First claim"

    def test_claim_inherits_from_previous_citation_end(self):
        text = "First [1]. Second part [2]."

        result = parse_citations(text)

        assert result[1].claim_text == ". Second part"

    def test_text_between_citations_is_claim(self):
        text = "Start [1] middle text [2] end."

        result = parse_citations(text)

        assert len(result) == 2
        assert result[0].claim_text == "Start"
        assert result[1].claim_text == "middle text"

    def test_preserves_claim_text_punctuation(self):
        text = "Question: What is this? [1]"

        result = parse_citations(text)

        assert result[0].claim_text == "Question: What is this?"

    def test_empty_claim_between_citations(self):
        """Consecutive citations with nothing between them have an empty claim."""
        text = "Claim [1][2]."

        result = parse_citations(text)

        # The claim is the text from start to the first citation run
        assert result[0].claim_text == "Claim"
        assert result[0].chunk_indices == [1, 2]

    def test_trailing_text_after_final_citation(self):
        """Text after the last citation is not captured (not a claim)."""
        text = "Final claim [1]. Trailing text."

        result = parse_citations(text)

        # "Trailing text." comes after the citation, so it's not a claim.
        assert len(result) == 1
        assert result[0].claim_text == "Final claim"
