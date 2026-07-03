"""Unit tests for the grounded generation prompt builder."""

import dataclasses

import pytest

from src.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    GroundedPrompt,
    build_grounded_prompt,
    format_context_blocks,
)
from src.retrieval.models import VectorStoreHit


def make_hit(
    chunk_id: str = "chunk-1",
    text: str = "Paris is the capital of France.",
    doc_id: str = "doc-1",
    source_path: str = "/docs/geography.md",
    title: str = "Geography Facts",
    section_heading: str | None = "Capitals",
    chunk_index: int = 0,
    strategy: str = "fixed_size",
    similarity: float = 0.9,
) -> VectorStoreHit:
    return VectorStoreHit(
        chunk_id=chunk_id,
        text=text,
        doc_id=doc_id,
        source_path=source_path,
        title=title,
        section_heading=section_heading,
        chunk_index=chunk_index,
        strategy=strategy,
        similarity=similarity,
    )


class TestFormatContextBlocks:
    def test_single_hit_includes_title_section_and_text(self):
        hit = make_hit(
            title="Geography Facts",
            section_heading="Capitals",
            text="Paris is the capital.",
        )

        result = format_context_blocks([hit])

        assert "[1]" in result
        assert "Geography Facts" in result
        assert "Capitals" in result
        assert "Paris is the capital." in result

    def test_numbers_blocks_starting_at_one(self):
        hits = [make_hit(chunk_id="a"), make_hit(chunk_id="b"), make_hit(chunk_id="c")]

        result = format_context_blocks(hits)

        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_preserves_input_order_without_resorting(self):
        hits = [
            make_hit(chunk_id="first", text="FIRST_MARKER", similarity=0.1),
            make_hit(chunk_id="second", text="SECOND_MARKER", similarity=0.99),
        ]

        result = format_context_blocks(hits)

        # Even though "second" has higher similarity, it must stay in list order.
        first_pos = result.index("FIRST_MARKER")
        second_pos = result.index("SECOND_MARKER")
        assert result.index("[1]") < first_pos
        assert first_pos < second_pos

    def test_handles_section_heading_none(self):
        hit = make_hit(
            title="Standalone Doc", section_heading=None, text="Some body text."
        )

        result = format_context_blocks([hit])

        assert "[1] Source: Standalone Doc" in result
        assert "Some body text." in result
        # No dangling separator when there's no section heading.
        assert "Standalone Doc —" not in result
        assert "Standalone Doc -" not in result

    def test_includes_separator_when_section_heading_present(self):
        hit = make_hit(title="Geography Facts", section_heading="Capitals")

        result = format_context_blocks([hit])

        assert "Geography Facts — Capitals" in result

    def test_empty_hits_returns_explicit_marker_not_empty_string(self):
        result = format_context_blocks([])

        assert result != ""
        assert result.strip() != ""
        assert "no context" in result.lower()


class TestBuildGroundedPrompt:
    def test_embeds_question(self):
        hits = [make_hit()]

        prompt = build_grounded_prompt("What is the capital of France?", hits)

        assert "What is the capital of France?" in prompt.user

    def test_embeds_all_formatted_blocks(self):
        hits = [
            make_hit(chunk_id="a", text="Block A text", title="Doc A"),
            make_hit(chunk_id="b", text="Block B text", title="Doc B"),
        ]

        prompt = build_grounded_prompt("Some question", hits)

        assert "[1]" in prompt.user
        assert "Block A text" in prompt.user
        assert "Doc A" in prompt.user
        assert "[2]" in prompt.user
        assert "Block B text" in prompt.user
        assert "Doc B" in prompt.user

    def test_user_prompt_wraps_context_and_question_in_xml_tags(self):
        hits = [make_hit()]

        prompt = build_grounded_prompt("Some question", hits)

        assert "<context>" in prompt.user
        assert "</context>" in prompt.user
        assert "<question>" in prompt.user
        assert "</question>" in prompt.user

    def test_system_prompt_contains_citation_format_instructions(self):
        hits = [make_hit()]

        prompt = build_grounded_prompt("Some question", hits)

        assert "[1]" in prompt.system

    def test_system_prompt_contains_fallback_phrase_instruction(self):
        hits = [make_hit()]

        prompt = build_grounded_prompt("Some question", hits)

        assert "enough information" in prompt.system.lower()

    def test_system_prompt_equals_module_constant(self):
        prompt = build_grounded_prompt("Some question", [make_hit()])

        assert prompt.system == GROUNDED_SYSTEM_PROMPT

    def test_empty_hits_produces_no_context_marker_in_user_prompt(self):
        prompt = build_grounded_prompt("Some question", [])

        assert "no context" in prompt.user.lower()
        assert "Some question" in prompt.user

    def test_returns_grounded_prompt_instance(self):
        prompt = build_grounded_prompt("Some question", [make_hit()])

        assert isinstance(prompt, GroundedPrompt)


class TestGroundedSystemPrompt:
    def test_mentions_citation_bracket_format(self):
        assert "[1]" in GROUNDED_SYSTEM_PROMPT

    def test_mentions_multi_citation_format(self):
        assert "[1][3]" in GROUNDED_SYSTEM_PROMPT or "[1][2]" in GROUNDED_SYSTEM_PROMPT

    def test_mentions_no_outside_knowledge(self):
        text = GROUNDED_SYSTEM_PROMPT.lower()
        assert "outside knowledge" in text or "only using" in text or "only use" in text

    def test_contains_fallback_phrase(self):
        assert (
            "I don't have enough information in the provided context to answer this."
            in GROUNDED_SYSTEM_PROMPT
        )

    def test_mentions_no_fabricated_citations(self):
        text = GROUNDED_SYSTEM_PROMPT.lower()
        assert "fabricate" in text


class TestGroundedPromptImmutability:
    def test_is_frozen_dataclass(self):
        prompt = GroundedPrompt(system="sys", user="usr")

        with pytest.raises(dataclasses.FrozenInstanceError):
            prompt.system = "changed"  # type: ignore[misc]

    def test_is_frozen_dataclass_for_user_field(self):
        prompt = GroundedPrompt(system="sys", user="usr")

        with pytest.raises(dataclasses.FrozenInstanceError):
            prompt.user = "changed"  # type: ignore[misc]
