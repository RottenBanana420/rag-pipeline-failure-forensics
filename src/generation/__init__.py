"""Generation module — grounded prompt construction, citation parsing/verification,
and confidence scoring (Phase 2)."""

from src.generation.citation_parser import Citation, parse_citations
from src.generation.citation_verifier import (
    CITATION_JUDGE_SYSTEM_PROMPT,
    CitationJudgeProtocol,
    CitationVerificationResult,
    JudgeVerdict,
    build_judge_prompt,
    make_citation_judge,
    verify_citations,
)
from src.generation.confidence_scorer import (
    ANSWER_COMPLETENESS_SYSTEM_PROMPT,
    CompletenessJudgeProtocol,
    CompletenessVerdict,
    ConfidenceScore,
    build_completeness_judge_prompt,
    make_completeness_judge,
    score_confidence,
)
from src.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    INSUFFICIENT_CONTEXT_RESPONSE,
    GroundedPrompt,
    build_grounded_prompt,
    wrap_with_nonce,
)

__all__ = [
    "ANSWER_COMPLETENESS_SYSTEM_PROMPT",
    "CITATION_JUDGE_SYSTEM_PROMPT",
    "Citation",
    "CitationJudgeProtocol",
    "CitationVerificationResult",
    "CompletenessJudgeProtocol",
    "CompletenessVerdict",
    "ConfidenceScore",
    "GROUNDED_SYSTEM_PROMPT",
    "INSUFFICIENT_CONTEXT_RESPONSE",
    "GroundedPrompt",
    "JudgeVerdict",
    "build_completeness_judge_prompt",
    "build_grounded_prompt",
    "build_judge_prompt",
    "make_citation_judge",
    "make_completeness_judge",
    "parse_citations",
    "score_confidence",
    "verify_citations",
    "wrap_with_nonce",
]
