# Architecture Overview

## 2026-07-15 — Phase 5: Query Dashboard (Complete)

### The First Generation Orchestrator: Ask a Question, Get a Cited Answer, See the Confidence Behind It

`docs/PROJECT_SPEC.md` (Phase 5, item 4) asks for a dashboard where a user asks a question and sees the generated answer with clickable citations, ranked retrieved chunks, a per-dimension confidence breakdown, and a hybrid-vs-dense-only comparison toggle. Unlike every other Phase 5 item, this one couldn't be built as a UI over existing pieces: no module anywhere called an LLM to produce the actual `[N]`-cited answer text, and no orchestrator wired retrieval → generation → citation verification → confidence scoring → tracing together for a live request. This entry builds both the missing pipeline and the UI over it.

**`src/generation/answer_generator.py` (+ `providers/answer_generator_anthropic.py`/`answer_generator_openai.py`):** `AnswerGeneratorProtocol` (`generate(prompt) -> str`, `provider_id`) + `make_answer_generator(settings)`, the same lazy-import factory pattern as `make_citation_judge`/`make_completeness_judge`. Providers use each SDK's plain `messages.create`/`chat.completions.create` (not `.parse`/structured output — an answer is free text, not a verdict), wrapped in `span("generation", ...)` with `llm_prompt`/`token_count` set, same as the judge providers' inner spans.

**`src/frontend/query_service.py` (`ask_question`, `build_hybrid_retriever`):** The second per-feature orchestrator in `src/frontend/`, mirroring `diagnosis_service.py`. `ask_question(query, retriever, settings)` runs the full pipeline inside `collect_spans()`, then always calls `persist_trace` — "success" or "degraded" (fallback fired) on completion, "failure" (with whatever spans completed) if any step raised, re-raised afterward. This makes the query dashboard the first live caller of `persist_trace` — every question asked becomes a real, inspectable `Trace` in the existing trace view. `build_hybrid_retriever` wires an embedder, vector store, BM25 index, and optional reranker into a `HybridRetriever`; it holds no caching of its own (same rule `run_diagnosis` follows), so the page wraps it in `@st.cache_resource`.

**Multipage restructuring (`app.py`, `app_pages/`):** Adding a second, genuinely different screen justified real navigation. `src/frontend/app.py` became a ~25-line `st.navigation`/`st.Page` entry (`app_pages/` per Streamlit's current guidance, not the legacy `pages/` auto-discovery it conflicts with); the former single-page trace view body moved to `app_pages/trace_view.py` unchanged aside from removing its own now-illegal second `st.set_page_config` call and reading `st.session_state["preselect_trace_id"]` to default its trace selectbox when arriving from the dashboard's "View full trace" button. `streamlit run src/frontend/app.py` is unchanged.

**Confidence scoring runs automatically, not gated like root-cause diagnosis:** Citation verification and completeness judging fire on every `ask_question` call — the confidence breakdown is the dashboard's core displayed output, not an optional deep-dive, so gating it behind a second click would mean shipping an answer with no confidence signal by default.

**Fallback wiring:** When `build_fallback_response` fires (retrieval confidence below `settings.retrieval_confidence_threshold`), the dashboard shows the fallback message instead of the generated answer, but `QueryResult.answer_text`/`citation_results`/`confidence` are computed and persisted regardless — engineers can see what the model would have answered.

**Hybrid-vs-dense-only comparison:** Toggling it triggers one extra `DenseRetriever.retrieve()` call for a side-by-side chunk list — no second generation call, no second confidence score — cached per-query in `st.session_state` and run outside `collect_spans()` (a UI exploration aid, not part of the canonical persisted request).

**Settings additions:**
- `generation_llm_provider` (default `"anthropic"`), `generation_llm_model` (default `"claude-sonnet-4-5"`), `generation_llm_temperature` (default `0.0`)

**Public API:**

```python
from src.config import settings
from src.frontend.query_service import ask_question, build_hybrid_retriever

bundle = build_hybrid_retriever(settings)  # expensive — cache with @st.cache_resource
result = ask_question("How often does on-call rotate?", bundle.hybrid, settings)

result.answer_text        # raw generated text, always present
result.fallback            # FallbackResponse | None — UI prefers this over answer_text when set
result.citation_results    # list[CitationVerificationResult]
result.confidence          # ConfidenceScore (retrieval / citation / completeness / composite)
result.trace_id            # inspectable in the trace view immediately
```

**Design notes:**
- Citations are rendered as plain `[N]` markers with a row of "jump to source" buttons below (`cited_chunk_indices`, `view_models.py`) rather than true anchor-scroll — Streamlit has no in-page DOM scroll-to primitive. A deliberate, pragmatic reading of "clickable citations linking to source chunk."
- `ask_question`'s exception path persists a `"failure"` trace before re-raising, so a crashed question is still inspectable, not just an opaque error.
- See `docs/DECISIONS.md` (2026-07-15 entry) for the full design rationale, including why confidence scoring isn't gated like root-cause diagnosis and why the hybrid/dense-only comparison stays outside the persisted trace.

---

## 2026-07-12 — Phase 5: Flagging Interface (Complete)

### Mark Any Trace "Bad Output", Confirm or Override the Diagnosis

`docs/PROJECT_SPEC.md` (Phase 5, item 3) asks for a button that lets a user mark any trace as "bad output," runs the backward trace analysis, displays the diagnosis, and lets the user confirm or override it — "the feedback signal for your eval loop." This repurposes the trace view's existing "Diagnose root cause" button (from the entry below) into "Flag as bad output" — one control, not two — and adds a new persistence module, `src/frontend/flags.py`.

**Button repurposed, status gate removed (`app.py`):** The old button only rendered for `trace.status != "success"`. The spec's "mark *any* trace" wording removes that gate entirely — a human can catch a bad output the pipeline's own status/confidence checks missed. `run_diagnosis` itself, and its `st.session_state`-keyed caching, are unchanged; only the sidebar block around it and what happens after a diagnosis is shown are new. This is a deliberate divergence from the diff view (previous entry), which stays gated on `trace.status != "success"` — a different spec sentence for a different feature.

**`flags.py` (`DiagnosisSummary`, `HumanReview`, `FlagRecord`, `save_flag`/`load_flag`):** one JSON file per trace (`data/eval/flags/{trace_id}.json`), mirroring `corrections.py`'s one-file-per-id convention but holding a single record rather than a per-span mapping, since a flag is a property of the whole trace. `DiagnosisSummary` flattens a `DiagnosisResult` into plain `str`/`int` fields (no live `Span`/pydantic objects), the same rationale `EvidenceEntry` already established. `HumanReview` (`confirmed`, `span_id`, `category`, `note`) always carries a complete verdict — confirming copies the algorithm's own span/category across verbatim rather than just setting a boolean, so a downstream reader never branches on `confirmed` to find the human's actual answer. Serialized via `dataclasses.asdict` + manual reconstruction; no new dependency (`dacite`/`cattrs`/`dataclasses-json` are not declared).

**Confirm or override (`app.py`):** "Confirm diagnosis" (shown only when a root cause was found) copies the algorithm's span/category/rationale straight into a `FlagRecord`. "Override diagnosis" is an `st.form` — a root-cause-span selectbox (from `trace.spans`), a failure-category selectbox (`get_args(FailureCategory)`, the existing 7-value taxonomy), and a note `text_area`, batched into one submit-triggered rerun — and is the *only* finalize path when the diagnosis run found no root cause. Persistence happens only on one of these two actions, never from running the diagnosis alone, mirroring `corrections.py`'s "button click persists, nothing else does" rule. A trace with an existing `FlagRecord` shows a compact summary (including the human's note, if any) plus a "Redo review" button instead of the full flow every time.

**Graph-coloring fallback:** `root_cause_span_id` for `build_graph_view_model` still comes from the in-session `DiagnosisResult` first; when that's unavailable (a fresh session with no diagnosis run yet) and a persisted `FlagRecord` exists, `app.py` falls back to `flag_record.diagnosis.root_cause_span_id`. So revisiting an already-flagged trace in a brand-new session still shows the root-cause node red, with zero LLM spend — verified manually (Playwright-driven browser session, real Anthropic judge calls on first flag, then a fresh session against the same trace with no new calls).

**Settings additions:**
- `flagged_traces_dir` (default `Path("./data/eval/flags")`)

**Public API:**

```python
from src.config import settings
from src.tracing.storage import load_trace
from src.frontend.diagnosis_service import run_diagnosis
from src.frontend.flags import (
    FlagRecord, HumanReview, diagnosis_summary_from_result, load_flag, save_flag,
)

trace = load_trace(trace_id, settings.trace_output_dir)
diagnosis_result = run_diagnosis(trace, settings)  # real LLM spend
summary = diagnosis_summary_from_result(diagnosis_result)  # None if no root cause found

record = FlagRecord(
    flagged_at="2026-07-12T00:00:00+00:00",
    diagnosis=summary,
    human_review=HumanReview(confirmed=True, span_id=summary.root_cause_span_id,
                              category=summary.category, note=""),
)
save_flag(trace_id, record, settings.flagged_traces_dir)

# Later, possibly in a different session:
flag_record = load_flag(trace_id, settings.flagged_traces_dir)
```

**Design notes:**
- `Trace`/`Span` gained no new field — a flag is a side-store, same pattern `corrections.py` already established for human-entered data, not a schema change.
- Confirmed via mypy `strict = true` that narrowing `DiagnosisResult.category`/`.evidence_chain` from `.diagnosis is not None` needs an explicit `assert`, not automatic inference (they're independent `Optional` fields on a frozen dataclass) — resolved with two `assert` statements in `diagnosis_summary_from_result` rather than `# type: ignore` comments, verified clean with zero ignores.
- Superseded claims: the "Document Loader" entry's Module Layout snippet (2026-06-28, below) lists `src/frontend/` without `flags.py` and `data/eval/` as "[planned, Phase 6] — EXCEPT eval/corrections/" without mentioning `eval/flags/`. Both are now real: `src/frontend/flags.py` is implemented, and `data/eval/flags/` is actively written by it, same as `eval/corrections/`.
- See `docs/DECISIONS.md` (2026-07-12 entry) for the full design rationale, including why the diff view's `trace.status`-gate and the flag button's now-removed one are a deliberate divergence rather than an inconsistency.

---

## 2026-07-11 — Phase 5: Trace View & Diff View (Complete)

### Streamlit Trace Explorer + Failure Diff View

`src/frontend/` (`streamlit run src/frontend/app.py`) turns a persisted `Trace` into an interactive flow diagram with click-through span detail, on-demand backward root-cause diagnosis (Phase 4's three functions, wired together for the first time by anything in the codebase), and — for failed/degraded traces — a side-by-side diff against a human-entered "expected output." Streamlit was chosen over React specifically because no API/HTTP layer exists yet (`src/api/` is still a Phase 7 placeholder); the app calls `load_trace`/`list_trace_records`/`find_root_cause_span`/`categorize_failure`/`build_evidence_chain` directly as Python functions instead of requiring `src/api/main.py` to be built first.

**Trace view (`app.py`, `view_models.py`, `graph_render.py`, `detail_panel.py`, `diagnosis_service.py`):**
- `build_graph_view_model(trace, root_cause_span_id, low_confidence_threshold) -> TraceGraphViewModel` (`view_models.py`) makes **each span its own node** in `trace.spans` order, connected by sequential edges — not one node per distinct step name, since `Trace.spans` is a flat list with no parent/child field and multiple spans can share one `step` (e.g. `HybridRetriever`'s dense/sparse legs are both `"retrieval"`).
- `node_status(span, root_cause_span_id, low_confidence_threshold) -> NodeStatus` colors a node red if it matches the current diagnosis's root-cause span (always wins), else yellow if `confidence_score <= low_confidence_threshold`, else green — no LLM call, so opening any trace is free.
- The node graph renders via `streamlit_flow` (`streamlit-flow-component`, new `frontend` extra), chosen over `streamlit-agraph` after a Context7 lookup showed the latter has no documentation coverage there (a maintenance-signal red flag). `graph_render.render_graph` returns the clicked `span_id`; `detail_panel.render(span, status, order)` shows input/output/LLM prompt/confidence/latency/tokens/error, plus an explicit "embeddings not captured" note for retrieval/ranking spans (`Span` has no embeddings field — extending it would mean touching instrumentation across every retrieval/ranking provider, out of scope for a UI-only task).
- Root-cause coloring requires an explicit "Diagnose root cause" click — real LLM spend, per this doc's cost-management guidance — which calls `diagnosis_service.run_diagnosis(trace, settings)`: `find_root_cause_span` → `categorize_failure` → `build_evidence_chain`, short-circuiting the latter two when no root cause is found. The result is cached in `st.session_state` keyed by `trace_id`, so revisiting the same trace within a session doesn't re-spend. This is the first code in the repo to wire all three Phase 4 functions together end-to-end (see the "Superseded claims"/design notes on the Phase 4 entries above, which previously said nothing calls them automatically).

**Diff view (`diff_panel.py`, `corrections.py`, plus additions to `view_models.py`):** the project spec asks for a per-span diff against "the golden dataset or human correction" — neither existed in code (`src/evaluation/` is still a 1-line Phase 6 placeholder, `data/eval/` was empty). This entry implements only the "human correction" half:
- `corrections.py`: `save_correction`/`load_correction(trace_id, span_id, corrections_dir)` persist a human-typed expected output as one JSON file per trace (`data/eval/corrections/{trace_id}.json` → `{span_id: expected_output}`), mirroring `src.tracing.storage`'s one-file-per-id convention.
- `view_models.py` additions: `DiffSegment` (`text`, `tag: Literal["equal", "expected_only", "produced_only"]`), `SpanDiffViewModel`, and `build_span_diff_view_model(span, expected_output) -> SpanDiffViewModel` — a word-level diff via stdlib `difflib.SequenceMatcher` over whitespace-preserving tokens (the same technique `difflib.HtmlDiff`/`git diff --word-diff` use, no new dependency). Two independent segment lists are built from one set of opcodes: `expected_segments` from the `a`-side ranges, `produced_segments` from the `b`-side ranges — letting the UI show a clean "removed" side and "added" side instead of one interleaved diff. Returns `None` segments when no correction has been entered, rather than diffing against `""`.
- `diff_panel.render(span, trace_id, settings)`: three columns (received / produced / should-have-produced). Computes the diff from the **live** text-area value, not the last-saved value, so highlighting updates immediately on blur rather than requiring a save round-trip; "Save correction" (`st.toast(...)` on click, no `st.rerun()`) is purely for persistence. Segments render via `st.html()` — not `st.markdown(unsafe_allow_html=True)`, which would run the diff text through Markdown parsing first and corrupt literal `*`/`_` characters — each segment's text passed through `html.escape()` first (untrusted span/correction text), wrapped in a `white-space:pre-wrap` container so whitespace-only divergence doesn't visually collapse. `st.html` postdates the project's original `streamlit>=1.38` floor, so `pyproject.toml`'s `frontend` extra now requires `streamlit>=1.41`.

**Settings additions:**
- `human_corrections_dir` (default `Path("./data/eval/corrections")`)

**Public API:**

```python
from src.config import settings
from src.tracing.storage import load_trace
from src.tracing.index import list_trace_records
from src.frontend.view_models import build_graph_view_model, build_span_diff_view_model
from src.frontend.diagnosis_service import run_diagnosis
from src.frontend.corrections import load_correction

trace = load_trace(trace_id, settings.trace_output_dir)
view_model = build_graph_view_model(trace, root_cause_span_id=None, low_confidence_threshold=settings.root_cause_quality_threshold)

diagnosis_result = run_diagnosis(trace, settings)  # real LLM spend
if diagnosis_result.diagnosis is not None:
    correction = load_correction(trace_id, diagnosis_result.diagnosis.root_cause_span.span_id, settings.human_corrections_dir)
    diff = build_span_diff_view_model(diagnosis_result.diagnosis.root_cause_span, correction)
```

**Design notes:**
- Not the full Phase 6 golden dataset (50+ hand-written Q&A pairs, automated eval metrics, regression tracking) — that remains a materially larger, separate concern. The correction store is real, reusable infrastructure Phase 6 item 4 ("auto-generate eval cases from production flags") can read later, without scope-creeping into Phase 6 itself now.
- Widget-key design verified via Context7 against Streamlit's own docs on `session_state` widget-key behavior: `diff_panel.py`'s text-area key is scoped to `f"expected_output::{trace_id}::{span_id}"` (both globally unique), so there's no cross-span stale-value collision of the kind `docs/DECISIONS.md`'s streamlit_flow `key=` gotcha (2026-07-09 entry, above) already hit once in `graph_render.py`.
- A post-implementation review (documented in `docs/DECISIONS.md`, 2026-07-11) found and fixed three issues after the initial pass: the `st.markdown`/Markdown-parsing bug above, the whitespace-collapse bug above, and a UX gap where the diff only updated after clicking Save rather than live.
- Only `diagnosis_service.py` imports from `src/analysis/` — isolates the real LLM-spend surface to one module, so it's auditable at a glance which frontend code can trigger judge calls.
- `view_models.py`, `corrections.py`, and `diagnosis_service.py` (with fake judges, same pattern as `tests/unit/analysis/`) are unit tested (`tests/unit/frontend/`); `graph_render.py`/`detail_panel.py`/`diff_panel.py`/`app.py` are Streamlit UI, verified manually via `streamlit run src/frontend/app.py` (including a Playwright-driven browser session during the post-implementation review, to confirm the `st.html`/whitespace/live-diff fixes actually rendered correctly, not just passed lint/mypy).

---

## 2026-07-09 — Phase 4: Evidence Chain Narrative (Complete)

### LLM-as-Judge Causal Narrative Synthesis

Given the `RootCauseDiagnosis` that `find_root_cause_span` (Phase 4, Task 1) produces and the `FailureCategoryVerdict` that `categorize_failure` (Phase 4, Task 2, below) produces, `build_evidence_chain` synthesizes a structured causal explanation — e.g. "Retrieval ranked the most relevant chunk at position 7 instead of position 1. This propagated to Generation, which selected from the top 5 and missed the answer" — plus the ordered input/output evidence backing it. This is the direct continuation of the first two tasks: root-cause finds *which span* broke the pipeline, categorization names *what kind* of failure it was, and this narrates *how it happened*.

**`EvidenceEntry`** (`src/analysis/evidence_chain.py`) — frozen dataclass (`step`, `input`, `output`, `score`, `rationale`), a purpose-built, flat type decoupling provider implementations from `root_cause.py`'s `SpanQualityResult`.

**`EvidenceChainVerdict`** — pydantic `BaseModel` with a single `narrative: str` field — unlike `StepQualityVerdict`/`FailureCategoryVerdict`, there's no separate decision field, since the narrative already is the explanation.

**`EvidenceChainJudgeProtocol`** — `narrate(category: FailureCategory, category_rationale: str, chain: list[EvidenceEntry]) -> EvidenceChainVerdict`, `provider_id: str`. `make_evidence_chain_judge(settings)` returns the configured provider, same lazy-import factory pattern as `make_failure_category_judge`.

**Implemented providers:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `anthropic` (default) | `AnthropicEvidenceChainJudge` | `src/analysis/providers/evidence_chain_judge_anthropic.py` | `claude-sonnet-4-5` |
| `openai` | `OpenAIEvidenceChainJudge` | `src/analysis/providers/evidence_chain_judge_openai.py` | `gpt-4o-2024-08-06` |

**`build_evidence_chain(diagnosis, category_verdict, judge) -> EvidenceChain`** — the standalone entry point:
- Reverses `diagnosis.evaluated_spans` (last-executed-first) into chronological, root-cause-first order
- Maps each `SpanQualityResult` to an `EvidenceEntry`
- Calls `judge.narrate(...)` once and assembles the final `EvidenceChain` (`narrative`, `category`, `category_rationale`, `evidence`)
- Adds no span of its own — same reasoning as `find_root_cause_span`/`categorize_failure`; only the provider's `narrate()` call emits a `step="analysis"` span

**Settings additions:**
- `evidence_chain_judge_provider` (default `"anthropic"`)
- `evidence_chain_judge_model` (default `"claude-sonnet-4-5"`)
- `evidence_chain_judge_temperature` (default `0.0`)

**Public API:**

```python
from src.config import settings
from src.analysis.root_cause import find_root_cause_span, make_step_quality_judge
from src.analysis.failure_categorizer import categorize_failure, make_failure_category_judge
from src.analysis.evidence_chain import build_evidence_chain, make_evidence_chain_judge

quality_judge = make_step_quality_judge(settings)
diagnosis = find_root_cause_span(trace, quality_judge, threshold=settings.root_cause_quality_threshold)
if diagnosis:
    category_judge = make_failure_category_judge(settings)
    category_verdict = categorize_failure(diagnosis, category_judge)
    narrative_judge = make_evidence_chain_judge(settings)
    chain = build_evidence_chain(diagnosis, category_verdict, narrative_judge)
    print(chain.narrative)
```

**Design notes:**
- Standalone, directly-callable unit at the module level — no `src/api/` orchestrator yet loads a flagged trace, runs root-cause identification, categorizes it, and narrates the result automatically over HTTP (Phase 7 work). It is wired together end-to-end from the UI though: `src/frontend/diagnosis_service.py`'s `run_diagnosis` calls `find_root_cause_span` → `categorize_failure` → `build_evidence_chain` on an explicit "Diagnose root cause" button click (see the "Phase 5: Trace View & Diff View" entry below).
- LLM-as-judge over a deterministic template, matching every other qualitative synthesis in this codebase — a template mechanically concatenating each span's already-isolated per-span rationale can't produce genuine cross-span causal reasoning.
- Multi-entry nonce wrapping (one shared nonce, indexed tag names `span-{i}-input`/`span-{i}-output`/`span-{i}-rationale`) is the first prompt in this codebase wrapping an unbounded number of untrusted blocks in one call. See `docs/DECISIONS.md` for the full rationale.

---

## 2026-07-09 — Phase 4: Failure-Type Categorization (Complete)

### LLM-as-Judge Failure Classification

Given the `RootCauseDiagnosis` that `find_root_cause_span` (Phase 4, Task 1, above) produces, `categorize_failure` classifies the diagnosed root-cause span into one of the project spec's six failure categories, plus a 7th catch-all. This is the direct continuation of root-cause identification: root-cause finds *which span* broke the pipeline, categorization names *what kind* of failure it was.

**`FailureCategory`** (`src/analysis/failure_categorizer.py`) — `Literal["retrieval_failure", "ranking_failure", "extraction_hallucination", "citation_error", "generation_incomplete", "context_loss", "other"]`. The first six are the project spec's named taxonomy; `"other"` is a 7th value covering root causes from steps the spec's taxonomy doesn't name (`"ingestion"`, `"analysis"`).

**`FailureCategoryVerdict`** — pydantic `BaseModel` with `category: FailureCategory` and `rationale: str`, same structured-output convention as `StepQualityVerdict`.

**`FailureCategoryJudgeProtocol`** — `classify(step: PipelineStep, input: str, output: str, quality_rationale: str) -> FailureCategoryVerdict`, `provider_id: str`. `make_failure_category_judge(settings)` returns the configured provider (anthropic or openai), same lazy-import factory pattern as `make_step_quality_judge`.

**Implemented providers:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `anthropic` (default) | `AnthropicFailureCategoryJudge` | `src/analysis/providers/failure_category_judge_anthropic.py` | `claude-sonnet-4-5` |
| `openai` | `OpenAIFailureCategoryJudge` | `src/analysis/providers/failure_category_judge_openai.py` | `gpt-4o-2024-08-06` |

**`STEP_TO_PLAUSIBLE_CATEGORIES`** — a `dict[PipelineStep, tuple[FailureCategory, ...]]` restricting which categories are valid for a given root-cause span's step (e.g. `"verification"` can only be `"citation_error"`; `"generation"` can be any of `"extraction_hallucination"`, `"generation_incomplete"`, or `"context_loss"`, since the mapping from step to category isn't 1:1). The system prompt states this subset explicitly as a guardrail, instructing the judge to choose only from it; `FAILURE_CATEGORY_CRITERIA` gives the judge the full taxonomy's descriptions so it understands the categories it's choosing between, not just its own.

**`categorize_failure(diagnosis, judge) -> FailureCategoryVerdict`** — the standalone entry point:
- Unpacks `diagnosis.root_cause_span.step/input/output` and `diagnosis.rationale` (the step-quality judge's own explanation from Task 1, passed through as `quality_rationale` — extra classification signal)
- Calls `judge.classify(...)` once and returns its verdict unchanged
- Adds no span of its own — same reasoning as `find_root_cause_span` not being traced itself; only the provider's `classify()` call emits a `step="analysis"` span

**Settings additions:**
- `failure_category_judge_provider` (default `"anthropic"`)
- `failure_category_judge_model` (default `"claude-sonnet-4-5"`)
- `failure_category_judge_temperature` (default `0.0`)

**Public API:**

```python
from src.config import settings
from src.analysis.root_cause import find_root_cause_span, make_step_quality_judge
from src.analysis.failure_categorizer import categorize_failure, make_failure_category_judge

quality_judge = make_step_quality_judge(settings)
diagnosis = find_root_cause_span(trace, quality_judge, threshold=settings.root_cause_quality_threshold)
if diagnosis:
    category_judge = make_failure_category_judge(settings)
    verdict = categorize_failure(diagnosis, category_judge)
    print(f"{diagnosis.root_cause_span.step} failed as {verdict.category}: {verdict.rationale}")
```

**Design notes:**
- Standalone, directly-callable unit at the module level — no `src/api/` orchestrator yet loads a flagged trace, runs root-cause identification, and categorizes the result automatically over HTTP (that's Phase 7 work). It is however wired together end-to-end from the UI: `src/frontend/diagnosis_service.py`'s `run_diagnosis` calls `find_root_cause_span` → `categorize_failure` → `build_evidence_chain` on an explicit "Diagnose root cause" button click (see the "Phase 5: Trace View & Diff View" entry below).
- The narrative evidence-chain builder (Phase 4, Task 3) is implemented separately — see the "Evidence Chain Narrative" section above.

---

## 2026-07-05 — Phase 4: Backward Root-Cause Span Identification (Complete)

### LLM-as-Judge Span Quality Scoring

When a `Trace` is flagged as failed, `find_root_cause_span` walks its spans in reverse execution order, scoring each one's input→output transformation quality to identify where the pipeline broke. The span with the greatest quality drop is the root cause — not the last-executed bad span (which is often a symptom), but the earliest bad span in the contiguous unhealthy tail (where corruption originated).

**`StepQualityVerdict`** (`src/analysis/root_cause.py`) — pydantic `BaseModel` with `score: int` (`ge=1, le=5`) and `rationale: str`. Passed directly to LLM SDKs as structured-output schema, matching `JudgeVerdict`/`CompletenessVerdict` convention.

**`StepQualityJudgeProtocol`** (`src/analysis/root_cause.py`) — `judge(step: PipelineStep, input: str, output: str) -> StepQualityVerdict`, `provider_id: str`. `make_step_quality_judge(settings)` returns the configured provider (anthropic or openai), same lazy-import factory pattern as `make_embedder`/`make_citation_judge`.

**Implemented providers:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `anthropic` (default) | `AnthropicStepQualityJudge` | `src/analysis/providers/step_quality_judge_anthropic.py` | `claude-sonnet-4-5` |
| `openai` | `OpenAIStepQualityJudge` | `src/analysis/providers/step_quality_judge_openai.py` | `gpt-4o-2024-08-06` |

**`STEP_QUALITY_CRITERIA`** — a dict mapping each `PipelineStep` (including new `"analysis"` value) to step-specific criteria text. The judge's system prompt branches per step so it doesn't apply generation criteria to a retrieval failure or vice versa.

**`find_root_cause_span(trace, judge, threshold) -> RootCauseDiagnosis | None`** — the walker function:
- Iterates `trace.spans` in reverse execution order
- Skips any span with `Span.is_gate=True` entirely (no judge call, no candidate update, can never end the walk) — gate spans (`score_confidence`, `build_fallback_response`) mechanically transform already-computed upstream signals and are internally self-consistent by construction, so a judge would score one "reasonable" regardless of whether the upstream data it received was already corrupted; treating one as a healthy boundary could prematurely mask a genuinely corrupted upstream span. See `docs/DECISIONS.md` (2026-07-09, "Gate Spans Can Mask Root-Cause Detection") for the bug this fixes.
- Calls `judge.judge(step, input, output)` per non-gate span
- A span scoring `<= threshold` (default 2) is "unreasonable"
- Remembers the candidate as the root cause; stops when a non-gate span scores `> threshold` (healthy boundary found)
- Returns `None` if no non-gate span is ever at/below threshold (nothing wrong)
- Returns `RootCauseDiagnosis` with `root_cause_span`, `score`, `rationale`, and `evaluated_spans` (the judged unhealthy tail, in reverse-walk order)

**Cascade handling:** Only the contiguous unhealthy tail is judged. If a healthy span is found, earlier spans (before the boundary) are never judged. This ensures the root cause surfaces where the corruption originated, not at downstream symptoms.

**`PipelineStep` extension:** The Literal now has 6 values: `"ingestion"`, `"retrieval"`, `"ranking"`, `"generation"`, `"verification"`, `"analysis"` (new). The judge's own LLM call wraps itself in `span("analysis", ...)`, giving RCA the same observability (prompt/tokens/latency/errors) as every other judge.

**Settings additions:**
- `root_cause_judge_provider` (default `"anthropic"`)
- `root_cause_judge_model` (default `"claude-sonnet-4-5"`)
- `root_cause_judge_temperature` (default `0.0`)
- `root_cause_quality_threshold` (default `2`, `ge=1, le=5`)

**Public API:**

```python
from src.config import settings
from src.analysis.root_cause import find_root_cause_span, make_step_quality_judge

judge = make_step_quality_judge(settings)
diagnosis = find_root_cause_span(trace, judge, threshold=settings.root_cause_quality_threshold)
if diagnosis:
    print(f"Root cause at {diagnosis.root_cause_span.step}: {diagnosis.rationale}")
```

**Design notes:**
- Like `verify_citations`/`score_confidence`/`build_fallback_response`, this is a standalone, directly-callable unit at the module level — no `src/api/` orchestrator yet loads a flagged trace and calls this automatically over HTTP (Phase 7 work). It is wired together end-to-end from the UI though: `src/frontend/diagnosis_service.py`'s `run_diagnosis` calls this on an explicit "Diagnose root cause" button click (see the "Phase 5: Trace View & Diff View" entry below).
- `find_root_cause_span` itself is not wrapped in `@traced` (each `judge.judge()` call already gets its own `span("analysis", ...)` from the provider). Same reasoning as `HybridRetriever` not being separately wrapped.
- Failure-type categorization (Phase 4, Task 2) is implemented separately — see the "Failure-Type Categorization" section below. The narrative evidence-chain builder (Phase 4, Task 3) is implemented separately too — see the "Evidence Chain Narrative" section above.
- Superseded claims: the "Trace/Span Data Models" entry (2026-07-04, below) lists `Span`'s fields without `error: str | None` (set when an instrumented call raised) or `is_gate: bool` (default `False`; the gate-skip flag this entry's walker checks, added specifically to support the logic above). Both fields exist on the current `Span` model in `src/tracing/models.py`.

---

## 2026-07-05 — Phase 3: Trace Persistence — JSON Files + SQLite Index (Complete)

### Writing and indexing completed traces

`Trace` (`src/tracing/models.py`) gained two fields: `timestamp` (UTC-aware `datetime`, `default_factory`) and `final_score` (optional `float`, no `ge`/`le` bounds — `ConfidenceScore.composite`'s three weights are independently `Settings`-configurable and not validated to sum to 1, so the composite it's usually populated from isn't guaranteed to land in `[0,1]`).

**`src/tracing/storage.py`** — `save_trace(trace, output_dir)` / `load_trace(trace_id, output_dir)` read/write one `{trace_id}.json` file per trace via `model_dump_json(indent=2)` / `model_validate_json()`, the same convention as `src/ingestion/storage.py`'s `save_processed`/`load_processed` — but deliberately *without* that function's `shutil.rmtree`: each trace is an independent file, and writing a new one must never delete previously written ones.

**`src/tracing/index.py`** — a SQLite metadata index (raw stdlib `sqlite3`, no ORM; not a pluggable backend so no `Protocol`/factory) with one `traces` table: `trace_id` (PK), `timestamp`, `status`, `final_score`, and `trace_path` (not in the original task's literal column list, but required to actually resolve a metadata row back to its full JSON trace — otherwise the index can't do the one thing it exists for). `init_trace_index()`, `index_trace()` (`INSERT OR REPLACE`, idempotent per `trace_id`), `get_trace_record()`, `list_trace_records()` (filterable by `status`, ordered by `timestamp` desc). An internal `_connection()` context manager wraps `with conn:` (commits/rolls back) in `try/finally: conn.close()`, since `with conn:` alone never closes the connection (confirmed via the stdlib `sqlite3` docs). Timestamps are stored as `isoformat()` text and re-parsed by Pydantic on read, rather than via `sqlite3`'s built-in datetime adapters — sidesteps their deprecation in Python 3.12+.

**`src/tracing/persistence.py`** — `persist_trace(trace, settings)` is the standalone entry point tying the two together: `save_trace` then `index_trace`, returning the JSON path. Same "directly-callable unit, no orchestrator" shape as `verify_citations`/`score_confidence`/`build_fallback_response` — it takes an already-assembled `Trace` rather than calling `collect_spans()` itself, since no per-request orchestrator exists yet to hand it one automatically.

**Settings** gained `trace_output_dir` (default `./data/traces`) and `sqlite_db_path` (default `./data/traces.db`), matching the `raw_data_dir`/`processed_data_dir` convention.

**Design notes:**
- If `save_trace` succeeds but `index_trace` then raises, the exception propagates uncaught and the JSON file is left in place — it's the durable source of truth; a missing index row is repairable later by re-running `index_trace` against it, an already-deleted trace file is not.
- No WAL mode or connection timeout configured for SQLite — each `index.py` call opens/closes its own short-lived connection. Acceptable today since nothing calls `persist_trace` concurrently (no request orchestrator exists yet); revisit if/when one does.
- Superseded claims: the "Document Loader" entry's Module Layout snippet (2026-06-28, below) lists `tracing/` as "context manager, decorator, JSON+SQLite writers [planned]" and `data/traces/` as "[planned]" — both are now implemented by this entry.

---

## 2026-07-05 — Phase 3: Confidence Scoring in Spans (Complete)

### Populating `Span.confidence_score`

The Phase 3 span instrumentation entry below left `Span.confidence_score` unset everywhere — no function produced a discrete 1-5 rating, only continuous 0-1 floats. This closes that gap for every step that has a natural quality signal.

**`confidence_from_score(value)`** (`src/tracing/instrumentation.py`) — a small pure function mapping a continuous `0-1` signal onto the `1-5` scale: `round(clamp(value, 0, 1) * 4) + 1`. Clamping first means an out-of-range input (e.g. a caller-supplied weight set that pushes `ConfidenceScore.composite` above `1.0`) can't violate `Span.confidence_score`'s `ge=1, le=5` constraint.

**`traced()` gained an optional `confidence_fn` parameter** — called with the wrapped function's return value once it succeeds; its result (`int` 1-5, or `None`) becomes `s.confidence_score`. Typed `Callable[[Any], int | None]`, deliberately *not* against `traced()`'s own `T`: an earlier attempt to type it as `Callable[[T], ...]` made mypy mis-solve `T` before it saw the expression applied as a decorator (surfaced as 6 new mypy errors — `Never` return types and over-widened `Sequence` types leaking into `RerankerProtocol` conformance and `HybridRetriever`). See the `DECISIONS.md` entry for the full mechanism.

**`mean_similarity_confidence(hits)`** (`src/retrieval/models.py`) is the shared `confidence_fn` for `DenseRetriever.retrieve`, `SparseRetriever.retrieve`, and all three rerankers' `rerank()` — mean `similarity` across the returned hits, `None` if empty. Takes `list[VectorStoreHit]` specifically (not the broader `Sequence[VectorStoreHit]`), for the same T-solving reason above.

**`reciprocal_rank_fusion` is the one exception** — uses a manual `with span("retrieval", ...)` block instead of `@traced("retrieval")`, and calls `mean_similarity_confidence` itself, inside the function, on `result`. **Updated 2026-07-09** (see `docs/DECISIONS.md`): this section originally justified the manual `span()` block by saying RRF's returned `similarity` was the fused RRF score (`~1/60` scale) rather than a `[0,1]` signal, requiring a separate lookup of "pre-fusion hits" before `similarity` got overwritten. That RRF-score-as-`similarity` behavior was reversed (see the Fusion Layer entry above) — `reciprocal_rank_fusion` now returns each hit's real pre-fusion `similarity` already, so no separate pre-fusion lookup is needed. The manual `span()` block itself is unchanged for a narrower reason: `reciprocal_rank_fusion` needs to compute its own confidence score from hit similarity *inside* the function on `result`, which a `confidence_fn` hook (which only ever sees a call's already-returned value) can express just as well now — the manual block is retained as-is rather than migrated back to `@traced(confidence_fn=...)`, since both are equivalent here and there's no correctness reason to churn it.

**`verify_citations`** derives confidence from the fraction of citations verified `supported=True`. **`score_confidence`** derives it from `ConfidenceScore.composite`. **`build_fallback_response`** gets no confidence score — its result is a threshold gate, not a graded signal of its own.

Updated table of what `traced(step, confidence_fn=...)` now wraps:

| Function | Step | `confidence_fn` |
|---|---|---|
| `DenseRetriever.retrieve`, `SparseRetriever.retrieve` | `retrieval` | `mean_similarity_confidence` |
| `SentenceTransformersReranker.rerank`, `CohereReranker.rerank`, `VoyageReranker.rerank` | `ranking` | `mean_similarity_confidence` |
| `verify_citations` | `verification` | citation-coverage fraction |
| `score_confidence` | `generation` | `ConfidenceScore.composite` |
| `build_fallback_response` | `generation` | none — threshold gate, not a graded signal |

`reciprocal_rank_fusion` (`retrieval`) is no longer in this table — it uses `span()` directly (see above).

**Public API:**

```python
from src.tracing.context import collect_spans

with collect_spans() as spans:
    hits = dense_retriever.retrieve("What is RRF?")
    reranked = reranker.rerank("What is RRF?", hits, top_n=5)

for s in spans:
    print(s.step, s.confidence_score)  # e.g. "retrieval 4", "ranking 5"
```

**Design notes:**
- Linear mapping (not percentile/bucket-based) — none of the three signals mapped here (mean similarity, citation coverage, composite) has a documented distribution to calibrate bucket edges against, so an evenly-spaced map is the least assumption-laden choice.
- Superseded claims: the "Span Instrumentation" entry immediately below originally stated confidence_score is left unset by every function and that `reciprocal_rank_fusion` is one of the `traced()`-wrapped functions — both statements were accurate for that entry's date and are corrected by this one.

---

## 2026-07-04 — Phase 3: Span Instrumentation (Complete)

### Wrapping Pipeline Steps in Spans

`src/tracing/context.py` and `src/tracing/instrumentation.py` give every
retrieval and generation function a way to record itself as a `Span`
without needing an orchestrator yet:

- **`collect_spans()`** (`context.py`) — a context manager holding a
  `ContextVar[list[Span] | None]`. Instrumented calls append their
  completed span to whichever list is active; outside any `collect_spans()`
  block, instrumentation is a no-op.
- **`span(step, input)`** (`instrumentation.py`) — a context manager for
  sites needing to attach detail mid-function (LLM prompt, token count)
  that a decorator can't infer from arguments/return value alone. Used
  directly inside each judge provider's `judge()` method.
- **`traced(step)`** (`instrumentation.py`) — a decorator built on `span()`
  that auto-serializes a function's bound arguments (`self` excluded) and
  return value. Applied as one line above each simpler pipeline function.
  Typed with `ParamSpec`/`TypeVar` (`Callable[[Callable[P, T]], Callable[P, T]]`)
  rather than `Callable[..., Any]` — the untyped version tripped mypy
  strict mode's "untyped decorator" error at the very first `@traced(...)`
  call site, which would have needed a `# type: ignore` at every one of the
  nine call sites this phase adds; the `ParamSpec` signature preserves the
  wrapped function's real parameter and return types instead.

Applied at:

| Function | Step |
|---|---|
| `DenseRetriever.retrieve`, `SparseRetriever.retrieve`, `reciprocal_rank_fusion` | `retrieval` |
| `SentenceTransformersReranker.rerank`, `CohereReranker.rerank`, `VoyageReranker.rerank` | `ranking` |
| `verify_citations`, `AnthropicCitationJudge.judge`, `OpenAICitationJudge.judge` | `verification` |
| `score_confidence`, `AnthropicCompletenessJudge.judge`, `OpenAICompletenessJudge.judge`, `build_fallback_response` | `generation` |

`HybridRetriever.retrieve` itself is not separately wrapped — `Span` has no
parent/child relationship, so instrumenting both it and the four leaf calls
it makes would add a redundant, duplicate `retrieval`-step span.

By contrast, `verify_citations` and `score_confidence` keep their outer
`@traced` wrapper *even though* each judge call they make opens its own
inner span at the same step (`"verification"`/`"generation"`) — unlike
`HybridRetriever`, both functions do real work (short-circuit logic,
arithmetic) that no judge span ever records, so the outer span isn't
redundant. See the `DECISIONS.md` entry "`verify_citations`/`score_confidence`'s
wrapper spans nest with their judge spans, unlike `HybridRetriever`" for the
full rationale.

`Span.confidence_score` is left unset by every function above: none of them
produce a discrete 1–5 rating today (retrieval similarity and
`ConfidenceScore.composite` are continuous 0–1 floats) — populating it is a
future orchestrator's decision.

**Public API:**

```python
from src.tracing.context import collect_spans

with collect_spans() as spans:
    hits = dense_retriever.retrieve("What is RRF?")
    reranked = reranker.rerank("What is RRF?", hits, top_n=5)
# spans now holds a Span for the retrieve() call and one for the rerank() call
```

Still no orchestrator exists to assemble these into a `Trace` per request,
or a JSON/SQLite writer to persist one — those remain future tasks.

---

## 2026-07-04 — Phase 3: Trace/Span Data Models (Complete)

### The Record of What Happened

`Span` and `Trace` (`src/tracing/models.py`) are the data model a future context manager/decorator will populate as pipeline steps run, and a future JSON/SQLite writer will persist. This entry covers only the models — no instrumentation exists yet to construct them automatically.

**`Span`** — one pipeline step's record:

| Field | Type | Description |
|-------|------|--------------|
| `span_id` | `str` | Auto-generated UUID4 hex, unique per instance |
| `step` | `Literal["ingestion", "retrieval", "ranking", "generation", "verification"]` | Which pipeline stage produced this span |
| `input` | `str` | Serialized step input — serialization itself is the future decorator's job |
| `output` | `str` | Serialized step output |
| `llm_prompt` | `str \| None` | The prompt sent, if this step called an LLM |
| `token_count` | `int \| None` | `>= 0`; `None` for non-LLM steps |
| `latency_ms` | `float` | `>= 0.0` |
| `confidence_score` | `int \| None` | `1-5`; optional because not every step naturally produces one (e.g. ingestion) |

**`Trace`** — the complete record for one request:

| Field | Type | Description |
|-------|------|--------------|
| `trace_id` | `str` | Auto-generated UUID4 hex, unique per instance |
| `spans` | `list[Span]` | Defaults to `[]`; a future context manager appends as steps complete |
| `final_output` | `str \| None` | Defaults to `None` (e.g. a failed request may never produce output) |
| `status` | `Literal["success", "failure", "degraded"]` | Required, no default — callers must state the outcome explicitly |

**Public API:**

```python
from src.tracing.models import Span, Trace

span = Span(step="retrieval", input='{"question": "..."}', output='{"hits": [...]}', latency_ms=12.5)
trace = Trace(spans=[span], final_output="The answer is 42.", status="success")

# Round-trips for the future JSON/SQLite writers:
restored = Trace.model_validate_json(trace.model_dump_json())
```

**Design notes:**
- Both are plain (non-frozen) pydantic `BaseModel`s, matching `ProcessedDocument`/`Chunk` in `src/ingestion/models.py` — not the frozen-dataclass convention used for judge-free result values (`FallbackResponse`, `ConfidenceScore`) — because they need free JSON serialization for the not-yet-built writers, the same rationale `ProcessedDocument`/`Chunk` already established.
- `step` and `status` are closed `Literal`s, matching `ChunkingStrategy`'s convention, since both sets are fixed by the project spec.
- Standalone models only — the context manager, decorator, and JSON/SQLite writers described in the project spec are future work, same situation as every Phase 2 generation module before an orchestrator exists.

---

## 2026-07-04 — Phase 2: Graceful Fallback for Low Retrieval Confidence (Complete)

### Structured "Insufficient Information" Response

`build_fallback_response` closes the last gap called out in the confidence-scoring entry below: deciding what to do with a low retrieval-confidence score. It checks `ConfidenceScore.retrieval_confidence` specifically — not the composite — against a new `settings.retrieval_confidence_threshold` (default `0.5`). The composite mixes in citation coverage and answer completeness, which can be low for reasons unrelated to whether the right documents were found; the project spec calls out retrieval confidence by name for this decision.

Below threshold, it returns a frozen `FallbackResponse` instead of `None`:

- `message` — fixed framing text (`FALLBACK_MESSAGE`), parallel to `INSUFFICIENT_CONTEXT_RESPONSE` in `src/generation/prompts.py`
- `retrieved_summary` — one line per retrieved hit (title, section heading, similarity), or an explicit "nothing retrieved" line if `hits` is empty
- `documents_to_check` — deduplicated document identifiers ("worth checking manually"), ordered by descending similarity; hits are identified by `title`, falling back to `"title (source_path)"` only when two hits share a title but different source paths

No LLM call is involved — unlike citation coverage and answer completeness, this dimension only needs arithmetic over data already attached to `VectorStoreHit`, so it's deterministic and free to compute.

**Public API:**

```python
from src.config import settings
from src.generation import build_fallback_response, score_confidence

score = score_confidence(query, answer_text, hits, citation_results, judge, ...)
fallback = build_fallback_response(
    hits, score.retrieval_confidence, settings.retrieval_confidence_threshold
)
if fallback is not None:
    return fallback  # instead of the generated answer
```

**Design notes:**
- `retrieval_confidence >= threshold` → `None` (proceed with generation); this matches the `>=` convention `ChromaVectorStore` already uses for its dedup check (`(1.0 - distances[0]) >= self._threshold`) rather than introducing a new comparison convention.
- `FallbackResponse` is a frozen dataclass, not a pydantic `BaseModel` — there's no LLM structured-output call here to justify the pydantic convention used by `JudgeVerdict`/`CompletenessVerdict`.
- Like citation verification and confidence scoring, this is a standalone, directly-callable unit — the codebase has no generation orchestrator yet to call it automatically after generation.

---

## 2026-07-03 — Phase 2: Answer Confidence Scoring (Complete)

### Composite Confidence Score

`score_confidence` rates a generated answer on three independent dimensions and combines them into one composite score, closing the loop described in the project's "Confidence Scoring" spec.

**Retrieval confidence** — mean `similarity` across the `VectorStoreHit`s actually used for generation (`0.0` if none retrieved). All reranker providers already overwrite `similarity` with their own relevance score before this runs, so the signal is consistent regardless of which reranker (or none) produced the hits.

**Citation coverage** — the fraction of `verify_citations`' `CitationVerificationResult`s with `supported=True` (`0.0` if no citations were found). Pure arithmetic over Phase 2's existing citation verification output — no new LLM call.

**Answer completeness** — the one dimension that needs judgment: whether the answer addresses every part of the question. `CompletenessJudgeProtocol` (`judge(question, answer) -> CompletenessVerdict`, `provider_id`) mirrors `CitationJudgeProtocol` exactly. `make_completeness_judge(settings)` reads `settings.answer_completeness_judge_provider` and returns the matching implementation, same lazy-import factory pattern as `make_citation_judge`.

**Implemented providers:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `anthropic` (default) | `AnthropicCompletenessJudge` | `src/generation/providers/completeness_judge_anthropic.py` | `claude-sonnet-4-5` |
| `openai` | `OpenAICompletenessJudge` | `src/generation/providers/completeness_judge_openai.py` | `gpt-4o-2024-08-06` |

**Public API:**

```python
from src.config import settings
from src.generation import make_completeness_judge, score_confidence

judge = make_completeness_judge(settings)
result = score_confidence(
    query, answer_text, hits, citation_results, judge,
    retrieval_weight=settings.confidence_retrieval_weight,
    citation_weight=settings.confidence_citation_weight,
    completeness_weight=settings.confidence_completeness_weight,
)
print(result.retrieval_confidence, result.citation_coverage, result.answer_completeness, result.composite)
```

**Design notes:**
- `CompletenessVerdict` is a pydantic `BaseModel` (`complete: bool`, `reasoning: str`), not a frozen dataclass — same rationale as `JudgeVerdict`: it's passed directly as the structured-output schema type to both providers' SDKs.
- `score_confidence` takes plain `float` weight parameters with defaults (not a `Settings` object), mirroring `reciprocal_rank_fusion(dense_weight=..., sparse_weight=...)` — the composite is an unnormalized weighted sum, same convention as RRF.
- This module is a standalone, directly-callable unit — like citation verification, there is no generation orchestrator yet to wire it into automatically. The "if retrieval confidence is below threshold, return a structured 'I don't know' response" fallback described in the project spec is explicitly out of scope here; `score_confidence` only returns the score.

---

## 2026-07-03 — Phase 2: Citation Verification (Complete)

### LLM-as-Judge Citation Checking

The grounded prompt (below) asks the generation LLM to cite every claim with `[N]` markers, but nothing verified those citations were honest until now. `verify_citations` closes that gap: it re-reads the model's own answer text, pairs each citation with the chunk(s) it claims to cite, and asks a second LLM call whether the cited text actually supports the claim.

**`parse_citations`** (`src/generation/citation_parser.py`): a v1 regex heuristic, not sentence-boundary NLP. It finds each contiguous run of `[N]` markers (`r"(?:\[\d+\])+"`) and pairs it with the text since the previous run (or start of string) as the claim. Good enough to bound "what text does this citation apply to" without a full parser. **Updated 2026-07-09** (see `docs/DECISIONS.md`): `GROUNDED_SYSTEM_PROMPT` instructs the model to place markers *after* the claim they support, but not every model complies — Haiku in particular will open a sentence with the marker instead (e.g. "According to the context, [1] the rotation is weekly"). When the text preceding a run is empty or ends with a comma — the structural signature of a lead-in with no claim yet — `parse_citations` now treats it as a **leading marker** and scans forward instead, to the nearer of the next sentence-terminal punctuation (`.`/`!`/`?`) or the next citation run, merging that forward text with whatever (possibly empty) text preceded the marker. This is a punctuation-structure signal, not a lead-in-phrase blocklist. Known accepted edge case: if a leading marker's forward scan is capped by another citation run with no terminator in between, that next run's own preceding text is left empty and it too gets treated as leading.

**`CitationJudgeProtocol`** (`src/generation/citation_verifier.py`): `judge(claim, evidence) -> JudgeVerdict`, `provider_id: str`. `JudgeVerdict` is a pydantic `BaseModel` (`supported: bool`, `reasoning: str`) rather than the codebase's usual frozen dataclass — deliberately, so it can be passed straight through as the structured-output schema type to both providers' SDKs (`output_format=JudgeVerdict` for Anthropic, `response_format=JudgeVerdict` for OpenAI). `make_citation_judge(settings)` reads `settings.citation_judge_provider` and returns the matching implementation, same lazy-import factory pattern as `make_embedder`/`make_reranker`.

**Implemented providers:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `anthropic` (default) | `AnthropicCitationJudge` | `src/generation/providers/citation_judge_anthropic.py` | `claude-sonnet-4-5` |
| `openai` | `OpenAICitationJudge` | `src/generation/providers/citation_judge_openai.py` | `gpt-4o-2024-08-06` |

**`verify_citations(answer_text, hits, judge) -> list[CitationVerificationResult]`:** resolves each citation's (1-indexed) chunk indices against `hits`. An index outside `1..len(hits)` is untrusted LLM output — the model referenced a chunk that doesn't exist — and short-circuits to an unsupported result without ever calling the judge. In-range citations get exactly one `judge.judge(claim, evidence)` call each (no batching), with the cited hits' text joined as evidence.

**Public API:**

```python
from src.config import settings
from src.generation import make_citation_judge, verify_citations

judge = make_citation_judge(settings)          # provider chosen by settings.citation_judge_provider
results = verify_citations(answer_text, hits, judge)
for r in results:
    print(r.supported, r.chunk_indices, r.reasoning)
```

**Design notes:**
- Prompt-injection defense reuses the grounded prompt's spotlighting pattern: `build_judge_prompt` wraps both the claim and the evidence in nonce-suffixed `<claim-...>`/`<evidence-...>` tags via `wrap_with_nonce` (extracted from `build_grounded_prompt` for this reuse), so neither the model's own citation text nor untrusted chunk content can forge a closing tag and escape its block.
- Both provider `judge()` implementations raise `RuntimeError` (not a bare `assert`) when the SDK returns no parsed structured output — `assert` is stripped under `python -O`, which would otherwise let a `None` verdict propagate into an opaque `AttributeError` several frames later.
- This module is a standalone, directly-callable unit — the codebase has no generation orchestrator yet (nothing calls an LLM to produce the initial grounded answer), so `verify_citations` currently takes `answer_text` as a plain parameter rather than generating it itself. Wiring it into an end-to-end `ask()` flow is future work.

---

## 2026-07-03 — Phase 1: Cohere & Voyage Reranker Providers (Complete)

### Additional Reranker Providers

Extends the reranker beyond the local `sentence_transformers` cross-encoder to two hosted rerank APIs, mirroring the embedder's multi-provider pattern.

| Provider | Class | File | Default model |
|---|---|---|---|
| `cohere` | `CohereReranker` | `src/retrieval/providers/reranker_cohere.py` | `rerank-v4.0-pro` |
| `voyage` | `VoyageReranker` | `src/retrieval/providers/reranker_voyage.py` | `rerank-2.5` |

Both follow `reranker_sentence_transformers.py`'s contract exactly (`rerank(query, hits, top_n) -> list[VectorStoreHit]`, results mapped back via `dataclasses.replace(hit, similarity=...)`, empty `hits` short-circuits without an API call) and reuse the embedder's existing `cohere_api_key`/`voyage_api_key` settings and `embed-cohere`/`embed-voyage` extras — no new API keys or extras needed.

**Model-default resolution can't reuse `make_embedder`'s prefix trick:** Cohere's model (`rerank-v4.0-pro`) and Voyage's model (`rerank-2.5`) both start with `"rerank-"`, so `make_reranker` can't tell them apart by prefix the way `make_embedder` distinguishes `text-embedding-*` from `voyage-*`. Instead, it applies a provider's own default only when `settings.reranker_model` still equals the sentence_transformers default (`cross-encoder/ms-marco-MiniLM-L6-v2`) — i.e., the user hasn't customized it at all — and otherwise passes the configured value through verbatim, trusting the user set it correctly for whichever provider they chose.

**Public API:**

```python
from src.config import Settings
from src.retrieval.reranker import make_reranker

settings = Settings(reranker_provider="cohere")   # or "voyage"
reranker = make_reranker(settings)
hits = reranker.rerank(query, candidate_hits, top_n=5)
```

**Design notes:**
- Both providers preserve the hosted API's returned order (already sorted descending by relevance) rather than re-sorting locally.
- Voyage's SDK parameter is `top_k`, not `top_n` like Cohere's — an easy name to mis-copy between the two nearly-identical provider modules; each was verified independently against current SDK docs before implementation.

---

## 2026-07-02 — Phase 1: Cross-Encoder Reranker (Complete)

### Second-Pass Reranking

RRF fusion ranks purely by rank position across the dense/sparse lists — it has no view of the actual query text once fusion runs. A reranker adds a precision pass: RRF now widens its own cutoff to a `rerank_candidate_pool` (default 20), and each candidate is re-scored against the literal query text before the final `rerank_top_n` (default 5) is kept.

**`RerankerProtocol`** (`src/retrieval/reranker.py`): `rerank(query, hits, top_n) -> list[VectorStoreHit]`, `provider_id: str`. `make_reranker(settings)` reads `settings.reranker_provider` and returns the matching implementation, lazily importing the provider SDK inside the factory branch — same pattern as `make_embedder`.

**Implemented provider:**

| Provider | Class | File | Default model |
|---|---|---|---|
| `sentence_transformers` (default) | `SentenceTransformersReranker` | `src/retrieval/providers/reranker_sentence_transformers.py` | `cross-encoder/ms-marco-MiniLM-L6-v2` |

LLM-as-judge reranking is documented as a future provider only — not implemented, to avoid duplicating capability Phase 4's LLM-as-judge root-cause analysis will already exercise.

**Settings split:** `rerank_candidate_pool` (new, default 20) is RRF's own cutoff; `rerank_top_n` (existing, default 5, unchanged meaning) is the final number of chunks reaching generation, whether or not a reranker is installed. A `model_validator` enforces `rerank_top_n <= rerank_candidate_pool`.

**`HybridRetriever`** accepts an optional `reranker: RerankerProtocol | None` constructor arg (mirrors `Indexer`'s optional `embedder`/`vector_store`/`bm25_store` pattern). When `reranking_enabled=False` or no reranker was injected, `retrieve()` falls back to slicing the RRF candidate pool directly — today's exact pre-reranker behavior, so no existing caller breaks.

**Public API:**

```python
from src.config import Settings
from src.retrieval.reranker import make_reranker
from src.retrieval.hybrid_retriever import HybridRetriever

settings = Settings()
reranker = make_reranker(settings)          # provider chosen by settings.reranker_provider

retriever = HybridRetriever(dense, sparse, settings, reranker=reranker)
hits = retriever.retrieve("how do I configure chunking?")  # list[VectorStoreHit], len <= rerank_top_n
```

**Design notes:**
- Cross-encoder scores overwrite `VectorStoreHit.similarity` via `dataclasses.replace()` — same convention RRF and BM25 already use at each pipeline stage. Scores are unbounded and model-dependent, not guaranteed to lie in [0, 1].
- Verified live against the real `cross-encoder/ms-marco-MiniLM-L6-v2` model (not just mocked unit tests): a lexically-favored but semantically irrelevant chunk correctly dropped from rank 1 to last after reranking, while the actually-relevant chunk rose from last to first.

---

## 2026-07-02 — Phase 1: Voyage, Gemini, Cohere Embedding Providers (Complete)

### Additional Embedding Providers

Fills in the three providers that were stubbed as "planned" in `Settings`'s `Literal` type and the factory's `NotImplementedError`/`ValueError` branches.

| Provider | Class | File | Default model | Batch size |
|---|---|---|---|---|
| `voyage` | `VoyageEmbedder` | `providers/embedder_voyage.py` | `voyage-3.5` | 128 (Voyage's documented rate-limit-safe batch) |
| `gemini` | `GeminiEmbedder` | `providers/embedder_gemini.py` | `gemini-embedding-001` | 100 (conservative default — no documented fixed limit) |
| `cohere` | `CohereEmbedder` | `providers/embedder_cohere.py` | `embed-v4.0` | 96 (Cohere's documented max per `embed()` call) |

**Gemini targets `google-genai`, not `google-generativeai`** — the unified SDK Google is standardizing on, superseding the legacy package originally pinned as a placeholder in `pyproject.toml`.

Each provider follows the pattern established by `OpenAIEmbedder`: batched calls sized to the provider's own documented limits, an API-key field on `Settings`, and a `make_embedder` branch with the same OpenAI-model-name guard (falls back to the provider's own default if `settings.embedding_model` looks like an OpenAI model name).

---

## 2026-07-02 — ChromaVectorStore: Required Embedder Guard

`ChromaVectorStore` previously accepted `embedder=None` and silently skipped both metadata stamping and the provider/dimension mismatch check when constructed directly — bypassing the guard that `make_vector_store` already enforced. A collection created this way could later be reopened under a mismatched embedding provider with no warning, until a raw ChromaDB dimension error surfaced deep inside a query.

`embedder` is now a required constructor argument, matching `make_vector_store`'s existing contract. Direct construction without an embedder now fails fast at startup instead of corrupting silently at query time.

---

## 2026-06-30 — Phase 1: Embedding & Vector Store Provider Abstraction (Complete)

### Provider Abstraction

`Embedder` and `VectorStore` were refactored from concrete OpenAI/ChromaDB classes into `Protocol`-based interfaces with factory functions, so switching providers is an environment variable change, not a code change.

**`EmbedderProtocol`** (`src/retrieval/embedder.py`): `embed(texts) -> list[list[float]]`, `dimensions: int`, `provider_id: str`. `make_embedder(settings)` reads `settings.embedding_provider` and returns the matching implementation, importing each provider SDK lazily inside the factory branch so installing one provider's optional extra doesn't pull in the others.

**`VectorStoreProtocol`** (`src/retrieval/vector_store.py`): `filter_duplicates`, `upsert`, `query`, `get_by_ids`, `count`. `make_vector_store(settings, embedder)` returns the configured implementation.

**Implemented providers:**

| Provider | Class | File | Requires |
|---|---|---|---|
| `sentence_transformers` (default) | `SentenceTransformersEmbedder` | `src/retrieval/providers/embedder_sentence_transformers.py` | Nothing — already a base dependency |
| `openai` | `OpenAIEmbedder` | `src/retrieval/providers/embedder_openai.py` | `OPENAI_API_KEY`, `pip install -e ".[embed-openai]"` |
| `voyage` | `VoyageEmbedder` | `src/retrieval/providers/embedder_voyage.py` | `VOYAGE_API_KEY`, `pip install -e ".[embed-voyage]"` |
| `gemini` | `GeminiEmbedder` | `src/retrieval/providers/embedder_gemini.py` | `GEMINI_API_KEY`, `pip install -e ".[embed-gemini]"` |
| `cohere` | `CohereEmbedder` | `src/retrieval/providers/embedder_cohere.py` | `COHERE_API_KEY`, `pip install -e ".[embed-cohere]"` |
| `chroma` (default vector store) | `ChromaVectorStore` | `src/retrieval/vector_store.py` | Nothing — already a base dependency |

The `qdrant` vector store is declared in `Settings`'s `Literal` type and `pyproject.toml`'s `store-qdrant` extra but not yet implemented; selecting it raises `NotImplementedError` from `make_vector_store`. All five embedding providers above are fully implemented.

**Dimension guard:** Embedding dimensions vary by provider (e.g. OpenAI `text-embedding-3-small` = 1536, `all-MiniLM-L6-v2` = 384). `ChromaVectorStore` stamps `embedding_provider` and `embedding_dimensions` into the collection's metadata the first time it's created. On every later open, if an `embedder` is passed and its `provider_id` doesn't match the stored metadata, construction raises `ValueError` with a message telling the user to delete `data/chroma/` and re-index — this prevents silently querying a collection with vectors from a different embedding space.

**`embedder_openai.py` model-name guard:** `make_embedder` won't blindly pass `settings.embedding_model` to `SentenceTransformersEmbedder` — if the configured model name starts with `text-embedding` (an OpenAI-style name) and the provider is `sentence_transformers`, it falls back to `SentenceTransformersEmbedder`'s own default (`all-MiniLM-L6-v2`) instead of trying to load an OpenAI model name as a local model, which would fail confusingly.

**Backward-compatible aliases:** `Embedder` and `VectorStore` remain importable (as aliases for `OpenAIEmbedder`/`ChromaVectorStore` respectively) via module-level `__getattr__`, so existing call sites and tests that import the old names keep working without eagerly importing every provider SDK at module load time.

**Public API:**

```python
from src.config import settings
from src.retrieval.embedder import make_embedder
from src.retrieval.vector_store import make_vector_store

embedder = make_embedder(settings)          # provider chosen by settings.embedding_provider
vector_store = make_vector_store(settings, embedder)  # provider chosen by settings.vector_store_provider
```

---

## 2026-06-30 — Phase 1: RRF Fusion + HybridRetriever (Complete)

### Fusion Layer

**`reciprocal_rank_fusion`** (`fusion.py`):

- Combines `dense_hits` and `sparse_hits` into a single ranked list
- Score per chunk: `sum_r: weight_r / (k + rank_r)` where `k = 60` (Cormack et al. 2009)
- Default weights: `dense_weight=0.7`, `sparse_weight=0.3` (configurable via `Settings`)
- When a chunk appears in both lists, scores accumulate — overlap boosts rank
- Dense hit metadata takes priority when a chunk appears in both lists (`hits_by_id[id] = hit` for dense, `setdefault` for sparse)
- Output: `list[VectorStoreHit]` — **reversed 2026-07-09** (see `docs/DECISIONS.md`): each selected hit's original pre-fusion `similarity` (dense cosine, or sparse max-normalized BM25) is now returned unmodified; the RRF score (`~1/60` scale) still drives internal selection/ordering but is never written onto `similarity`. The original 2026-06-30 design overwrote `similarity` with the RRF score via `dataclasses.replace` — that turned out to be exactly the kind of rank-based magnitude RRF's own design says shouldn't be compared as a quality signal, and downstream code (`score_confidence`, `build_fallback_response`) was silently misreading it as a `[0,1]`-scaled value when reranking was disabled.
- `top_n` limits final output (default: `settings.rerank_top_n = 5`)

**`HybridRetriever`** (`hybrid_retriever.py`):

- Wires `DenseRetriever` + `SparseRetriever` + `reciprocal_rank_fusion` via `Settings`
- `.retrieve(query)` → calls both retrievers with configured `k` → fuses → returns top-N

**Public API:**

```python
from src.retrieval import HybridRetriever, DenseRetriever, SparseRetriever, BM25Store
from src.retrieval.embedder import make_embedder
from src.retrieval.vector_store import make_vector_store
from src.config import Settings

settings = Settings()
embedder = make_embedder(settings)
vector_store = make_vector_store(settings, embedder)
dense = DenseRetriever(embedder, vector_store)
sparse = SparseRetriever(BM25Store(settings), vector_store)

retriever = HybridRetriever(dense, sparse, settings)
hits = retriever.retrieve("how do I configure chunking?")  # list[VectorStoreHit], len ≤ rerank_top_n
```

**Design notes:**
- `reciprocal_rank_fusion` is a pure function with no dependencies on retriever internals — independently testable (11 unit tests, no mocks)
- `HybridRetriever` contains no scoring logic; tests mock both retrievers and verify wiring only (8 tests)
- **Superseded 2026-07-09** (see `docs/DECISIONS.md`): RRF scores no longer replace `similarity`. `reciprocal_rank_fusion` now returns each selected hit's original, unmutated pre-fusion object — frozen `VectorStoreHit` instances still are never mutated, but there's no `dataclasses.replace()` call for `similarity` at all anymore, since the RRF score is used only internally for selection/ordering.

---

## 2026-06-30 — Phase 1: Hybrid Retrieval — Dense & Sparse Retrievers (Complete)

### Retrieval Query Path

Two retriever classes implement the query side of hybrid retrieval. Both return `list[VectorStoreHit]`.

**`VectorStoreHit`** — shared result model for all retrieval paths:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | Content-addressed chunk ID |
| `text` | `str` | Chunk text |
| `doc_id` | `str` | Source document ID |
| `source_path` | `str` | Path to original file |
| `title` | `str` | Document title |
| `section_heading` | `str \| None` | Nearest section heading |
| `chunk_index` | `int` | 0-based position within document |
| `strategy` | `str` | Chunking strategy used |
| `similarity` | `float` | Relevance score in [0, 1] |

**`DenseRetriever`** (`dense_retriever.py`):
- Embeds the query via `Embedder.embed([query])` (single-item list, unpacked)
- Calls `VectorStore.query(embedding, k)` → ranked by cosine similarity
- Default `k=10`

**`SparseRetriever`** (`sparse_retriever.py`):
- Scores all indexed chunks via `BM25Store.get_scores(query)`
- Sorts by BM25 score descending, takes top-k, discards zero-score chunks
- Normalizes scores by dividing by max score → `similarity` in (0, 1]
- Fetches full chunk metadata from `VectorStore.get_by_ids(ids)` (single round-trip)
- Replaces `similarity` field with the normalized BM25 score via `dataclasses.replace`

**Public API:**

```python
from src.retrieval import DenseRetriever, SparseRetriever, BM25Store
from src.retrieval.embedder import make_embedder
from src.retrieval.vector_store import make_vector_store
from src.config import Settings

settings = Settings()
embedder = make_embedder(settings)
vector_store = make_vector_store(settings, embedder)
bm25_store = BM25Store(settings)

dense = DenseRetriever(embedder, vector_store)
sparse = SparseRetriever(bm25_store, vector_store)

dense_hits = dense.retrieve("how do I configure chunking?", k=10)
sparse_hits = sparse.retrieve("how do I configure chunking?", k=10)
```

**Design notes:**
- `SparseRetriever` depends on `VectorStore` to fetch metadata (text, source_path, title, etc.) — BM25 only stores chunk IDs and corpus tokens, not full metadata. This keeps BM25 persistence lightweight and avoids duplicating metadata.
- Score normalization (`/ max_score`) makes BM25 scores comparable to cosine similarity scores, which is required for RRF fusion in the next step.
- If BM25 has no scores or all scores are 0.0, `SparseRetriever` returns `[]` without touching ChromaDB.

---

## 2026-06-29 — Phase 1: Retrieval Indexing (Complete)

### Retrieval Module

The retrieval module (`src/retrieval/`) handles embedding, vector storage, BM25 indexing, and orchestrated ingestion into both indexes.

**Public API:**

```python
from src.retrieval import BM25Store, Indexer
from src.config import Settings

settings = Settings()
indexer = Indexer(settings)          # builds its own embedder/vector store via the factories, loads existing BM25 index from disk on init

# Full pipeline: embed → dedup → parallel upsert → save BM25
stored_ids = indexer.index(chunks)   # chunks: list[Chunk] from src.ingestion
```

**Classes:**

| Class | File | Responsibility |
|-------|------|----------------|
| `EmbedderProtocol` / `make_embedder` | `embedder.py` | Provider-agnostic embedding interface + factory (see provider abstraction entry above) |
| `OpenAIEmbedder` | `providers/embedder_openai.py` | Batch OpenAI embeddings (`BATCH_SIZE = 200`) |
| `SentenceTransformersEmbedder` | `providers/embedder_sentence_transformers.py` | Local embeddings via `sentence-transformers`, no API key |
| `VectorStoreProtocol` / `make_vector_store` | `vector_store.py` | Provider-agnostic vector store interface + factory |
| `ChromaVectorStore` | `vector_store.py` | ChromaDB persistent client, cosine-space collection `"rag_chunks"`, upsert + dedup + dimension guard |
| `BM25Store` | `bm25_store.py` | BM25Okapi index, cumulative add, pickle persistence, scored retrieval |
| `Indexer` | `indexer.py` | Orchestrator: embed → dedup → parallel upsert (ThreadPoolExecutor) → save BM25 |

**Deduplication:** `VectorStore.filter_duplicates` queries ChromaDB for the nearest neighbour of each incoming embedding. Chunks where `(1 - cosine_distance) >= settings.dedup_threshold` (default 0.95) are excluded from both stores. Fast-paths to accept all when the collection is empty.

**Sync guarantee:** `Indexer.index` applies deduplication before touching either store, so ChromaDB and BM25 always receive the same accepted set. The `ThreadPoolExecutor` `with` block is exited (which calls `shutdown(wait=True)`) before either `.result()` is collected — ensuring both threads complete before any exception propagates or `bm25_store.save()` is called.

**BM25 persistence:** `BM25Store` pickles `{"chunk_ids": list[str], "corpus": list[list[str]]}` to `data/bm25_index.pkl` (sibling of `data/chroma/`). On load it reconstructs `BM25Okapi` from the corpus. The `BM25Okapi` object itself is not pickled to avoid library-version compatibility issues.

**ChromaDB metadata** stored per chunk:

| Field | Type | Source |
|-------|------|--------|
| `source_path` | `str` | `chunk.source_path` |
| `chunk_index` | `int` | `chunk.chunk_index` |
| `section_heading` | `str` | `chunk.section_heading or ""` |
| `strategy` | `str` | `chunk.strategy` |
| `char_count` | `int` | `len(chunk.text)` |
| `doc_id` | `str` | `chunk.doc_id` |
| `title` | `str` | `chunk.title` |

---

## 2026-06-28 — Phase 1: Chunking (Complete)

### Chunking Module

The chunker (`src/ingestion/chunker.py`) splits `ProcessedDocument` objects into `Chunk` objects using one of three switchable strategies. Strategy is set via `Settings.chunk_strategy`.

**`Chunk`** — the output of chunking:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str` | SHA-256 of `"{doc_id}:{text}"` — deterministic content address |
| `doc_id` | `str` | Inherited from source `ProcessedDocument` |
| `source_path` | `str` | Inherited |
| `source_format` | `Literal[...]` | Inherited |
| `title` | `str` | Inherited |
| `section_heading` | `str \| None` | Inherited |
| `page_number` | `int \| None` | Inherited |
| `text` | `str` | Chunk text |
| `chunk_index` | `int` | 0-based, continuous across all docs in one `chunk()` call |
| `strategy` | `ChunkingStrategy` | `"fixed_size"`, `"recursive_header"`, or `"semantic"` |
| `processed_at` | `str` | ISO-8601 UTC timestamp |

**Strategies:**

| Strategy | Splitter | Separators | Overlap |
|----------|----------|------------|---------|
| `fixed_size` | `RecursiveCharacterTextSplitter` | Default (`\n\n`, `\n`, ` `, `""`) | Yes |
| `recursive_header` | `RecursiveCharacterTextSplitter` | `\n\n`, `\n`, `. `, `! `, `? `, ` `, `""` | Yes |
| `semantic` | Custom (cosine distance on embeddings) | Sentence boundaries | No |

The semantic strategy calls `openai.embeddings.create` for each multi-sentence document. The OpenAI client is lazily instantiated — single-sentence documents skip the API entirely.

**Public API:**

```python
from src.ingestion import DocumentLoader, Chunker, Chunk, ChunkingStrategy, chunk_id

loader = DocumentLoader(settings)
docs = loader.load(Path("data/raw/guide.pdf"))

chunker = Chunker(settings)          # strategy read from settings.chunk_strategy
chunks: list[Chunk] = chunker.chunk(docs)
```

---

## 2026-06-28 — Phase 1: Document Loader (Complete)

### Module Layout

```
src/
  config.py           # Central settings via pydantic-settings (singleton)
  ingestion/
    models.py         # ProcessedDocument, Chunk, ChunkingStrategy, chunk_id
    chunker.py        # Chunker — three switchable chunking strategies
    loader.py         # DocumentLoader — dispatches by file extension
    storage.py        # save_processed / load_processed / list_raw_files
  retrieval/          # EmbedderProtocol/make_embedder, VectorStoreProtocol/make_vector_store,
                      # BM25Store, Indexer (complete)
                      # providers/        # embedder_openai.py, embedder_sentence_transformers.py,
                      #                   # embedder_voyage.py, embedder_gemini.py, embedder_cohere.py
                      #                   # (all complete; qdrant vector store still planned)
                      # DenseRetriever, SparseRetriever, VectorStoreHit (complete)
                      # reciprocal_rank_fusion, HybridRetriever (complete)
                      # RerankerProtocol/make_reranker, SentenceTransformersReranker (complete)
  generation/         # Grounded prompt, citation parser/verifier, confidence scorer, fallback
                      # response (complete; no generation orchestrator wiring them together yet)
  tracing/            # Trace/Span models (complete); context manager, decorator, JSON+SQLite
                      # writers (complete)
  analysis/           # StepQualityJudgeProtocol, make_step_quality_judge factory, 
                      # find_root_cause_span walker (complete)
                      # FailureCategoryJudgeProtocol, make_failure_category_judge factory,
                      # categorize_failure (complete)
                      # EvidenceChainJudgeProtocol, make_evidence_chain_judge factory,
                      # build_evidence_chain (complete)
                      # providers/        # step_quality_judge_anthropic.py, step_quality_judge_openai.py,
                      #                   # failure_category_judge_anthropic.py, failure_category_judge_openai.py,
                      #                   # evidence_chain_judge_anthropic.py, evidence_chain_judge_openai.py
  evaluation/         # Golden dataset runner, metric calculators, regression tracker [planned, Phase 6]
  api/                # FastAPI app, route handlers [planned, Phase 7]
  frontend/           # Streamlit trace view + diff view (complete; see "Phase 5: Trace View &
                      # Diff View" entry, below) — app.py, view_models.py, graph_render.py,
                      # detail_panel.py, diagnosis_service.py, diff_panel.py, corrections.py
scripts/
  seed_corpus.py      # Index sample docs for local testing [stub, Phase 1 follow-up]
  run_eval.py         # Execute full eval suite and print metrics [stub, Phase 6]
tests/
  fixtures/           # Sample files (sample.md, sample.txt, sample.html) + PDF generator
  unit/ingestion/     # Unit tests for models, loader, storage
  unit/retrieval/     # Unit tests for Embedder, VectorStore, BM25Store, Indexer
  unit/frontend/      # Unit tests for view_models, diagnosis_service, corrections (graph_render.py/
                      # detail_panel.py/diff_panel.py/app.py are Streamlit UI, verified manually)
  integration/        # End-to-end pipeline tests
data/
  raw/                # Source documents (original, untouched)
  processed/          # Normalised plaintext + metadata (one JSON per section/page)
  chroma/             # ChromaDB file-based persistence
  bm25_index.pkl      # BM25 index (pickled corpus + chunk_ids, rebuilt on load)
  traces/             # JSON trace files (one per request) (complete — superseded the [planned]
                      # marker this line originally had; see the "Superseded claims" note at the
                      # top of this doc's Trace Persistence entry)
  eval/               # Golden Q&A dataset and flagged failure cases [planned, Phase 6] — EXCEPT
                      # eval/corrections/, which is real and actively written by
                      # src/frontend/corrections.py (one JSON file per trace_id, human-entered
                      # per-span "expected output" corrections for the diff view)
```

### Ingestion Module

The ingestion module (`src/ingestion/`) is the entry point for all content. It accepts `.md`, `.txt`, `.html`, and `.pdf` files, normalises them to clean plaintext, and attaches structured metadata.

**`ProcessedDocument`** is the single output type for all formats:

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | SHA-256 of raw file bytes — stable across re-ingestion |
| `source_path` | `str` | Path to the original file |
| `source_format` | `Literal[...]` | `"markdown"`, `"text"`, `"html"`, or `"pdf"` |
| `title` | `str` | First heading found, or filename stem |
| `section_heading` | `str \| None` | Nearest preceding heading; `None` for plain text and PDF |
| `page_number` | `int \| None` | 1-indexed page number for PDF; `None` otherwise |
| `text` | `str` | Clean plaintext — no markup |
| `processed_at` | `str` | ISO-8601 UTC timestamp |

**`DocumentLoader.load(path)`** dispatches by file extension:

| Format | Splits on | `section_heading` | `page_number` |
|--------|-----------|-------------------|---------------|
| `.md` | `#` / `##` / `###` headings | Nearest preceding heading | `None` |
| `.txt` | Whole file | `None` | `None` |
| `.html` / `.htm` | `<h1>`–`<h6>` tags | Nearest preceding heading | `None` |
| `.pdf` | Pages | `None` | 1-indexed |

**Storage** mirrors the source path under `data/processed/`:

```
data/raw/guide.pdf          → data/processed/guide.pdf/page_001.json
data/raw/setup.md           → data/processed/setup.md/section_000.json
```

Re-ingesting a file overwrites its output directory. Because `doc_id` derives from raw bytes, the ID is identical every time the same file is loaded — safe for downstream deduplication checks.

**Public API:**

```python
from src.ingestion import DocumentLoader, Chunker, save_processed, load_processed, list_raw_files

loader = DocumentLoader(settings)
docs = loader.load(Path("data/raw/guide.pdf"))
save_processed(docs, source_raw_path, settings.processed_data_dir)

# Re-index without re-upload:
docs = load_processed(source_raw_path, settings.processed_data_dir)
```

### Pipeline Flow

```
Document → Ingestion → Chunking → Embedding → [ChromaDB | BM25 Index]
                                                        ↓
User Question → Embed → Dense Retrieval ─┐
                      → Sparse Retrieval ─┤→ RRF Fusion → Reranker → Top-5 Chunks
                                                                           ↓
                                                              LLM Generation + Citations
                                                                           ↓
                                                         Citation Verification + Confidence Score
                                                                           ↓
                                                                    Final Answer
```

Every request is wrapped in a **Trace** (`trace_id`) containing **Spans** — one per pipeline step. Spans capture input, output, LLM prompt, token count, latency, and confidence score (1–5). (No orchestrator yet assembles a `Trace` automatically for a live request — see the Trace Persistence entry above.)

**Backward failure diagnosis and the trace view**, layered on top of a persisted `Trace` (not shown in the forward-flow diagram above, since neither runs during the request itself):

```
Persisted Trace (JSON + SQLite index)
        ↓
[Trace view: color-coded flow graph, click-through span detail — streamlit run src/frontend/app.py]
        ↓ ("Flag as bad output" button, on demand — real LLM spend, on any trace)
find_root_cause_span → categorize_failure → build_evidence_chain
        ↓
[Diagnosis shown; human Confirms or Overrides → FlagRecord persisted — src/frontend/flags.py]
        ↓
[Diff view: received / produced / should-have-produced, per span, with a human-correction input]
```

See the "Phase 5: Flagging Interface" and "Phase 5: Trace View & Diff View" entries above for the frontend implementation, and the Phase 4 entries above for the three analysis functions.

### Configuration

All runtime configuration flows through `src/config.py` (`Settings` class, pydantic-settings). Values are read from environment variables or `.env`. The module exposes a singleton `settings` object; tests instantiate `Settings()` directly to allow monkeypatching. See `.env.example` for the full variable reference.
