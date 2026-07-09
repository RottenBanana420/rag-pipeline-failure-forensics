## Project: RAG Pipeline with Integrated Failure Forensics

**What You're Building:** A production-grade Retrieval-Augmented Generation system that ingests internal documentation with hybrid search, retrieves the most relevant context, generates grounded answers with inline citations, and includes a built-in observability layer that traces every intermediate step, identifies exactly where failures originate when outputs degrade, and feeds flagged failures back into a growing evaluation dataset.

> **Why This Project Lands Interviews**
> RAG is the single most requested skill in AI engineering job descriptions, but most candidates build toy demos. You're building a system with hybrid retrieval, chunking strategy decisions, and production observability—the difference between someone who followed a LangChain quickstart and a real AI engineer. You're also demonstrating observability-first thinking: understanding that when multi-step AI pipelines produce garbage, most teams have no idea which step broke. Being able to articulate why tracing matters for production RAG systems and how to diagnose failures at the component level is a senior-level signal.

### Tech Stack

| Component | Tool / Library | Why This Choice |
|---|---|---|
| Language | Python 3.11+ | Standard for ML tooling and ecosystem |
| Embeddings | OpenAI text-embedding-3-small | Cost-effective, high quality |
| Vector Store | ChromaDB or Qdrant | File-based or containerized flexibility |
| Sparse Search | BM25 via rank_bm25 | Keyword matching for exact terms |
| LLM | GPT-4o or Claude Sonnet | Strong grounding and citation capability |
| Chunking | LangChain text splitters | Configurable overlap and size |
| Tracing | OpenTelemetry + custom spans | Industry-standard observability |
| Storage | SQLite + JSON trace files | Simple, inspectable, git-friendly |
| API | FastAPI | Async-native, production-grade |
| Visualization | React frontend or Streamlit | Interactive trace explorer and query interface |
| Feedback | Simple REST API | Humans flag bad outputs for analysis |
| Containerization | Docker | Production packaging |

### Step-by-Step Build Guide

#### Phase 1: Build the Ingestion, Chunking & Hybrid Retrieval Pipeline (Day 1–6)

**Ingestion and Chunking (Days 1–3)**

1. **Build a multi-format document loader:** Accept markdown, text, HTML, and PDF files. Normalize everything into clean plaintext with metadata (source file, section heading, page number). Store raw documents alongside processed versions so you can re-index without re-uploading.

2. **Implement configurable chunking:** Build three chunking strategies and make them switchable: fixed-size with overlap (baseline), recursive character splitting by section headers (structure-aware), and semantic chunking that splits on topic boundaries using embedding similarity. Track which strategy each chunk used.

3. **Generate and store embeddings:** Embed every chunk using text-embedding-3-small. Store in ChromaDB with metadata: source document, chunk index, section heading, chunking strategy, and character count. Build the BM25 index in parallel over the same chunks. Both indexes must stay in sync.

4. **Add deduplication:** Before inserting a chunk, check for near-duplicates (cosine similarity > 0.95 against existing chunks). Flag and skip duplicates. This prevents the retriever from wasting context window slots on redundant content when the same information appears in multiple docs.

**Hybrid Retrieval Engine (Days 3–6)**

1. **Implement dense retrieval:** Query the vector store with the embedded user question. Return the top-k chunks ranked by cosine similarity. Start with k=10.

2. **Implement sparse retrieval:** Run the same query through BM25 over the chunk corpus. Return top-k by BM25 score. This catches exact keyword matches that semantic search might miss — critical for technical documentation with specific function names, config keys, or error codes.

3. **Build the fusion layer:** Implement Reciprocal Rank Fusion (RRF) to combine dense and sparse results into a single ranked list. RRF assigns scores based on rank position across both lists and merges them. Make the weighting configurable (e.g., 0.7 dense / 0.3 sparse) so you can tune it per use case.

4. **Add a reranker:** After fusion, send the top 20 candidates through a cross-encoder reranker (use a small model or LLM-as-judge) that scores each chunk's relevance to the actual question. Keep the top 5. This second pass dramatically improves precision and is a strong interview talking point.

#### Phase 2: Build Generation with Citations & Grounded Answers (Day 6–9)

1. **Design the grounded generation prompt:** Construct a system prompt that instructs the LLM to answer only from the provided context, cite specific chunks using bracketed references ([1], [2]), and explicitly state when the context doesn't contain enough information to answer. Include the retrieved chunks as numbered context blocks.

2. **Implement citation verification:** After generation, parse the model's citations and verify each one. Does [1] actually support the claim it's attached to? Send each citation-claim pair to an LLM-as-judge for verification. Flag unsupported citations. This is the quality layer most RAG systems skip entirely.

3. **Build the answer confidence scorer:** Score each answer on: retrieval confidence (how relevant were the top chunks?), citation coverage (what percentage of claims have verified citations?), and answer completeness (did the response address all parts of the question?). Return a composite confidence score alongside the answer.

4. **Handle the "I don't know" case gracefully:** If retrieval confidence is below a threshold, don't hallucinate. Return a structured response that says what the system found, what it couldn't find, and which documents might be worth checking manually. This is more useful than a fabricated answer and signals production maturity.

#### Phase 3: Build the Tracing & Instrumentation Layer (Day 9–12)

1. **Create a Trace object:** Every RAG request gets a unique `trace_id`. The Trace contains: a list of Span objects (one per pipeline step: ingestion, retrieval, ranking, generation, verification), the final output, and a status (success/failure/degraded). This becomes your complete record of what happened.

2. **Instrument each retrieval and generation step with spans:** Wrap each pipeline component in a context manager that automatically captures: step name, input (serialized), output (serialized), LLM prompt sent (if applicable), LLM raw response, token count, latency, and any errors. Use a decorator pattern so instrumenting a new step is one line of code.

3. **Add confidence scoring at each step:** After each component (retrieval, ranking, generation), output a confidence score (1–5) for its own result. Store this in the span. When you're tracing backward from a failure, low-confidence spans are your primary suspects.

4. **Store traces as structured JSON:** Write each complete trace to a JSON file and index it in SQLite (trace_id, timestamp, status, final_score). This gives you both human-readable traces and queryable metadata. Ensure spans include enough context to reconstruct what happened without re-running.

#### Phase 4: Build Backward Trace Analysis for Failure Diagnosis (Day 12–15)

1. **Implement root cause analysis logic (COMPLETE):** `find_root_cause_span` walks a failed `Trace` backward through its spans using an LLM-as-judge (`StepQualityJudgeProtocol`) to score each step's input→output transformation quality (1–5 scale). The first span with a significant quality drop (`<= threshold`, default 2) is the root cause. Handles cascade failures correctly: only judges the contiguous unhealthy tail, returning the earliest bad span (where corruption originated), not the last-executed bad span (a symptom). Two providers implemented: Anthropic (claude-sonnet-4-5) and OpenAI (gpt-4o-2024-08-06). Settings fields and new `"analysis"` PipelineStep value added. See [ARCHITECTURE.md](ARCHITECTURE.md) (Phase 4 entry) and [CLAUDE.md](../CLAUDE.md) (Module Layout: `src/analysis/` and root-cause decision section) for full details.

2. **Categorize failure types in RAG context (COMPLETE):** `categorize_failure` classifies a `RootCauseDiagnosis` (from Task 1) into a taxonomy specific to this pipeline: Retrieval Failure (wrong documents retrieved or ranked too low), Ranking Failure (reranker deprioritized the right answer), Extraction Hallucination (generation added facts not in source chunks), Citation Error (claims not actually supported by cited chunks), Generation Incomplete (answer was cut off or didn't address all parts), Context Loss (important information from retrieval was not used by generator), or Other (root causes from steps — ingestion, analysis — outside this six-item taxonomy). Classification is delegated to an LLM-as-judge (`FailureCategoryJudgeProtocol`), guarded by `STEP_TO_PLAUSIBLE_CATEGORIES` so the judge only picks from the category subset plausible for the root-cause span's step. Two providers implemented: Anthropic (claude-sonnet-4-5) and OpenAI (gpt-4o-2024-08-06). See [ARCHITECTURE.md](ARCHITECTURE.md) (Phase 4: Failure-Type Categorization entry) and [CLAUDE.md](../CLAUDE.md) (Module Layout: `src/analysis/` and failure categorization decision section) for full details.

3. **Build the evidence chain (PLANNED):** For each diagnosed failure, produce a structured explanation: "Retrieval ranked the most relevant chunk at position 7 instead of position 1. This propagated to Generation, which selected from the top 5 and missed the answer." Include the specific input/output pairs as evidence. This narrative helps both you and reviewers understand the diagnosis.

#### Phase 5: Build Visual Explorers & Interactive Interfaces (Day 15–18)

1. **Create the trace view:** A visual representation of the pipeline where each step is a node. Color-code by status: green (healthy), yellow (low confidence), red (identified root cause). Clicking a node shows the full span details — input, output, embeddings, LLM prompt (if any), confidence score.

2. **Add the diff view:** For failed traces, show a side-by-side comparison: what the step received vs. what it produced vs. what it should have produced (based on the golden dataset or human correction). Highlight the specific divergence. This makes diagnosis instant.

3. **Build the flagging interface:** A simple button that lets a user mark any trace as "bad output." When clicked, the system runs the backward trace analysis and displays the root cause diagnosis. The user can confirm or override the diagnosis. This creates the feedback signal for your eval loop.

4. **Build the query dashboard:** A Streamlit or React frontend where you can ask questions and see: the generated answer with clickable citations (each citation links to the source chunk), the retrieved chunks ranked by relevance, confidence scores broken down by dimension, and a toggle to compare hybrid vs. dense-only retrieval side by side. Users see the polished interface; engineers see the traces behind it.

#### Phase 6: Build Evaluation Framework & Feedback Loop (Day 18–21)

1. **Create a golden Q&A dataset:** Write 50+ question-answer pairs by hand, each tied to specific sections of your document corpus. Include straightforward lookups, multi-hop questions (answer requires combining information from two documents), questions with no answer in the corpus, ambiguous questions, and edge cases. This is your ground truth for measuring progress.

2. **Implement automated eval metrics:** For each test case, measure: answer correctness (LLM-as-judge against golden answer), faithfulness (are all claims grounded in retrieved context?), retrieval relevance (were the right chunks retrieved?), and citation accuracy (do citations actually support claims?). Run the full suite on every pipeline change. Track these metrics over time to spot regressions.

3. **Build retrieval and chunking strategy comparison:** Run the same eval suite across your chunking strategies (fixed-size, structure-aware, semantic) and retrieval configurations (dense-only, sparse-only, hybrid with different RRF weights). Generate a comparison report showing which strategy wins on which metrics. This data drives your architecture decisions and gives you concrete numbers for interviews.

4. **Auto-generate eval cases from production flags:** Every time a human flags a bad output via the flagging interface and the root cause analysis confirms the diagnosis, automatically create a new test case: the original question, the correct answer (human-provided correction), the failure category (retrieval/ranking/generation/citation), and the step where it failed. Append it to your growing eval dataset.

5. **Build regression tracking:** Periodically re-run the accumulated eval dataset against the current pipeline. Track whether known failure cases are still failing or have been fixed. Show a trend line of "known issues resolved" over time. This demonstrates progress and catches when a code change breaks something that was previously fixed.

6. **Create failure analytics dashboard:** Report showing: most common failure types in production, which pipeline step fails most often (retrieval vs. ranking vs. generation), failure rate over time, average time to root cause, and citations that most often fail verification. This is the data that tells you where to invest engineering effort.

#### Phase 7: Expose as API, Implement Production Monitoring & Polish for Portfolio (Day 21–22)

1. **Build the FastAPI service:** `POST /v1/ask` accepts a question and returns the answer with citations, confidence scores, and source metadata. `POST /v1/flag` allows users to flag responses as bad and triggers root cause analysis. `GET /v1/documents` lists indexed documents. `POST /v1/ingest` accepts new documents for indexing. Include OpenAPI documentation so reviewers can explore the API interactively.

2. **Containerize everything:** Docker-compose with the API service, ChromaDB, the frontend, and the evaluation harness. Include a seed script that indexes a sample documentation corpus so reviewers can spin everything up and test immediately. One command should bring the entire system online.

3. **Record a demo walkthrough:** Show: ingesting a set of documents, asking questions of varying difficulty (lookup, multi-hop, unanswerable), seeing the system correctly cite sources, deliberately triggering a failure (ask about something the docs don't cover or use a chunking strategy that breaks for this domain), opening the trace explorer, diagnosing the root cause via backward analysis, flagging the trace, and watching the eval dataset grow. Keep it under 4 minutes.

4. **Write the architecture doc:** Include a diagram of the RAG pipeline with the tracing layer, the backward analysis flow, the feedback loop, and the evaluation cycle. Frame it as: "I built a production-grade RAG system where observability is not bolted on but built-in from day one. When answers degrade, we don't guess which component failed—we trace it in seconds and feed the diagnosis into an automated eval suite that prevents regressions."

5. **Articulate the interview narrative:** Lead with the problem: "Most RAG systems work in isolation—retrieval, ranking, generation—with no visibility into where failures come from. When they ship to production and start producing bad answers, teams have no systematic way to debug." Then show your solution: "I built integrated tracing that instruments every step, automated root cause analysis that identifies exactly where the pipeline broke, and a feedback loop that turns every production failure into a regression test." Mention the numbers: "On a 50-question eval suite, my system achieves X% faithfulness and Y% citation accuracy. When I flag a bad output, root cause diagnosis runs in Z seconds."
