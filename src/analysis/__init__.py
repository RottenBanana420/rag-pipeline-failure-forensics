"""Analysis module — backward root-cause span identification, failure
categorization, and evidence-chain narrative synthesis (Phase 4).

`root_cause.py` walks a failed `Trace`'s spans in reverse execution order,
using an LLM-as-judge to score each span's input→output transformation
quality (1-5), to find the earliest span in the trace's unhealthy tail.

`failure_categorizer.py` classifies that root-cause span into a failure
taxonomy (Retrieval Failure, Ranking Failure, Extraction Hallucination,
Citation Error, Generation Incomplete, Context Loss, or Other), also via an
LLM-as-judge.

`evidence_chain.py` synthesizes a causal narrative from that root-cause
diagnosis and failure category, plus the ordered input/output evidence
backing it, also via an LLM-as-judge.

Orchestrator wiring (loading a flagged trace and calling all three of the
above automatically) is a separate, later task.
"""
