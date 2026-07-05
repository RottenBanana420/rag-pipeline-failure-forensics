"""Analysis module — backward root-cause span identification (Phase 4).

`root_cause.py` walks a failed `Trace`'s spans in reverse execution order,
using an LLM-as-judge to score each span's input→output transformation
quality (1-5), to find the earliest span in the trace's unhealthy tail.

Failure-type categorization (Retrieval Failure, Ranking Failure, Extraction
Hallucination, Citation Error, Generation Incomplete, Context Loss), the
narrative evidence-chain builder, and orchestrator wiring (loading a flagged
trace and calling this automatically) are separate, later tasks.
"""
