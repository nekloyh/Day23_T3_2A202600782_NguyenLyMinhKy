# Day 08 Lab Report

## 1. Team / student

- Name: Nguyễn Lý Minh Kỳ
- MSSV: 2A202600782
- Repo/commit: local working tree
- Date: 29/06/2026

## 2. Architecture

The workflow is a LangGraph `StateGraph` for support-ticket orchestration:
`START -> intake -> classify`, then conditional routing to simple answer, tool lookup,
missing-information clarification, risky-action approval, or retry/error handling. All terminal
paths pass through `finalize -> END`.

The graph demonstrates stateful orchestration that is difficult to express as a linear chain:
classification controls branches, `evaluate` gates retry loops, `approval` models human-in-the-loop
control, and `dead_letter` handles exhausted retries.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| messages | append | Keeps a compact audit trail of node activity. |
| tool_results | append | Preserves each tool attempt for evaluation and debugging. |
| errors | append | Records retry/failure history. |
| events | append | Structured grading/debug events. |
| route | overwrite | Current route selected by classification. |
| risk_level | overwrite | Current risk level for routing/audit. |
| attempt | overwrite | Bounded retry counter. |
| evaluation_result | overwrite | Retry-loop gate: `success` or `needs_retry`. |
| pending_question | overwrite | Clarification output for vague requests. |
| proposed_action | overwrite | Risky action awaiting approval. |
| approval | overwrite | Human or mock approval decision. |
| final_answer | overwrite | Final response returned to the user. |

## 4. Scenario results

| Metric | Value |
|---|---:|
| Total scenarios | 7 |
| Success rate | 100.00% |
| Average nodes visited | 6.43 |
| Total retries | 3 |
| Total interrupts/approvals | 2 |
| Resume success | True |

| Scenario | Expected route | Actual route | Success | Retries | Approvals | Latency ms |
|---|---|---|---:|---:|---:|---:|
| S01_simple | simple | simple | True | 0 | 0 | 3009 |
| S02_tool | tool | tool | True | 0 | 0 | 2917 |
| S03_missing | missing_info | missing_info | True | 0 | 0 | 1107 |
| S04_risky | risky | risky | True | 0 | 1 | 2942 |
| S05_error | error | error | True | 2 | 0 | 2147 |
| S06_delete | risky | risky | True | 0 | 1 | 2839 |
| S07_dead_letter | error | error | True | 1 | 0 | 919 |

## 5. Failure analysis

1. Retry or tool failure: transient tool errors are represented by tool results containing `ERROR`.
   `evaluate` turns those into `needs_retry`, `retry` increments `attempt`, and
   `route_after_retry` enforces `attempt < max_attempts` before trying the tool again.

2. Risky action without approval: refund/delete/email style requests route to `risky_action`, then
   `approval`. The tool only proceeds when `approval.approved` is true; otherwise the graph asks for
   clarification instead of executing a side-effecting action.

## 6. Persistence / recovery evidence

The graph is compiled with a checkpointer. The lab config uses SQLite and each scenario receives a
unique `thread_id` such as `thread-S01_simple-<run_id>`, so repeated runs do not reuse stale
checkpoint state. SQLite support is implemented in `build_checkpointer("sqlite", database_url)`
using `langgraph-checkpoint-sqlite` and WAL mode. The generated metrics set `resume_success` from
`graph.get_state_history(...)`, proving checkpoint history is available for replay/recovery.

## 7. Extension work

- SQLite checkpoint persistence extension implemented.
- Mock human-in-the-loop approval path implemented, with optional `LANGGRAPH_INTERRUPT=true`
  support.
- Graph diagram export implemented via `agent-lab export-graph --output reports/graph.mmd`.
- Structured event trail supports metrics, retry counts, approval observation, and route auditing.

## 8. Improvement plan

With one more day, I would productionize tool contracts first: replace string-based mock results
with typed tool result models, add LLM-as-judge evaluation for ambiguous tool outputs, and add
state-history replay tests against the SQLite checkpointer.
