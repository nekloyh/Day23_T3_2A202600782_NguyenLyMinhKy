"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class ClassificationResult(BaseModel):
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="Support-ticket route."
    )
    risk_level: Literal["low", "medium", "high"] = Field(description="Risk level for the route.")
    rationale: str = Field(description="One short reason for the selected route.")


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    llm = get_llm(temperature=0).with_structured_output(ClassificationResult)
    result = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Classify a support-ticket query into exactly one route. "
                    "Use these definitions and priority order: "
                    "risky = side-effect actions such as refunds, deletions, cancellations, "
                    "sending emails, account changes; "
                    "tool = information lookup such as order status, tracking, search, "
                    "account lookup; "
                    "missing_info = vague or incomplete requests lacking the target/action "
                    "details; "
                    "error = system failures such as timeout, crash, service unavailable, "
                    "unrecoverable failure; "
                    "simple = general support questions answerable without tools. "
                    "Priority: risky > tool > missing_info > error > simple."
                )
            ),
            HumanMessage(content=f"Query: {query}"),
        ]
    )
    route = result.route
    risk_level = "high" if route == "risky" else result.risk_level
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                "query classified",
                route=route,
                risk_level=risk_level,
                rationale=result.rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")
    approval = state.get("approval") or {}

    if route == "error" and attempt < 2:
        result = f"ERROR: transient backend timeout while handling '{query}' on attempt {attempt}"
        event_type = "failed"
    elif route == "tool":
        result = f"Order lookup result: order/status data found for query '{query}'."
        event_type = "completed"
    elif route == "risky":
        if not approval.get("approved"):
            result = "ERROR: risky action blocked because approval was not granted."
            event_type = "blocked"
        else:
            result = (
                "Approved risky action staged successfully. "
                f"Reviewer={approval.get('reviewer', 'unknown')}; "
                f"action='{state.get('proposed_action')}'."
            )
            event_type = "completed"
    else:
        result = f"Support tool completed for query '{query}'."
        event_type = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, "tool execution finished", result=result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    latest = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [
            make_event(
                "evaluate",
                "completed",
                "tool result evaluated",
                evaluation_result=evaluation_result,
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    context = {
        "route": state.get("route", ""),
        "tool_results": state.get("tool_results", []),
        "approval": state.get("approval"),
        "errors": state.get("errors", []),
    }
    llm = get_llm(temperature=0.2)
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a support agent. Produce a concise final answer grounded only in "
                    "the supplied context. Do not invent order details, confirmations, "
                    "or user data. "
                    "If an action was only staged or approved, say that clearly."
                )
            ),
            HumanMessage(content=f"User query: {query}\nContext: {context}"),
        ]
    )
    final_answer = str(response.content).strip()
    return {
        "final_answer": final_answer,
        "messages": ["answer:completed"],
        "events": [make_event("answer", "completed", "final answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    pending_question = (
        "Could you provide the specific account, order, or issue details so I can help with "
        f"'{query}'?"
    )
    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "messages": ["clarify:pending"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    proposed_action = (
        f"Review and approve the requested side-effecting support action: {state.get('query', '')}"
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:prepared"],
        "events": [
            make_event(
                "risky_action",
                "completed",
                "risky action prepared for approval",
                proposed_action=proposed_action,
            )
        ],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {"approved": bool, "reviewer": str, "comment": str},
    "events": [make_event(...)]}
    """
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        payload = interrupt(
            {
                "proposed_action": state.get("proposed_action"),
                "query": state.get("query"),
                "instruction": "Approve or reject the proposed support action.",
            }
        )
        decision = ApprovalDecision.model_validate(payload)
    else:
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Mock approval granted for offline lab execution.",
        )
    return {
        "approval": decision.model_dump(),
        "messages": [f"approval:{decision.approved}"],
        "events": [
            make_event(
                "approval",
                "completed",
                "approval decision recorded",
                approved=decision.approved,
                reviewer=decision.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    error = f"Retry attempt {attempt}/{max_attempts} after route={state.get('route', 'unknown')}"
    return {
        "attempt": attempt,
        "errors": [error],
        "messages": [f"retry:{attempt}"],
        "events": [make_event("retry", "completed", "retry decision recorded", attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    final_answer = (
        "I could not complete this request after the allowed retry attempts. "
        "The ticket has been moved to the dead-letter path for manual support review."
    )
    return {
        "final_answer": final_answer,
        "route": state.get("route", "error"),
        "messages": ["dead_letter:completed"],
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {
        "messages": ["finalize:completed"],
        "events": [
            make_event(
                "finalize",
                "completed",
                "workflow finished",
                route=state.get("route"),
                attempt=state.get("attempt", 0),
            )
        ],
    }
