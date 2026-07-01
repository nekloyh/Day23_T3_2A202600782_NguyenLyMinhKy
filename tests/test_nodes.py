from langgraph_agent_lab.nodes import (
    ask_clarification_node,
    dead_letter_node,
    evaluate_node,
    retry_or_fallback_node,
    tool_node,
)


def test_evaluate_node_requests_retry_for_error_tool_result() -> None:
    result = evaluate_node({"tool_results": ["ERROR: timeout"]})
    assert result["evaluation_result"] == "needs_retry"


def test_retry_node_increments_attempt_and_records_error() -> None:
    result = retry_or_fallback_node({"attempt": 1, "max_attempts": 3, "route": "error"})
    assert result["attempt"] == 2
    assert result["errors"]


def test_dead_letter_sets_terminal_answer() -> None:
    result = dead_letter_node({"attempt": 3})
    assert result["final_answer"]
    assert result["events"][0]["node"] == "dead_letter"


def test_clarification_node_sets_pending_question() -> None:
    result = ask_clarification_node({"query": "Can you fix it?"})
    assert result["pending_question"]
    assert result["final_answer"] == result["pending_question"]


def test_tool_node_simulates_retryable_error_route() -> None:
    result = tool_node({"route": "error", "attempt": 0, "query": "timeout"})
    assert "ERROR" in result["tool_results"][0]
