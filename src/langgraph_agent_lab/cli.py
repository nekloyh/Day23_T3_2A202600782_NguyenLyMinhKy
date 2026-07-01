"""CLI for the lab."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    first_thread_id = ""
    run_id = str(cfg.get("run_id") or uuid.uuid4().hex[:8])
    for scenario in scenarios:
        state = initial_state(scenario)
        state["thread_id"] = f"{state['thread_id']}-{run_id}"
        first_thread_id = first_thread_id or state["thread_id"]
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        started = time.perf_counter()
        final_state = graph.invoke(state, config=run_config)
        latency_ms = max(1, round((time.perf_counter() - started) * 1000))
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
                latency_ms=latency_ms,
            )
        )
    report = summarize_metrics(metrics, resume_success=_has_state_history(graph, first_thread_id))
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


def _has_state_history(graph: object, thread_id: str) -> bool:
    """Return whether checkpoint-backed state history is available for a scenario."""
    get_state_history = getattr(graph, "get_state_history", None)
    if get_state_history is None or not thread_id:
        return False
    config = {"configurable": {"thread_id": thread_id}}
    try:
        return bool(list(get_state_history(config)))
    except Exception:
        return False


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


@app.command("export-graph")
def export_graph(
    output: Annotated[Path, typer.Option("--output")] = Path("reports/graph.mmd"),
) -> None:
    """Export the compiled graph structure as Mermaid text."""
    graph = build_graph()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    typer.echo(f"Wrote graph diagram to {output}")


if __name__ == "__main__":
    app()
