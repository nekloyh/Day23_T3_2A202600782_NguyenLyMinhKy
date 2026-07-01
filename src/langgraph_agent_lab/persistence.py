"""Checkpointer adapter."""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:
    """Return a LangGraph checkpointer for memory, SQLite, or Postgres."""
    if kind == "none":
        return None
    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if kind == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        db_path = database_url or "outputs/checkpoints.sqlite"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        saver = SqliteSaver(conn)
        saver.setup()
        return saver
    if kind == "postgres":
        try:
            module = importlib.import_module("langgraph.checkpoint.postgres")
        except ImportError as exc:
            raise RuntimeError(
                "Postgres checkpointer requires langgraph-checkpoint-postgres"
            ) from exc
        return module.PostgresSaver.from_conn_string(database_url or "")
    raise ValueError(f"Unknown checkpointer kind: {kind}")
