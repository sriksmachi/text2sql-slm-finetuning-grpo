"""
rewards.py – Reward functions for GRPO Text-to-SQL training.

Rewards
-------
format_reward         : 1.0 if the completion contains valid SQL, else 0.0
exec_reward           : 1.0 if the SQL executes without error on the target DB,
                        0.0 on execution error, -1.0 when no SQL is found.
schema_fidelity_reward: fraction of referenced tables/columns that exist in
                        the provided schema context.
"""

from __future__ import annotations

import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import sqlglot
from loguru import logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SQL_FENCE_RE = re.compile(
    r"```(?:sql)?\s*(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
_INLINE_SQL_RE = re.compile(
    r"(SELECT\s+.+?;)",
    re.DOTALL | re.IGNORECASE,
)

SUPPORTED_DIALECTS = ("sqlite", "duckdb", "postgres", "mysql", "tsql", "bigquery")


def extract_sql(text: str) -> str | None:
    """Return the first SQL block found in *text*, or ``None``."""
    m = _SQL_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _INLINE_SQL_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------


def format_reward(
    completions: list[list[dict[str, str]]],
    **kwargs: Any,
) -> list[float]:
    """Return 1.0 for each completion that contains parseable SQL, else 0.0.

    Parameters
    ----------
    completions:
        List of message-list completions, each being a list of dicts with
        a ``"content"`` key (standard TRL reward signature).

    Returns
    -------
    List of floats, one per completion.
    """
    rewards: list[float] = []
    for messages in completions:
        text = messages[-1]["content"] if messages else ""
        sql = extract_sql(text)
        if sql is None:
            rewards.append(0.0)
            continue
        try:
            parsed = sqlglot.parse(sql, error_level=sqlglot.ErrorLevel.RAISE)
            rewards.append(1.0 if parsed else 0.0)
        except sqlglot.errors.ParseError:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# exec_reward
# ---------------------------------------------------------------------------


def _exec_on_sqlite(sql: str, db_path: str | None = None) -> bool:
    """Try to execute *sql* against an in-memory (or file) SQLite database."""
    conn = sqlite3.connect(db_path or ":memory:")
    try:
        conn.execute(sql)
        return True
    except Exception:  # noqa: BLE001
        return False
    finally:
        conn.close()


def exec_reward(
    completions: list[list[dict[str, str]]],
    prompts: list[list[dict[str, str]]] | None = None,
    dialect: str = "sqlite",
    db_paths: list[str | None] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Execute each SQL completion and return a reward.

    Scoring
    -------
    * No SQL found  → -1.0
    * Execution error → 0.0
    * Successful execution → 1.0

    Parameters
    ----------
    completions:
        TRL-style list of message-lists.
    prompts:
        Optional prompt message-lists (unused but part of TRL signature).
    dialect:
        Target SQL dialect for transpilation before execution.
    db_paths:
        Optional per-sample SQLite DB paths. ``None`` means in-memory.

    Returns
    -------
    List of floats.
    """
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(f"Unsupported dialect '{dialect}'. Choose from {SUPPORTED_DIALECTS}.")

    rewards: list[float] = []
    n = len(completions)
    paths = db_paths if db_paths is not None else [None] * n

    for i, messages in enumerate(completions):
        text = messages[-1]["content"] if messages else ""
        sql = extract_sql(text)
        if sql is None:
            rewards.append(-1.0)
            continue

        # Transpile to SQLite for execution when a different dialect is requested
        try:
            if dialect != "sqlite":
                sql = sqlglot.transpile(sql, read=dialect, write="sqlite")[0]
        except sqlglot.errors.ParseError:
            rewards.append(0.0)
            continue

        ok = _exec_on_sqlite(sql, paths[i])
        rewards.append(1.0 if ok else 0.0)

    return rewards


# ---------------------------------------------------------------------------
# schema_fidelity_reward
# ---------------------------------------------------------------------------


def _extract_schema_items(sql: str) -> tuple[set[str], set[str]]:
    """Return (tables, columns) referenced in *sql* (lowercased)."""
    tables: set[str] = set()
    columns: set[str] = set()
    try:
        for stmt in sqlglot.parse(sql):
            if stmt is None:
                continue
            for node in stmt.walk():
                if isinstance(node, sqlglot.exp.Table) and node.name:
                    tables.add(node.name.lower())
                if isinstance(node, sqlglot.exp.Column) and node.name:
                    columns.add(node.name.lower())
    except sqlglot.errors.ParseError:
        pass
    return tables, columns


def schema_fidelity_reward(
    completions: list[list[dict[str, str]]],
    schemas: list[dict[str, list[str]]] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Reward based on fraction of referenced tables/columns that exist in the schema.

    Parameters
    ----------
    completions:
        TRL-style list of message-lists.
    schemas:
        Per-sample schema dicts mapping table name (lowercase) → list of column
        names (lowercase). If ``None`` or empty, returns 0.5 (neutral) for all.

    Returns
    -------
    List of floats in [0, 1].
    """
    n = len(completions)
    schema_list = schemas if schemas is not None else [{}] * n

    rewards: list[float] = []
    for i, messages in enumerate(completions):
        text = messages[-1]["content"] if messages else ""
        sql = extract_sql(text)
        if sql is None:
            rewards.append(0.0)
            continue

        schema = schema_list[i] if i < len(schema_list) else {}
        if not schema:
            rewards.append(0.5)
            continue

        tables_in_schema = set(schema.keys())
        columns_in_schema = {col for cols in schema.values() for col in cols}

        ref_tables, ref_columns = _extract_schema_items(sql)

        all_refs: set[str] = ref_tables | ref_columns
        if not all_refs:
            rewards.append(0.5)
            continue

        valid_refs = (ref_tables & tables_in_schema) | (ref_columns & columns_in_schema)
        rewards.append(len(valid_refs) / len(all_refs))

    return rewards


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------


def combined_reward(
    completions: list[list[dict[str, str]]],
    prompts: list[list[dict[str, str]]] | None = None,
    schemas: list[dict[str, list[str]]] | None = None,
    dialect: str = "sqlite",
    db_paths: list[str | None] | None = None,
    weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Weighted combination of all three rewards.

    Default weights: format=0.2, exec=0.5, schema_fidelity=0.3.
    """
    w = weights or {"format": 0.2, "exec": 0.5, "schema_fidelity": 0.3}

    fmt = format_reward(completions)
    exc = exec_reward(completions, prompts=prompts, dialect=dialect, db_paths=db_paths)
    sfr = schema_fidelity_reward(completions, schemas=schemas)

    result = [
        w["format"] * f + w["exec"] * e + w["schema_fidelity"] * s
        for f, e, s in zip(fmt, exc, sfr)
    ]
    return result
