"""
utils.py – Shared utilities for the Text-to-SQL GRPO project.

Key helpers
-----------
extract_sql_from_text   Extract the first SQL block from a model response.
build_prompt            Build a chat-format prompt (list of dicts) for a
                        question + schema pair, matching the notebook's
                        ``_SYSTEM_PROMPT`` convention.
make_prompt_record      Convert a dataset row into a training-ready dict with
                        keys: prompt, solution, schema, source, db_id.
serialize_schema        Compact string representation of a schema dict.
parse_schema_string     Parse a serialised schema string back to a dict.
load_schema_lookup      Load the schema_lookup.json produced by data prep.
normalise_sql           Parse → re-serialise SQL for consistent comparison.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import sqlglot

# ---------------------------------------------------------------------------
# SQL extraction
# ---------------------------------------------------------------------------

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_INLINE_SQL_RE = re.compile(r"(SELECT\s+.+?;)", re.DOTALL | re.IGNORECASE)


def extract_sql_from_text(text: str) -> str | None:
    """Extract the first SQL block from *text*, or ``None`` if not found.

    Checks for a fenced ```sql``` block first, then falls back to a bare
    ``SELECT … ;`` pattern.
    """
    m = _SQL_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _INLINE_SQL_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural language "
    "question, write a correct SQL query that answers the question.\n"
    "Return ONLY the SQL query inside a ```sql ... ``` code block."
)


def build_prompt(
    question: str,
    schema: dict[str, list[str]] | str | None = None,
    system_prompt: str = _SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Return a chat-format prompt as a list of message dicts.

    The output is compatible with ``tokenizer.apply_chat_template`` and with
    TRL's GRPOTrainer which expects each training example's ``"prompt"`` field
    to be a list of ``{"role": …, "content": …}`` dicts.

    Parameters
    ----------
    question:
        Natural language question to answer with SQL.
    schema:
        Either a dict mapping table names to column lists, or a pre-serialised
        string.  ``None`` omits the schema section from the user message.
    system_prompt:
        System instruction for the assistant turn.

    Returns
    -------
    List of two dicts: a system message and a user message.

    Example
    -------
    >>> build_prompt("How many authors are there?", {"author": ["aid", "name"]})
    [
        {"role": "system", "content": "You are an expert SQL assistant…"},
        {"role": "user",   "content": "### Question\nHow many authors…\n### Schema\n…"},
    ]
    """
    parts: list[str] = []
    parts.append(f"### Question\n{question}")

    if schema is not None:
        schema_str = schema if isinstance(schema, str) else str(schema)
        parts.append(f"### Schema\n{schema_str}")

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "\n".join(parts)},
    ]


def make_prompt_record(
    question: str,
    schema: dict[str, list[str]] | str,
    answer: str,
    source: str,
    db_id: str,
) -> dict[str, Any]:
    """Build a training-ready record from a single dataset row.

    This mirrors the notebook's ``_make_prompt`` function and produces the
    dict format expected by GRPOTrainer and the reward functions.

    Parameters
    ----------
    question:
        Natural language question.
    schema:
        Schema dict or pre-serialised string.
    answer:
        Gold SQL string (used as ``solution`` during evaluation).
    source:
        Dataset source – ``"spider"`` or ``"bird"``.
    db_id:
        Database identifier (e.g. ``"academic"``).

    Returns
    -------
    Dict with keys: ``prompt``, ``solution``, ``schema``, ``source``, ``db_id``.
    """
    prompt = build_prompt(question, schema)
    return {
        "prompt": prompt,
        "solution": answer,
        "schema": schema,
        "source": source,
        "db_id": db_id,
    }


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def load_schema_lookup(data_dir: str) -> dict[str, dict[str, list[str]]]:
    """Load a schema lookup dict from *data_dir/schema_lookup.json* if present.

    The JSON file is expected to be a list of ``{"db_id": …, "schema": …}``
    objects as produced by the data-preparation pipeline.

    Returns an empty dict if the file does not exist.
    """
    path = Path(data_dir) / "schema_lookup.json"
    if not path.exists():
        return {}
    with open(path) as fh:
        records = json.load(fh)
    # Support both list-of-objects and plain dict formats
    if isinstance(records, list):
        return {r["db_id"]: r["schema"] for r in records}
    return records  # type: ignore[return-value]


def serialize_schema(schema: dict[str, list[str]]) -> str:
    """Serialise a schema dict to a compact pipe-delimited string.

    Example
    -------
    >>> serialize_schema({"author": ["aid", "name"]})
    'author(aid, name)'
    """
    lines = [f"{table}({', '.join(cols)})" for table, cols in schema.items()]
    return " | ".join(lines)


def parse_schema_string(schema_str: str) -> dict[str, list[str]]:
    """Parse a schema string produced by :func:`serialize_schema` back to a dict."""
    schema: dict[str, list[str]] = {}
    for part in schema_str.split("|"):
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(\w+)\(([^)]*)\)", part)
        if m:
            table = m.group(1).strip()
            cols = [c.strip() for c in m.group(2).split(",") if c.strip()]
            schema[table] = cols
    return schema


# ---------------------------------------------------------------------------
# SQL normalisation
# ---------------------------------------------------------------------------


def normalise_sql(sql: str, dialect: str = "sqlite") -> str:
    """Parse and re-serialise *sql* for consistent comparison.

    Returns the original (stripped) string if sqlglot cannot parse it.
    """
    try:
        return sqlglot.transpile(sql, read=dialect, write=dialect, pretty=False)[0]
    except sqlglot.errors.ParseError:
        return sql.strip()
