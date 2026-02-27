"""
utils.py – Shared utilities for the Text-to-SQL GRPO project.
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
    """Extract the first SQL block from *text*, or ``None`` if not found."""
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
) -> str:
    """Return a formatted prompt string for the model.

    Parameters
    ----------
    question:
        Natural language question.
    schema:
        Either a dict mapping table names to column lists, or a serialised
        string. ``None`` omits the schema section.
    system_prompt:
        System instruction prepended to the prompt.

    Returns
    -------
    A single string in ChatML / instruction format.
    """
    schema_section = ""
    if schema:
        if isinstance(schema, dict):
            lines = []
            for table, cols in schema.items():
                lines.append(f"Table {table}: ({', '.join(cols)})")
            schema_section = "\n".join(lines)
        else:
            schema_section = str(schema)

    parts = [f"### System\n{system_prompt}"]
    if schema_section:
        parts.append(f"### Schema\n{schema_section}")
    parts.append(f"### Question\n{question}")
    parts.append("### SQL\n")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def load_schema_lookup(data_dir: str) -> dict[str, dict[str, list[str]]]:
    """Load a schema lookup dict from *data_dir/schema_lookup.json* if present."""
    path = Path(data_dir) / "schema_lookup.json"
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return {}


def serialize_schema(schema: dict[str, list[str]]) -> str:
    """Serialise a schema dict to a compact string representation."""
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
    """Parse and re-serialise *sql* for consistent comparison."""
    try:
        return sqlglot.transpile(sql, read=dialect, write=dialect, pretty=False)[0]
    except sqlglot.errors.ParseError:
        return sql.strip()
