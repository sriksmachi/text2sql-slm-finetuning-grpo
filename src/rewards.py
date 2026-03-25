"""
rewards.py – Reward functions for GRPO Text-to-SQL training.

Each reward function follows the TRL reward-function signature:
    f(completions, prompts=None, **kwargs) -> list[float]

Rewards
-------
format_reward
    1.0  – completion contains a ```sql``` fence with parseable SQL
    0.0  – no SQL fence or sqlglot parse error

exec_reward
    1.0  – SQL executes without error on the target SQLite database
    0.0  – SQL execution error
   -1.0  – no SQL found in the completion

schema_fidelity_reward
    1.0  – all referenced tables/columns exist in the provided schema
    0.5  – no schema provided (neutral; does not penalise)
    0.0  – no SQL found or every reference is invalid

combined_reward
    Weighted sum: 0.2 × format + 0.5 × exec + 0.3 × schema_fidelity
    (weights are configurable via the ``weights`` argument)

Database path resolution
------------------------
``exec_reward`` and ``combined_reward`` accept a ``source`` list
("spider" | "bird") and a ``base_path`` for locating the actual
SQLite files:

    spider  →  <base_path>/spider/spider_data/database/<db_id>/<db_id>.sqlite
    bird    →  <base_path>/bird/dev_databases/<db_id>/<db_id>.sqlite
"""

from __future__ import annotations

import os
import re
import sqlite3
from typing import Any

import sqlglot
from loguru import logger


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# Matches bare SELECT … without requiring a trailing semicolon.
# Uses a lazy quantifier that stops at an explicit ";", a blank line, or
# end-of-string so multiline SQL is captured whole but trailing explanation
# text after a paragraph break is excluded.
_INLINE_SQL_RE = re.compile(r"(SELECT\b.+?)(?:;|\n\n|\Z)", re.DOTALL | re.IGNORECASE)

SUPPORTED_DIALECTS = ("sqlite", "duckdb", "postgres", "mysql", "tsql", "bigquery")


def _preview_text(text: str | None, limit: int = 240) -> str:
    """Return a single-line preview for log messages."""
    if text is None:
        return ""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."


def extract_sql(text: str) -> str | None:
    """Return the first SQL block found in *text*, or ``None``.

    Looks first for a fenced ```sql``` block, then for a bare
    ``SELECT … ;`` statement.
    """
    m = _SQL_FENCE_RE.search(text)
    if m:
        sql = m.group(1).strip()
    else:
        m = _INLINE_SQL_RE.search(text)
        if m:
            sql = m.group(1).strip()
        else:
            logger.debug(f"extract_sql: No SQL block found in text_preview={_preview_text(text)!r}")
            return None
    # Guard: an empty capture means the fence regex matched but enclosed nothing
    # (e.g. double-fenced output like ```sql\n```sql ... ```\n```).  Treat as
    # "no SQL found" so execution/format rewards return the correct penalty.
    if not sql:
        logger.debug(f"extract_sql: Empty SQL captured from text_preview={_preview_text(text)!r}")
        return None
    # 1) Normalize backslash-escaped quotes (\') -> SQL-standard doubled quote ('')
    sql = sql.replace("\\'", "''")

    # 2) Normalize in-word apostrophes (O'Gallagher -> O''Gallagher, Women's -> Women''s)
    sql = re.sub(r"(?<=[A-Za-z])'(?=[A-Za-z])", "''", sql)

    # 3) Normalize spaced apostrophes in names (O' Gallagher -> O'' Gallagher),
    #    but do NOT touch valid quote closures before SQL keywords/operators.
    sql = re.sub(
    r"'([^'\n\r]*)''(?=\s*(?:AND|OR|NOT|IN|LIKE|IS|BETWEEN|FROM|WHERE|JOIN|ON|GROUP|ORDER|HAVING|LIMIT|UNION|$|;|,|\)))",
    r"'\1'",
    sql,
    flags=re.IGNORECASE,
)
    return sql


# ---------------------------------------------------------------------------
# sql_format_strict_reward
# ---------------------------------------------------------------------------


def sql_format_strict_reward(
    completions: list[list[dict[str, str]]],
    **kwargs: Any,
) -> list[float]:
    """Score each completion on whether it uses a fenced ```sql``` block.

    Unlike ``format_reward`` which accepts bare ``SELECT …`` statements,
    this reward only gives 1.0 when the model wraps its SQL in a proper
    ```sql … ``` fence.  This directly reinforces the expected output
    format and penalises prose or Python answers.

    Returns
    -------
    List of floats (one per completion), each in {0.0, 1.0}.
    """
    rewards: list[float] = []
    for messages in completions:
        text = messages[-1]["content"] if messages else ""
        rewards.append(1.0 if _SQL_FENCE_RE.search(text) else 0.0)
    return rewards


# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------


def format_reward(
    completions: list[list[dict[str, str]]],
    **kwargs: Any,
) -> list[float]:
    """Score each completion on SQL format validity.

    A completion scores 1.0 when it contains either a fenced ```sql```
    block or a bare ``SELECT …`` statement that sqlglot can parse without
    errors, and 0.0 otherwise.

    Parameters
    ----------
    completions:
        TRL-style list of message-lists.  The last message in each list
        is the model's assistant turn.

    Returns
    -------
    List of floats (one per completion), each in {0.0, 1.0}.
    """
    rewards: list[float] = []
    for idx, messages in enumerate(completions):
        text = messages[-1]["content"] if messages else ""
        sql = extract_sql(text)
        if sql is None:
            logger.debug(
                f"[format_reward] [{idx}] completion_preview={_preview_text(text)!r}"
            )
            logger.warning(f"[format_reward] [{idx}] No SQL block found → 0.0")
            rewards.append(0.0)
            continue
        try:
            parsed = sqlglot.parse(sql, error_level=sqlglot.ErrorLevel.RAISE)
            score = 1.0 if parsed else 0.0
        except sqlglot.errors.ParseError as exc:
            logger.debug(
                f"[format_reward] [{idx}] sql_preview={_preview_text(sql)!r}"
            )
            logger.warning(f"[format_reward] [{idx}] Parse error: {exc} → 0.0")
            score = 0.0
        except sqlglot.errors.TokenError as exc:
            logger.debug(
                f"[format_reward] [{idx}] sql_preview={_preview_text(sql)!r}"
            )
            logger.warning(f"[format_reward] [{idx}] Token error: {exc} → 0.0")
            score = 0.0
        rewards.append(score)
    return rewards


# ---------------------------------------------------------------------------
# exec_reward
# ---------------------------------------------------------------------------


def _exec_on_sqlite(
    sql: str,
    db_path: str | None = None,
    source: str | None = None,
    base_path: str | None = None,
) -> tuple[bool, str | None]:
    """Execute *sql* against the correct SQLite database and return the result.

    Path resolution uses ``source`` to locate the ``.sqlite`` file:

    * ``source="spider"`` → ``<base_path>/spider/spider_data/database/<db_path>/<db_path>.sqlite``
    * ``source="bird"``   → ``<base_path>/bird/dev_20240627/dev_databases/<db_path>/<db_path>.sqlite``

    Parameters
    ----------
    sql:
        SQL string to execute.
    db_path:
        Database identifier (``db_id``).  Used to build the full file path.
    source:
        Dataset source – ``"spider"`` or ``"bird"``.
    base_path:
        Root path of the ``data/`` directory (set via the ``RAWDATA_DIR`` env var).
        For the local repo this is ``<repo_root>/data``.
        If ``None``, falls back to the ``RAWDATA_DIR`` environment variable.

    Returns
    -------
    ``(True, None)`` on success; ``(False, error_message)`` on failure.
    """
    if base_path is None:
        base_path = os.environ.get("RAWDATA_DIR", "")

    if source == "spider" and db_path:
        full_path = f"{base_path}/spider/spider_data/database/{db_path}/{db_path}.sqlite"
    elif source == "bird" and db_path:
        full_path = f"{base_path}/bird/dev_databases/{db_path}/{db_path}.sqlite"
    else:
        logger.warning(
            f"[exec] Cannot resolve DB path for db_path={db_path!r}, source={source!r}"
        )
        return False, "DB path could not be resolved"

    logger.debug(f"[exec] Resolved full_path={full_path!r}")
    try:
        conn = sqlite3.connect(full_path)
        conn.execute(sql)
        conn.close()
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def exec_reward(
    completions: list[list[dict[str, str]]],
    prompts: list[list[dict[str, str]]] | None = None,
    dialect: str = "sqlite",
    db_paths: list[str | None] | None = None,
    source: list[str | None] | None = None,
    no_sql_penalty: float = -1.0,
    **kwargs: Any,
) -> list[float]:
    """Score each completion by executing its SQL against the target database.

    Scoring
    -------
    *  1.0 – SQL executes without error
    *  0.0 – Execution error (bad SQL, wrong column, etc.)
    * -1.0 – No SQL found in the completion

    Parameters
    ----------
    completions:
        TRL-style list of message-lists.
    prompts:
        Optional prompt message-lists (part of TRL signature; unused here).
    dialect:
        Source SQL dialect.  Non-SQLite dialects are transpiled to SQLite
        before execution.
    db_paths:
        Per-sample database identifiers (e.g. ``"academic"``).
    source:
        Per-sample dataset source – ``"spider"`` or ``"bird"``.

    Returns
    -------
    List of floats (one per completion).
    """
    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError(f"Unsupported dialect '{dialect}'. Choose from {SUPPORTED_DIALECTS}.")

    n = len(completions)
    paths = db_paths if db_paths is not None else [None] * n
    sources = source if source is not None else [None] * n

    rewards: list[float] = []
    for idx, messages in enumerate(completions):
        text = messages[-1]["content"] if messages else ""
        sql = extract_sql(text)
        if sql is None:
            logger.warning(f"[exec_reward] [{idx}] No SQL found → {no_sql_penalty}")
            rewards.append(no_sql_penalty)
            continue

        if dialect != "sqlite":
            try:
                sql = sqlglot.transpile(sql, read=dialect, write="sqlite")[0]
            except sqlglot.errors.ParseError as exc:
                logger.debug(
                    f"[exec_reward] [{idx}] sql_preview={_preview_text(sql)!r}"
                )
                logger.warning(f"[exec_reward] [{idx}] Transpile error: {exc} → 0.0")
                rewards.append(0.0)
                continue
            except sqlglot.errors.TokenError as exc:
                logger.debug(
                    f"[exec_reward] [{idx}] sql_preview={_preview_text(sql)!r}"
                )
                logger.warning(f"[exec_reward] [{idx}] Token error during transpile: {exc} → 0.0")
                rewards.append(0.0)
                continue

        ok, err = _exec_on_sqlite(sql, paths[idx], sources[idx])
        score = 1.0 if ok else 0.0
        if not ok:
            logger.warning(f"[exec_reward] [{idx}] Execution failed: {err} → {score}")
        rewards.append(score)

    return rewards


# ---------------------------------------------------------------------------
# schema_fidelity_reward
# ---------------------------------------------------------------------------


def _extract_schema_items(sql: str) -> tuple[set[str], set[str]]:
    """Return ``(tables, columns)`` referenced in *sql* (all lowercased)."""
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
    except sqlglot.errors.ParseError as exc:
        logger.debug(f"[schema] sql_preview={_preview_text(sql)!r}")
        logger.warning(f"[schema] Parse error while extracting items: {exc}")
    except sqlglot.errors.TokenError as exc:
        logger.debug(f"[schema] sql_preview={_preview_text(sql)!r}")
        logger.warning(f"[schema] Token error while extracting items: {exc}")
    return tables, columns


def schema_fidelity_reward(
    completions: list[list[dict[str, str]]],
    schemas: list[dict[str, list[str]]] | None = None,
    unknown_schema_item_penalty: float = 0.0,
    **kwargs: Any,
) -> list[float]:
    """Score each completion on how faithfully it references the provided schema.

    The score is the fraction of referenced tables/columns that actually
    appear in the provided schema::

        score = |valid_refs| / |all_refs|

    Edge cases
    ----------
    * No SQL found             → 0.0
    * No schema provided       → 0.5 (neutral; does not penalise)
    * SQL references nothing   → 0.5 (neutral)

    Parameters
    ----------
    completions:
        TRL-style list of message-lists.
    schemas:
        Per-sample schema dicts: ``{table_name: [col1, col2, …]}``.
        Keys and values should be lower-cased to match SQL extraction.

    Returns
    -------
    List of floats (one per completion) in [0.0, 1.0].
    """
    n = len(completions)
    schema_list = schemas if schemas is not None else [{}] * n
    rewards: list[float] = []

    for idx, messages in enumerate(completions):
        text = messages[-1]["content"] if messages else ""
        sql = extract_sql(text)
        schema = schema_list[idx] if idx < len(schema_list) else {}

        if sql is None:
            rewards.append(0.0)
            continue

        if not schema:
            rewards.append(0.5)
            continue

        tables_in_schema  = {k.lower() for k in schema.keys()}
        columns_in_schema = {col.lower() for cols in schema.values() for col in cols}
        ref_tables, ref_columns = _extract_schema_items(sql)  # already lowercased

        all_refs = ref_tables | ref_columns
        if not all_refs:
            rewards.append(0.5)
            continue

        valid_refs = (ref_tables & tables_in_schema) | (ref_columns & columns_in_schema)
        valid_frac = len(valid_refs) / len(all_refs)
        # Apply penalty proportional to the fraction of invalid references.
        # With penalty=0.0 (default) this is identical to the original formula.
        score = valid_frac + unknown_schema_item_penalty * (1.0 - valid_frac)
        rewards.append(score)

    return rewards


# ---------------------------------------------------------------------------
# combined_reward
# ---------------------------------------------------------------------------


def combined_reward(
    completions: list[list[dict[str, str]]],
    prompts: list[list[dict[str, str]]] | None = None,
    schemas: list[dict[str, list[str]]] | None = None,
    dialect: str = "sqlite",
    db_paths: list[str | None] | None = None,
    source: list[str | None] | None = None,
    weights: dict[str, float] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Weighted combination of format, exec, and schema_fidelity rewards.

    Default weights (must sum to 1.0):
        format=0.2 · exec=0.5 · schema_fidelity=0.3

    Supports an optional ``no_sql_penalty`` key in *weights* (default -1.0)
    forwarded to ``exec_reward`` to control the penalty for completions that
    contain no SQL at all.
    """
    """Weighted combination of format, exec, and schema_fidelity rewards.

    Default weights (must sum to 1.0):
        format=0.2 · exec=0.5 · schema_fidelity=0.3

    Parameters
    ----------
    completions:
        TRL-style list of message-lists.
    prompts:
        Optional prompt message-lists from TRL (passed through to exec_reward).
    schemas:
        Per-sample schema dicts for schema_fidelity scoring.
    dialect:
        SQL dialect for transpilation in exec_reward.
    db_paths:
        Per-sample database identifiers (e.g. ``"academic"``).
    source:
        Per-sample dataset source – ``"spider"`` or ``"bird"``.
    weights:
        Override the default component weights.

    Returns
    -------
    List of floats (one per completion), rounded to 4 decimal places.
    """
    w = weights or {"format": 0.15, "exec": 0.5, "schema_fidelity": 0.25, "sql_fence": 0.1}
    no_sql_penalty: float = w.get("no_sql_penalty", -2.0)  # type: ignore[assignment]
    unknown_schema_item_penalty: float = w.get("unknown_schema_item_penalty", 0.0)  # type: ignore[assignment]
    fmt = format_reward(completions)
    exc = exec_reward(
        completions,
        prompts=prompts,
        dialect=dialect,
        db_paths=db_paths,
        source=source,
        no_sql_penalty=no_sql_penalty,
    )
    sfr = schema_fidelity_reward(
        completions,
        schemas=schemas,
        unknown_schema_item_penalty=unknown_schema_item_penalty,
    )
    fence = sql_format_strict_reward(completions)

    logger.debug(f"[combined_reward]============================================")
    logger.debug(f"[combined_reward] completions={len(completions)} items")
    logger.debug(f"[combined_reward] completions_preview={[ _preview_text(messages[-1]['content']) for messages in completions ]}")
    logger.debug(f"[combined_reward] fmt={fmt}")
    logger.debug(f"[combined_reward] exc={exc}")
    logger.debug(f"[combined_reward] sfr={sfr}")
    logger.debug(f"[combined_reward] fence={fence}")
    return [
        round(
            w["format"] * f
            + w["exec"] * e
            + w["schema_fidelity"] * s
            + w.get("sql_fence", 0.0) * fn,
            4,
        )
        for f, e, s, fn in zip(fmt, exc, sfr, fence)
    ]
