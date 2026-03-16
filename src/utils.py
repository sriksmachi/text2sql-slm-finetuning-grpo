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
import logging
import re
import sys
from pathlib import Path
from typing import Any

import sqlglot
from loguru import logger

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


class _InterceptHandler(logging.Handler):
    """Route stdlib logging records through loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)
        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru and intercept stdlib logging at *level*.

    Removes loguru's default handler and adds a fresh stderr sink at the
    requested level.  Also installs an :class:`_InterceptHandler` so that
    Transformers / TRL / vLLM log records (which use the stdlib ``logging``
    module) are routed through loguru at the same verbosity.

    Parameters
    ----------
    level:
        Any loguru-recognised level string: ``DEBUG``, ``INFO``, ``WARNING``,
        ``ERROR``, ``CRITICAL``.  Case-insensitive.
    """
    level = level.upper()
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=False,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}")
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    # Always suppress chatty third-party loggers regardless of level
    for noisy in (
        "urllib3",
        "urllib3.connectionpool",
        "filelock",
        "fsspec",
        "huggingface_hub",
        "azureml.mlflow",
        "azureml.mlflow._common._authentication",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)


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
# GPU / runtime resolution helpers (shared between trainer and evaluator)
# ---------------------------------------------------------------------------


def get_gpu_runtime_profile() -> dict[str, Any]:
    """Return the current CUDA device profile used for runtime decisions."""
    import torch

    profile: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "compute_capability": None,
        "major": None,
        "minor": None,
    }
    if not profile["cuda_available"]:
        return profile

    major, minor = torch.cuda.get_device_capability()
    profile.update(
        {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{major}.{minor}",
            "major": major,
            "minor": minor,
        }
    )
    return profile


def resolve_model_dtype(
    dtype_name: str | None,
    gpu_profile: dict[str, Any],
) -> str | None:
    """Resolve the configured model dtype for the active GPU.

    Falls back from bfloat16 to float16 on pre-Ampere GPUs (compute capability < 8).
    """
    if dtype_name is None:
        return None

    normalized = str(dtype_name).strip().lower()
    if normalized in {"", "auto", "none"}:
        return None

    if normalized == "bfloat16" and gpu_profile.get("major") is not None:
        if int(gpu_profile["major"]) < 8:
            logger.warning(
                "Configured dtype bfloat16 is not supported on GPU compute capability "
                "{}; falling back to float16.",
                gpu_profile["compute_capability"],
            )
            return "float16"

    return normalized


def resolve_fast_inference(
    requested: Any,
    gpu_profile: dict[str, Any],
) -> bool:
    """Decide whether to enable Unsloth fast inference on the current GPU.

    Parameters
    ----------
    requested:
        Value from ``model.fast_inference`` in grpo_config.yaml.
        Accepts ``True``/``False`` (bool), or the strings
        ``"true"``, ``"false"``, or ``"auto"``.
        ``"auto"`` enables fast inference only on Ampere+ (compute capability >= 8).
    gpu_profile:
        Dict returned by :func:`get_gpu_runtime_profile`.
    """
    major = gpu_profile.get("major")

    if isinstance(requested, str):
        normalized = requested.strip().lower()
        if normalized == "auto":
            enabled = bool(major is not None and int(major) >= 8)
            logger.info(
                "Resolved fast_inference={} for GPU compute capability {}.",
                enabled,
                gpu_profile.get("compute_capability"),
            )
            return enabled
        if normalized in {"true", "1", "yes", "on"}:
            requested_bool = True
        elif normalized in {"false", "0", "no", "off"}:
            requested_bool = False
        else:
            raise ValueError(
                "model.fast_inference must be one of true/false/auto. "
                f"Received: {requested!r}."
            )
    else:
        requested_bool = bool(requested)

    if requested_bool and major is not None and int(major) < 8:
        logger.warning(
            "Disabling fast_inference: GPU compute capability {} does not support "
            "the vLLM LoRA Triton path used by Unsloth on this runtime.",
            gpu_profile.get("compute_capability"),
        )
        return False

    return requested_bool


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


def configure_mlflow_tracking(
    experiment_name: str,
) -> tuple[bool, str | None]:
    """Configure MLflow tracking and return whether logging is enabled.

    When running inside an Azure ML job the tracking URI is automatically
    injected by the platform, so no explicit URI is required.
    """
    import mlflow

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        return False, f"Failed to configure MLflow tracking: {exc}"

    return True, None
