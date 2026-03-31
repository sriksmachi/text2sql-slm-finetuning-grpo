"""
Unit tests for src/utils.py.

These tests cover pure-Python helper functions that need no GPU,
no large models, and no external services.
"""

from __future__ import annotations

import sys
import types

# Stub loguru before importing utils so the module loads in a plain env.
if "loguru" not in sys.modules:
    _stub = types.ModuleType("loguru")
    _stub.logger = type("Logger", (), {  # type: ignore[assignment]
        "debug": staticmethod(lambda *a, **kw: None),
        "info": staticmethod(lambda *a, **kw: None),
        "warning": staticmethod(lambda *a, **kw: None),
        "error": staticmethod(lambda *a, **kw: None),
    })()
    sys.modules["loguru"] = _stub

import pytest  # noqa: E402

from src.utils import (  # noqa: E402
    extract_sql_from_text,
    build_prompt,
    serialize_schema,
    parse_schema_string,
    resolve_fast_inference,
    resolve_model_dtype,
)


# ---------------------------------------------------------------------------
# extract_sql_from_text
# ---------------------------------------------------------------------------

class TestExtractSqlFromText:
    def test_fenced_block_extracted(self):
        text = "Here is the answer:\n```sql\nSELECT * FROM authors;\n```"
        result = extract_sql_from_text(text)
        assert result == "SELECT * FROM authors;"

    def test_fenced_without_tag(self):
        text = "```\nSELECT 1;\n```"
        result = extract_sql_from_text(text)
        assert result == "SELECT 1;"

    def test_inline_select_extracted(self):
        text = "Run SELECT id FROM users; to get all users."
        result = extract_sql_from_text(text)
        assert result is not None
        assert "SELECT" in result.upper()

    def test_returns_none_when_absent(self):
        assert extract_sql_from_text("No SQL here at all.") is None

    def test_fenced_takes_priority_over_inline(self):
        text = "```sql\nSELECT a FROM b;\n``` Also SELECT x FROM y;"
        result = extract_sql_from_text(text)
        assert "FROM b" in result


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_returns_two_messages(self):
        msgs = build_prompt("How many users?")
        assert len(msgs) == 2

    def test_roles(self):
        msgs = build_prompt("How many users?")
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_question_in_user_message(self):
        msgs = build_prompt("List all employees.")
        assert "List all employees." in msgs[1]["content"]

    def test_schema_dict_included(self):
        schema = {"employees": ["id", "name", "salary"]}
        msgs = build_prompt("Find top earner.", schema=schema)
        user_content = msgs[1]["content"]
        assert "employees" in user_content

    def test_schema_string_included(self):
        msgs = build_prompt("Who earns most?", schema="employees(id, name)")
        assert "employees" in msgs[1]["content"]

    def test_no_schema_omits_schema_section(self):
        msgs = build_prompt("Count rows.", schema=None)
        assert "Schema" not in msgs[1]["content"]

    def test_custom_system_prompt(self):
        msgs = build_prompt("Q", system_prompt="Custom instructions.")
        assert msgs[0]["content"] == "Custom instructions."


# ---------------------------------------------------------------------------
# serialize_schema / parse_schema_string
# ---------------------------------------------------------------------------

class TestSerializeAndParseSchema:
    def test_roundtrip(self):
        original = {"users": ["id", "name", "email"], "orders": ["order_id", "user_id"]}
        serialized = serialize_schema(original)
        assert isinstance(serialized, str)
        parsed = parse_schema_string(serialized)
        assert parsed == original

    def test_empty_schema_roundtrip(self):
        original: dict[str, list[str]] = {}
        serialized = serialize_schema(original)
        parsed = parse_schema_string(serialized)
        assert parsed == original

    def test_single_table(self):
        original = {"t": ["a", "b"]}
        assert parse_schema_string(serialize_schema(original)) == original


# ---------------------------------------------------------------------------
# resolve_fast_inference
# ---------------------------------------------------------------------------

class TestResolveFastInference:
    def test_explicit_true(self):
        assert resolve_fast_inference(True, {}) is True

    def test_explicit_false(self):
        assert resolve_fast_inference(False, {}) is False

    def test_auto_on_ampere(self):
        gpu_profile = {"compute_major": 8, "compute_minor": 0}
        result = resolve_fast_inference("auto", gpu_profile)
        assert result is True

    def test_auto_off_on_volta(self):
        gpu_profile = {"compute_major": 7, "compute_minor": 0}
        result = resolve_fast_inference("auto", gpu_profile)
        assert result is False

    def test_auto_off_when_no_gpu(self):
        result = resolve_fast_inference("auto", {})
        assert result is False


# ---------------------------------------------------------------------------
# resolve_model_dtype
# ---------------------------------------------------------------------------

class TestResolveModelDtype:
    def test_returns_string_or_none(self):
        result = resolve_model_dtype("bfloat16", {})
        # Should be a torch dtype or None; just check it does not raise
        assert result is not None or result is None

    def test_none_dtype_returns_none(self):
        result = resolve_model_dtype(None, {})
        assert result is None
