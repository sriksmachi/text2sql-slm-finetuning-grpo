"""Unit tests for src/utils.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    build_prompt,
    extract_sql_from_text,
    normalise_sql,
    parse_schema_string,
    serialize_schema,
)


class TestExtractSqlFromText:
    def test_fenced_block(self):
        text = "```sql\nSELECT * FROM t;\n```"
        assert extract_sql_from_text(text) == "SELECT * FROM t;"

    def test_no_sql(self):
        assert extract_sql_from_text("hello world") is None


class TestBuildPrompt:
    def test_contains_question(self):
        prompt = build_prompt("How many users?")
        assert "How many users?" in prompt

    def test_contains_schema_dict(self):
        schema = {"users": ["id", "name"]}
        prompt = build_prompt("Count users", schema)
        assert "users" in prompt
        assert "id" in prompt

    def test_contains_schema_string(self):
        schema_str = "users(id, name)"
        prompt = build_prompt("Count users", schema_str)
        assert "users" in prompt

    def test_no_schema(self):
        prompt = build_prompt("Count users", None)
        assert "Schema" not in prompt


class TestSerializeSchema:
    def test_roundtrip(self):
        schema = {"employees": ["id", "name", "salary"], "departments": ["id", "name"]}
        serialised = serialize_schema(schema)
        parsed = parse_schema_string(serialised)
        assert parsed["employees"] == ["id", "name", "salary"]
        assert parsed["departments"] == ["id", "name"]

    def test_empty_schema(self):
        assert serialize_schema({}) == ""

    def test_parse_empty_string(self):
        assert parse_schema_string("") == {}


class TestNormaliseSql:
    def test_normalises_whitespace(self):
        sql1 = "SELECT  id  FROM  users"
        sql2 = "SELECT id FROM users"
        assert normalise_sql(sql1) == normalise_sql(sql2)

    def test_invalid_sql_returns_stripped(self):
        sql = "NOT VALID SQL ###"
        result = normalise_sql(sql)
        assert isinstance(result, str)
