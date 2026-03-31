"""
Unit tests for src/rewards.py.

Keeps the SQLite fixture entirely in-memory so the tests are hermetic and
fast without requiring the actual Spider/BIRD databases.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# ── helpers so tests run without GPU / vLLM installed ────────────────────────
import sys, types

# Stub heavy optional deps that are not available in a plain CI environment.
for _mod in ("loguru",):
    if _mod not in sys.modules:
        _stub = types.ModuleType(_mod)
        _stub.logger = type("Logger", (), {  # type: ignore[assignment]
            "debug": staticmethod(lambda *a, **kw: None),
            "info": staticmethod(lambda *a, **kw: None),
            "warning": staticmethod(lambda *a, **kw: None),
            "error": staticmethod(lambda *a, **kw: None),
        })()
        sys.modules[_mod] = _stub

from src.rewards import (  # noqa: E402 – after stub setup
    extract_sql,
    format_reward,
    exec_reward,
    combined_reward,
    schema_fidelity_reward,
    _exec_on_sqlite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(sql_text: str) -> list[dict[str, str]]:
    """Wrap text as a single-turn assistant completion."""
    return [{"role": "assistant", "content": sql_text}]


def _completions(*texts: str) -> list[list[dict[str, str]]]:
    return [_msg(t) for t in texts]


# ---------------------------------------------------------------------------
# extract_sql
# ---------------------------------------------------------------------------

class TestExtractSql:
    def test_fenced_sql(self):
        text = "```sql\nSELECT 1;\n```"
        assert extract_sql(text) == "SELECT 1;"

    def test_fenced_no_tag(self):
        text = "```\nSELECT id FROM users;\n```"
        assert extract_sql(text) is not None and "SELECT" in extract_sql(text)

    def test_inline_select(self):
        text = "The answer is SELECT id FROM t;"
        sql = extract_sql(text)
        assert sql is not None
        assert "SELECT" in sql.upper()

    def test_no_sql_returns_none(self):
        assert extract_sql("Hello, no SQL here.") is None

    def test_empty_fence_returns_none(self):
        assert extract_sql("```sql\n```") is None


# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------

class TestFormatReward:
    def test_valid_sql_gets_one(self):
        comps = _completions("```sql\nSELECT id FROM users;\n```")
        rewards = format_reward(comps)
        assert rewards == [1.0]

    def test_no_sql_gets_zero(self):
        comps = _completions("I don't know the answer.")
        rewards = format_reward(comps)
        assert rewards == [0.0]

    def test_multiple_completions(self):
        comps = _completions(
            "```sql\nSELECT 1;\n```",
            "no sql here",
        )
        rewards = format_reward(comps)
        assert rewards[0] == 1.0
        assert rewards[1] == 0.0


# ---------------------------------------------------------------------------
# _exec_on_sqlite  (read-only mode sanity check)
# ---------------------------------------------------------------------------

class TestExecOnSqlite:
    def _make_db(self, tmp_path: Path) -> Path:
        """Create a minimal SQLite DB with one table."""
        db = tmp_path / "test.sqlite"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'hello')")
        conn.commit()
        conn.close()
        return db

    def test_select_succeeds(self, tmp_path):
        db = self._make_db(tmp_path)
        # Patch path resolution so we can point at our temp DB
        ok, err = _exec_on_sqlite.__wrapped__(
            "SELECT * FROM t",
            db_path="test",
            source="spider",
            base_path=None,
        ) if hasattr(_exec_on_sqlite, "__wrapped__") else (None, None)
        # Direct call with fully resolved path via env var shortcut
        import os
        with patch.dict(os.environ, {"RAWDATA_DIR": str(tmp_path.parent)}):
            # Build a real spider-style directory tree
            db_dir = tmp_path / "spider" / "spider_data" / "database" / "test"
            db_dir.mkdir(parents=True)
            real_db = db_dir / "test.sqlite"
            real_db.write_bytes(db.read_bytes())
            ok, err = _exec_on_sqlite("SELECT * FROM t", db_path="test", source="spider", base_path=str(tmp_path))
        assert ok is True
        assert err is None

    def test_write_blocked_in_ro_mode(self, tmp_path):
        db = self._make_db(tmp_path)
        db_dir = tmp_path / "spider" / "spider_data" / "database" / "mydb"
        db_dir.mkdir(parents=True)
        (db_dir / "mydb.sqlite").write_bytes(db.read_bytes())
        ok, err = _exec_on_sqlite(
            "INSERT INTO t VALUES (99, 'x')",
            db_path="mydb",
            source="spider",
            base_path=str(tmp_path),
        )
        # Read-only mode must reject INSERT
        assert ok is False

    def test_bad_sql_returns_false(self, tmp_path):
        db = self._make_db(tmp_path)
        db_dir = tmp_path / "spider" / "spider_data" / "database" / "bad"
        db_dir.mkdir(parents=True)
        (db_dir / "bad.sqlite").write_bytes(db.read_bytes())
        ok, err = _exec_on_sqlite(
            "SELECT nonexistent_col FROM t",
            db_path="bad",
            source="spider",
            base_path=str(tmp_path),
        )
        assert ok is False
        assert err is not None


# ---------------------------------------------------------------------------
# exec_reward
# ---------------------------------------------------------------------------

class TestExecReward:
    def test_no_sql_penalty(self):
        comps = _completions("no sql here")
        rewards = exec_reward(comps, no_sql_penalty=-1.0)
        assert rewards == [-1.0]

    def test_passes_through_when_sql_present(self, tmp_path):
        """exec_reward returns 1.0 when SQL executes successfully."""
        import sqlite3 as _sq3
        db_dir = tmp_path / "spider" / "spider_data" / "database" / "demo"
        db_dir.mkdir(parents=True)
        _conn = _sq3.connect(str(db_dir / "demo.sqlite"))
        _conn.execute("CREATE TABLE a (x INTEGER)")
        _conn.execute("INSERT INTO a VALUES (1)")
        _conn.commit()
        _conn.close()

        import os
        with patch.dict(os.environ, {"RAWDATA_DIR": str(tmp_path)}):
            rewards = exec_reward(
                _completions("```sql\nSELECT * FROM a;\n```"),
                db_paths=["demo"],
                source=["spider"],
            )
        assert rewards == [1.0]


# ---------------------------------------------------------------------------
# schema_fidelity_reward
# ---------------------------------------------------------------------------

class TestSchemaFidelityReward:
    def test_exact_match(self):
        schema = {"users": ["id", "name"]}
        comps = _completions("```sql\nSELECT id, name FROM users;\n```")
        rewards = schema_fidelity_reward(comps, schemas=[schema])
        assert rewards[0] == 1.0

    def test_no_schema_returns_neutral(self):
        comps = _completions("```sql\nSELECT x FROM y;\n```")
        rewards = schema_fidelity_reward(comps, schemas=None)
        assert rewards[0] == 0.5

    def test_no_sql_returns_zero(self):
        comps = _completions("no sql here")
        rewards = schema_fidelity_reward(comps, schemas=[{"t": ["c"]}])
        assert rewards[0] == 0.0


# ---------------------------------------------------------------------------
# combined_reward
# ---------------------------------------------------------------------------

class TestCombinedReward:
    def test_output_length_matches_input(self):
        n = 4
        comps = _completions(*["no sql"] * n)
        rewards = combined_reward(comps)
        assert len(rewards) == n

    def test_all_floats(self):
        comps = _completions("```sql\nSELECT 1;\n```", "no sql")
        rewards = combined_reward(comps)
        assert all(isinstance(r, float) for r in rewards)

    def test_custom_weights_applied(self):
        """With zero exec/schema_fidelity weights, format weight dominates."""
        comps = _completions("```sql\nSELECT 1;\n```")
        weights = {"format": 1.0, "exec": 0.0, "schema_fidelity": 0.0, "sql_fence": 0.0}
        rewards = combined_reward(comps, weights=weights)
        assert rewards[0] == 1.0
