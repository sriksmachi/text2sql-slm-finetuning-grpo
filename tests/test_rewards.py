"""Unit tests for src/rewards.py – format_reward, exec_reward, schema_fidelity_reward."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rewards import (
    combined_reward,
    exec_reward,
    extract_sql,
    format_reward,
    schema_fidelity_reward,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg(text: str) -> dict[str, str]:
    """Wrap *text* in a TRL-style assistant message dict."""
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# extract_sql
# ---------------------------------------------------------------------------


class TestExtractSql:
    def test_fenced_sql_block(self):
        text = "Here is the query:\n```sql\nSELECT 1;\n```"
        assert extract_sql(text) == "SELECT 1;"

    def test_fenced_no_language_tag(self):
        text = "```\nSELECT * FROM t;\n```"
        assert extract_sql(text) == "SELECT * FROM t;"

    def test_inline_select(self):
        text = "The answer is SELECT id FROM users;"
        result = extract_sql(text)
        assert result is not None
        assert "SELECT" in result.upper()

    def test_no_sql_returns_none(self):
        assert extract_sql("No SQL here at all.") is None

    def test_empty_string(self):
        assert extract_sql("") is None


# ---------------------------------------------------------------------------
# format_reward
# ---------------------------------------------------------------------------


class TestFormatReward:
    def test_valid_sql_gets_one(self):
        completions = [[_msg("```sql\nSELECT id FROM users;\n```")]]
        assert format_reward(completions) == [1.0]

    def test_invalid_sql_gets_zero(self):
        completions = [[_msg("```sql\nSELECT FROM WHERE;\n```")]]
        result = format_reward(completions)
        assert result == [0.0]

    def test_no_sql_gets_zero(self):
        completions = [[_msg("I don't know the answer.")]]
        assert format_reward(completions) == [0.0]

    def test_multiple_completions(self):
        completions = [
            [_msg("```sql\nSELECT 1;\n```")],
            [_msg("no sql here")],
            [_msg("```sql\nSELECT name FROM employees;\n```")],
        ]
        rewards = format_reward(completions)
        assert rewards == [1.0, 0.0, 1.0]

    def test_empty_messages_list(self):
        completions = [[]]
        assert format_reward(completions) == [0.0]


# ---------------------------------------------------------------------------
# exec_reward
# ---------------------------------------------------------------------------


class TestExecReward:
    def test_simple_select_executes(self):
        completions = [[_msg("```sql\nSELECT 1;\n```")]]
        result = exec_reward(completions, dialect="sqlite")
        assert result == [1.0]

    def test_no_sql_penalty(self):
        completions = [[_msg("no sql")]]
        result = exec_reward(completions)
        assert result == [-1.0]

    def test_invalid_table_returns_zero(self):
        # References a non-existent table in a fresh in-memory DB
        completions = [[_msg("```sql\nSELECT * FROM nonexistent_table_xyz;\n```")]]
        result = exec_reward(completions, dialect="sqlite")
        assert result == [0.0]

    def test_unsupported_dialect_raises(self):
        with pytest.raises(ValueError, match="Unsupported dialect"):
            exec_reward([[_msg("SELECT 1;")]], dialect="oraclesql")

    def test_multiple_dialects_sqlite(self):
        completions = [
            [_msg("```sql\nSELECT 42;\n```")],
            [_msg("no sql here")],
        ]
        rewards = exec_reward(completions, dialect="sqlite")
        assert rewards[0] == 1.0
        assert rewards[1] == -1.0


# ---------------------------------------------------------------------------
# schema_fidelity_reward
# ---------------------------------------------------------------------------


class TestSchemaFidelityReward:
    def _schema(self) -> dict[str, list[str]]:
        return {
            "employee": ["employee_id", "first_name", "last_name", "salary", "department_id"],
            "department": ["department_id", "department_name"],
        }

    def test_fully_valid_references(self):
        sql = "```sql\nSELECT first_name FROM employee WHERE department_id = 1;\n```"
        completions = [[_msg(sql)]]
        rewards = schema_fidelity_reward(completions, schemas=[self._schema()])
        assert rewards[0] > 0.5

    def test_no_sql_returns_zero(self):
        completions = [[_msg("no sql here")]]
        rewards = schema_fidelity_reward(completions, schemas=[self._schema()])
        assert rewards[0] == 0.0

    def test_no_schema_returns_neutral(self):
        completions = [[_msg("```sql\nSELECT 1;\n```")]]
        rewards = schema_fidelity_reward(completions, schemas=None)
        assert rewards[0] == 0.5

    def test_empty_schema_returns_neutral(self):
        completions = [[_msg("```sql\nSELECT 1;\n```")]]
        rewards = schema_fidelity_reward(completions, schemas=[{}])
        assert rewards[0] == 0.5

    def test_completely_unknown_table_lowers_reward(self):
        sql = "```sql\nSELECT * FROM totally_unknown_table;\n```"
        completions = [[_msg(sql)]]
        rewards = schema_fidelity_reward(completions, schemas=[self._schema()])
        assert rewards[0] < 0.5


# ---------------------------------------------------------------------------
# combined_reward
# ---------------------------------------------------------------------------


class TestCombinedReward:
    def test_returns_correct_length(self):
        completions = [
            [_msg("```sql\nSELECT 1;\n```")],
            [_msg("no sql")],
        ]
        rewards = combined_reward(completions)
        assert len(rewards) == 2

    def test_good_sql_higher_than_no_sql(self):
        good = [[_msg("```sql\nSELECT 1;\n```")]]
        bad = [[_msg("no sql")]]
        r_good = combined_reward(good)[0]
        r_bad = combined_reward(bad)[0]
        assert r_good > r_bad

    def test_custom_weights(self):
        completions = [[_msg("```sql\nSELECT 1;\n```")]]
        weights = {"format": 1.0, "exec": 0.0, "schema_fidelity": 0.0}
        rewards = combined_reward(completions, weights=weights)
        # format_reward returns 1.0, exec and schema both 0 weight → 1.0 * 1.0
        assert rewards[0] == pytest.approx(1.0, abs=0.01)
