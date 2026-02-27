"""Unit tests for src/evaluator.py – cross_schema_exec_acc."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluator import cross_schema_exec_acc


class TestCrossSchemaExecAcc:
    def test_all_correct(self):
        predictions = ["SELECT 1", "SELECT 2"]
        references = ["SELECT 1", "SELECT 2"]
        metrics = cross_schema_exec_acc(predictions, references, dialects=["sqlite"])
        assert metrics["exec_acc_sqlite"] == 1.0

    def test_all_wrong(self):
        predictions = ["SELECT * FROM nonexistent_table_xyz", "SELECT * FROM also_missing"]
        references = ["SELECT 1", "SELECT 2"]
        metrics = cross_schema_exec_acc(predictions, references, dialects=["sqlite"])
        assert metrics["exec_acc_sqlite"] == 0.0

    def test_avg_across_dialects(self):
        predictions = ["SELECT 1"]
        references = ["SELECT 1"]
        metrics = cross_schema_exec_acc(predictions, references, dialects=["sqlite", "duckdb"])
        assert "exec_acc_avg" in metrics
        assert 0.0 <= metrics["exec_acc_avg"] <= 1.0

    def test_empty_inputs(self):
        metrics = cross_schema_exec_acc([], [], dialects=["sqlite"])
        assert metrics["exec_acc_sqlite"] == 0.0

    def test_returns_all_dialect_keys(self):
        predictions = ["SELECT 1"]
        references = ["SELECT 1"]
        dialects = ["sqlite", "postgres"]
        metrics = cross_schema_exec_acc(predictions, references, dialects=dialects)
        for d in dialects:
            assert f"exec_acc_{d}" in metrics
        assert "exec_acc_avg" in metrics
