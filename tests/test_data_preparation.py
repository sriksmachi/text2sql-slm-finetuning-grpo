"""
Unit tests for src/data_preparation.py.

Focuses on pure-logic helpers that do not require downloading datasets or
accessing real SQLite files.
"""

from __future__ import annotations

import sys
import types

# Stub heavy optional deps so the module loads in a plain CI environment.
for _name, _attrs in [
    ("loguru", {"logger": type("L", (), {
        "debug": staticmethod(lambda *a, **kw: None),
        "info": staticmethod(lambda *a, **kw: None),
        "warning": staticmethod(lambda *a, **kw: None),
        "error": staticmethod(lambda *a, **kw: None),
        "success": staticmethod(lambda *a, **kw: None),
    })()}),
    ("tqdm", {"tqdm": lambda *a, **kw: iter(a[0] if a else [])}),
]:
    if _name not in sys.modules:
        m = types.ModuleType(_name)
        for k, v in _attrs.items():
            setattr(m, k, v)
        sys.modules[_name] = m

import numpy as np
import pandas as pd
import pytest

from src.data_preparation import stratified_split, _ensure_question


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_merged(n_spider: int = 10, n_bird: int = 10, rows_per_db: int = 5) -> pd.DataFrame:
    """Build a synthetic merged_samples DataFrame."""
    rows = []
    for i in range(n_spider):
        for _ in range(rows_per_db):
            rows.append({"db_id": f"spider_db_{i}", "source": "spider",
                         "question": "q", "query": "SELECT 1"})
    for i in range(n_bird):
        for _ in range(rows_per_db):
            rows.append({"db_id": f"bird_db_{i}", "source": "bird",
                         "question": "q", "query": "SELECT 1"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# stratified_split
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def test_returns_three_parts(self):
        df = _make_merged()
        train, val, test = stratified_split(df, sample_size=10, train_ratio=0.7, val_ratio=0.15)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(val, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_no_overlap_between_splits(self):
        df = _make_merged(n_spider=8, n_bird=8)
        train, val, test = stratified_split(df, sample_size=12, train_ratio=0.6, val_ratio=0.2)
        train_ids = set(train["db_id"])
        val_ids = set(val["db_id"])
        test_ids = set(test["db_id"])
        assert train_ids.isdisjoint(val_ids), "Train and val share db_ids"
        assert train_ids.isdisjoint(test_ids), "Train and test share db_ids"
        assert val_ids.isdisjoint(test_ids), "Val and test share db_ids"

    def test_both_sources_represented_in_train(self):
        df = _make_merged(n_spider=8, n_bird=8)
        train, val, test = stratified_split(df, sample_size=10, train_ratio=0.7, val_ratio=0.15)
        assert "spider" in train["source"].values
        assert "bird" in train["source"].values

    def test_all_rows_accounted_for_with_full_sample(self):
        df = _make_merged(n_spider=5, n_bird=5)
        train, val, test = stratified_split(df, sample_size=-1)
        total = len(train) + len(val) + len(test)
        assert total == len(df)

    def test_reproducible_with_seed(self):
        df = _make_merged(n_spider=10, n_bird=10)
        t1, v1, te1 = stratified_split(df, sample_size=10, seed=42)
        t2, v2, te2 = stratified_split(df, sample_size=10, seed=42)
        assert set(t1["db_id"]) == set(t2["db_id"])
        assert set(v1["db_id"]) == set(v2["db_id"])

    def test_large_sample_size_clipped_to_available(self):
        df = _make_merged(n_spider=3, n_bird=3)
        # Requesting more db_ids than exist should not crash
        train, val, test = stratified_split(df, sample_size=999)
        assert len(train) > 0

    def test_train_larger_than_val_and_test(self):
        df = _make_merged(n_spider=10, n_bird=10)
        train, val, test = stratified_split(df, sample_size=20, train_ratio=0.7, val_ratio=0.15)
        assert len(train) >= len(val)
        assert len(train) >= len(test)


# ---------------------------------------------------------------------------
# _ensure_question
# ---------------------------------------------------------------------------

class TestEnsureQuestion:
    def test_question_mark_added_when_missing(self):
        result = _ensure_question("How many rows are there")
        assert result.endswith("?")

    def test_existing_question_mark_not_doubled(self):
        result = _ensure_question("How many rows?")
        assert result == "How many rows?"
        assert result.count("?") == 1

    def test_empty_string_unchanged_or_minimal(self):
        # Should not raise; exact behaviour for empty is implementation-defined
        result = _ensure_question("")
        assert isinstance(result, str)
