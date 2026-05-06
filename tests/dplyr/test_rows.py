"""Tests for rows_insert, rows_update, rows_upsert, rows_delete, rows_patch, rows_append."""

import pytest
import polars as pl

from datar.dplyr import (
    rows_insert,
    rows_update,
    rows_upsert,
    rows_delete,
    rows_patch,
    rows_append,
)
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


# ── rows_append ─────────────────────────────────────────────────────────────


def test_rows_append_simple():
    """rows_append adds rows from y to x."""
    x = _df({"a": [1, 2], "b": [3, 4]})
    y = _df({"a": [5, 6], "b": [7, 8]})
    result = rows_append(x, y)
    assert result.shape[0] == 4
    assert result.get_column("a").to_list() == [1, 2, 5, 6]


def test_rows_append_column_mismatch_error():
    """rows_append requires y columns to exist in x."""
    x = _df({"a": [1]})
    y = _df({"b": [2]})
    with pytest.raises(ValueError, match="columns"):
        rows_append(x, y)


# ── rows_insert ─────────────────────────────────────────────────────────────


def test_rows_insert_new_rows():
    """rows_insert adds only new rows (by key)."""
    x = _df({"id": [1, 2], "val": ["a", "b"]})
    y = _df({"id": [2, 3], "val": ["b2", "c"]})
    result = rows_insert(
        x, y, by="id", conflict="ignore"
    )
    assert result.shape[0] == 3
    ids = result.get_column("id").to_list()
    assert 3 in ids


def test_rows_insert_conflict_error():
    """rows_insert raises error on conflict by default."""
    x = _df({"id": [1, 2], "val": ["a", "b"]})
    y = _df({"id": [2, 3], "val": ["b2", "c"]})
    with pytest.raises(ValueError):
        rows_insert(x, y, by="id")


def test_rows_insert_conflict_ignore():
    """rows_insert with conflict='ignore' skips conflicting rows."""
    x = _df({"id": [1, 2], "val": ["a", "b"]})
    y = _df({"id": [2, 3], "val": ["b2", "c"]})
    result = rows_insert(
        x, y, by="id", conflict="ignore"
    )
    assert result.shape[0] == 3
    assert result.get_column("id").to_list() == [1, 2, 3]


# ── rows_update ─────────────────────────────────────────────────────────────


def test_rows_update_existing():
    """rows_update updates matching rows."""
    x = _df({"id": [1, 2], "val": ["a", "b"]})
    y = _df({"id": [2], "val": ["updated"]})
    result = rows_update(
        x, y, by="id"
    )
    vals = result.sort("id").get_column("val").to_list()
    assert "updated" in vals
    assert "a" in vals


def test_rows_update_unmatched_error():
    """rows_update raises error for unmatched keys by default."""
    x = _df({"id": [1, 2], "val": ["a", "b"]})
    y = _df({"id": [3], "val": ["c"]})
    with pytest.raises(ValueError, match="missing"):
        rows_update(x, y, by="id")


# ── rows_delete ─────────────────────────────────────────────────────────────


def test_rows_delete_matching():
    """rows_delete removes matching rows."""
    x = _df({"id": [1, 2, 3], "val": ["a", "b", "c"]})
    y = _df({"id": [2]})
    result = rows_delete(
        x, y, by="id"
    )
    assert result.shape[0] == 2
    assert 2 not in result.get_column("id").to_list()


def test_rows_delete_unmatched_error():
    """rows_delete raises error for unmatched keys by default."""
    x = _df({"id": [1, 2]})
    y = _df({"id": [3]})
    with pytest.raises(ValueError, match="missing"):
        rows_delete(x, y, by="id")


# ── rows_upsert ─────────────────────────────────────────────────────────────


def test_rows_upsert_insert_and_update():
    """rows_upsert inserts new rows and updates existing ones."""
    x = _df({"id": [1, 2], "val": ["a", "b"]})
    y = _df({"id": [2, 3], "val": ["b2", "c"]})
    result = rows_upsert(
        x, y, by="id"
    )
    assert result.shape[0] == 3
    result_sorted = result.sort("id")
    assert result_sorted.get_column("val").to_list() == ["a", "b2", "c"]


# ── rows_patch ──────────────────────────────────────────────────────────────


def test_rows_patch_fills_nas():
    """rows_patch fills NAs in x with values from y."""
    x = _df({"id": [1, 2], "val": [None, "b"]})
    y = _df({"id": [1], "val": ["patched"]})
    result = rows_patch(
        x, y, by="id"
    )
    result_sorted = result.sort("id")
    assert result_sorted.get_column("val").to_list() == ["patched", "b"]
