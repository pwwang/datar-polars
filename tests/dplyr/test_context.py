"""Tests for context functions: n, cur_data, cur_data_all, cur_group,
cur_group_id, cur_group_rows, cur_column, consecutive_id."""

import pytest

from datar import f
from datar.dplyr import (
    n,
    cur_data,
    cur_data_all,
    cur_group,
    cur_group_id,
    cur_group_rows,
    cur_column,
    consecutive_id,
    summarise,
    mutate,
    group_by,
    pull,
    across,
)
from datar.tibble import tibble
from datar_polars.tibble import as_tibble

from ..conftest import assert_iterable_equal


def _df(data: dict):
    return as_tibble(__import__("polars").DataFrame(data))


# ── n ───────────────────────────────────────────────────────────────────────


def test_n_ungrouped():
    """n() returns number of rows for ungrouped data."""
    df = _df({"x": [1, 2, 3]})
    out = df >> summarise(count=n())
    assert out.get_column("count").to_list() == [3]


def test_n_grouped():
    """n() returns group sizes for grouped data."""
    df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
    gf = df >> group_by(f.g)
    out = gf >> summarise(count=n())
    counts = out.get_column("count").to_list()
    assert sorted(counts) == [1, 2]


# ── cur_data / cur_data_all ─────────────────────────────────────────────────


def test_cur_data_ungrouped():
    """cur_data() returns the full dataframe for ungrouped data."""
    df = _df({"x": [1, 2], "y": [3, 4]})
    out = summarise(
        df,
        n=cur_data(),
        __ast_fallback="normal",
        __backend="polars",
    )
    result = out.get_column("n").to_list()
    assert len(result) == 1
    assert result[0]["x"].to_list() == [1, 2]
    assert result[0]["y"].to_list() == [3, 4]


def test_cur_data_grouped():
    """cur_data() excludes group vars in grouped context."""
    df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
    gf = df >> group_by(f.g)
    out = gf >> summarise(
        result=cur_data()
    )
    # cur_data for each group should have only non-group columns
    vals = out.get_column("result").to_list()
    assert len(vals) == 2
    assert vals[0]["x"].to_list() == [1, 2]
    assert vals[1]["x"].to_list() == [3]


def test_cur_data_all_ungrouped():
    """cur_data_all() includes group vars."""
    df = _df({"x": [1, 2, 3]})
    out = df >> summarise(
        result=cur_data_all()
    )
    result = out.get_column("result").to_list()
    assert len(result) == 1
    assert result[0]["x"].to_list() == [1, 2, 3]


def test_cur_data_all_grouped():
    """cur_data_all() includes group vars in grouped context."""
    df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
    gf = df >> group_by(f.g)
    out = gf >> summarise(
        result=cur_data_all()
    )
    vals = out.get_column("result").to_list()
    assert len(vals) == 2
    assert vals[0]["g"].to_list() == ["a", "a"]
    assert vals[0]["x"].to_list() == [1, 2]
    assert vals[1]["g"].to_list() == ["b"]
    assert vals[1]["x"].to_list() == [3]


# ── cur_group / cur_group_id ────────────────────────────────────────────────


def test_cur_group_id_ungrouped():
    """cur_group_id() returns 0 for ungrouped data."""
    df = _df({"x": [1, 2]})
    out = df >> summarise(
        gid=cur_group_id()
    )
    assert out.get_column("gid").to_list() == [0]


def test_cur_group_id_grouped():
    """cur_group_id() returns group indices for grouped data."""
    df = _df({"g": ["b", "a", "b"]})
    gf = df >> group_by(f.g)
    out = gf >> summarise(
        gid=cur_group_id()
    )
    gids = out.get_column("gid").to_list()
    assert len(gids) == 2

    out = gf >> mutate(
        gid=cur_group_id()
    )
    assert out.get_column("gid").to_list() == [0, 1, 0]


def test_cur_group_ungrouped():
    """cur_group() returns empty for ungrouped."""
    df = _df({"x": [1, 2]})
    out = df >> summarise(
        key=cur_group()
    )
    assert out.shape[0] == 1


def test_cur_group_grouped():
    """cur_group() returns group keys for grouped data."""
    df = _df({"g": ["b", "a", "b"]})
    gf = df >> group_by(f.g)
    out = gf >> summarise(
        key=cur_group()
    )
    keys = out.get_column("key")
    assert list(keys["g"]) == ["b", "a"]


# ── cur_group_rows ──────────────────────────────────────────────────────────


def test_cur_group_rows_ungrouped():
    """cur_group_rows() returns all row indices for ungrouped data."""
    df = _df({"x": [1, 2, 3]})
    out = summarise(
        df,
        rows=cur_group_rows(),
        __ast_fallback="normal",
        __backend="polars",
    )
    rows = out.get_column("rows").to_list()
    assert rows == [[0, 1, 2]]


def test_cur_group_rows_grouped():
    """cur_group_rows() returns row indices for each group."""
    df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
    gf = df >> group_by(f.g)
    out = gf >> summarise(
        rows=cur_group_rows()
    )
    rows = out.get_column("rows").to_list()
    assert rows == [[0, 1], [2]]


# ── cur_column ──────────────────────────────────────────────────────────────


def test_cur_column_returns_marker():
    """cur_column() returns a CurColumn marker."""
    marker = cur_column(__backend="polars")
    assert marker is not None
    # It should be an instance of the CurColumn class
    from datar_polars.api.dplyr.context import CurColumn

    assert isinstance(marker, CurColumn)


# ── consecutive_id ──────────────────────────────────────────────────────────


def test_consecutive_id_simple():
    """consecutive_id assigns IDs based on value changes."""
    result = consecutive_id([1, 1, 2, 1, 2])
    assert result.to_list() == [0, 0, 1, 2, 3]


def test_consecutive_id_multi():
    """consecutive_id with multiple series."""
    result = consecutive_id(
        [1, 1, 2], [10, 10, 20]
    )
    # Both change at row 2
    assert result.to_list() == [0, 0, 1]
