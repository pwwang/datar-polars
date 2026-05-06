"""Tests for tidyselect helpers: starts_with, ends_with, contains,
matches, all_of, any_of, num_range, everything, last_col, where.
"""

import pytest
import polars as pl
from datar import f
from datar.dplyr import (
    starts_with,
    ends_with,
    contains,
    matches,
    all_of,
    any_of,
    num_range,
    everything,
    last_col,
    where,
    select,
)
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── starts_with ─────────────────────────────────────────────────────────


class TestStartsWith:
    def test_starts_with_in_select(self):
        df = _df({"x1": [1], "x2": [2], "y1": [3]})
        out = df >> select(
            starts_with("x")
        )
        assert list(out.collect_schema().names()) == ["x1", "x2"]

    def test_starts_with_case_insensitive(self):
        df = _df({"Abc": [1], "abd": [2], "xyz": [3]})
        out = df >> select(
            starts_with("ab")
        )
        assert list(out.collect_schema().names()) == ["Abc", "abd"]


# ── ends_with ───────────────────────────────────────────────────────────


class TestEndsWith:
    def test_ends_with_in_select(self):
        df = _df({"name_x": [1], "name_y": [2], "other": [3]})
        out = df >> select(
            ends_with("_x")
        )
        assert list(out.collect_schema().names()) == ["name_x"]


# ── contains ────────────────────────────────────────────────────────────


class TestContains:
    def test_contains_in_select(self):
        df = _df({"col_a": [1], "col_b": [2], "other": [3]})
        out = df >> select(
            contains("col_")
        )
        assert list(out.collect_schema().names()) == ["col_a", "col_b"]


# ── matches ─────────────────────────────────────────────────────────────


class TestMatches:
    def test_matches_in_select(self):
        df = _df({"x1": [1], "x2": [2], "y1": [3], "y2": [4]})
        out = df >> select(
            matches(r"^x\d$")
        )
        assert list(out.collect_schema().names()) == ["x1", "x2"]


# ── all_of ──────────────────────────────────────────────────────────────


class TestAllOf:
    def test_all_of_in_select(self):
        df = _df({"a": [1], "b": [2], "c": [3]})
        out = df >> select(
            all_of(["a", "c"])
        )
        assert list(out.collect_schema().names()) == ["a", "c"]


# ── any_of ──────────────────────────────────────────────────────────────


class TestAnyOf:
    def test_any_of_in_select(self):
        df = _df({"a": [1], "b": [2]})
        out = df >> select(
            any_of(["a", "c"])
        )
        assert list(out.collect_schema().names()) == ["a"]

    def test_any_of_no_match(self):
        df = _df({"a": [1]})
        out = df >> select(
            any_of(["z"])
        )
        assert out.shape[1] == 0  # no columns selected


# ── num_range ───────────────────────────────────────────────────────────


class TestNumRange:
    def test_num_range_basic(self):
        result = num_range("x", 3)
        assert result == ["x0", "x1", "x2"]


# ── everything ──────────────────────────────────────────────────────────


class TestEverything:
    def test_everything_in_select(self):
        df = _df({"a": [1], "b": [2], "c": [3]})
        out = df >> select(everything())
        assert list(out.collect_schema().names()) == ["a", "b", "c"]


# ── last_col ────────────────────────────────────────────────────────────


class TestLastCol:
    def test_last_col_in_select(self):
        df = _df({"a": [1], "b": [2], "c": [3]})
        out = df >> select(last_col())
        assert list(out.collect_schema().names()) == ["c"]


# ── where ───────────────────────────────────────────────────────────────


class TestWhere:
    def test_where_is_numeric(self):
        df = _df({"a": [1], "b": [2.0], "c": ["text"]})

        def is_num(x):
            return x.dtype.is_numeric()

        # where() is registered and should work in select context
        out = df >> select(
            where(is_num)
        )
        assert "c" not in list(out.collect_schema().names())
        assert "a" in list(out.collect_schema().names())
