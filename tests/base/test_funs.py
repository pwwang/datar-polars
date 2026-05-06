"""Tests for base conditional and utility functions: if_else, case_when,
coalesce, na_if, nth, first, last.
"""

import pytest
import polars as pl
from datar import f
from datar.dplyr import if_else, case_when, coalesce, na_if, nth, first, last
from datar.dplyr import mutate, summarise
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── if_else ─────────────────────────────────────────────────────────────


class TestIfElse:
    def test_if_else_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=if_else(f.x > 2, "high", "low"),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["low", "low", "high", "high"]

    def test_if_else_scalar_true(self):
        result = if_else(True, 1, 2)
        assert result == 1

    def test_if_else_scalar_false(self):
        result = if_else(False, 1, 2)
        assert result == 2

    def test_if_else_with_missing(self):
        df = _df({"x": [1, None, 3]})
        out = df >> mutate(
            y=if_else(f.x > 1, f.x, 0, missing=99),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").to_list() == [0, 99, 3]


# ── case_when ───────────────────────────────────────────────────────────


class TestCaseWhen:
    def test_case_when_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=case_when((f.x == 1, "one"), (f.x == 2, "two"), True, "other"),
            __ast_fallback="normal",
            __backend="polars",
        )
        vals = out.get_column("y").to_list()
        assert vals == ["one", "two", "other", "other"]

    def test_case_when_with_default(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> mutate(
            y=case_when((f.x < 15, "low"), (f.x < 25, "mid"), True, "high"),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["low", "mid", "high"]


# ── coalesce ────────────────────────────────────────────────────────────


class TestCoalesce:
    def test_coalesce_in_mutate(self):
        df = _df({"a": [1, None, 3], "b": [4, 5, None]})
        out = df >> mutate(
            y=coalesce(f.a, f.b)
        )
        assert out.get_column("y").to_list() == [1, 5, 3]

    def test_coalesce_scalar_first_non_null(self):
        assert coalesce(None, None, 42) == 42

    def test_coalesce_scalar_all_null(self):
        assert coalesce(None, None) is None


# ── na_if ───────────────────────────────────────────────────────────────


class TestNaIf:
    def test_na_if_in_mutate(self):
        df = _df({"x": [1, 2, 3, 2]})
        out = df >> mutate(
            y=na_if(f.x, 2)
        )
        vals = out.get_column("y").to_list()
        assert vals[0] == 1
        assert vals[1] is None
        assert vals[2] == 3
        assert vals[3] is None

    def test_na_if_scalar_match(self):
        assert na_if(5, 5) is None

    def test_na_if_scalar_no_match(self):
        assert na_if(5, 3) == 5


# ── nth ─────────────────────────────────────────────────────────────────


class TestNth:
    def test_nth_in_summarise(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> summarise(
            y=nth(f.x, 1)
        )
        assert out.get_column("y").to_list() == [20]

    def test_nth_scalar(self):
        assert nth([10, 20, 30], 0) == 10
        assert nth([10, 20, 30], 2) == 30

    def test_nth_out_of_bounds(self):
        assert nth([10, 20], 10, default=999) == 999


# ── first ───────────────────────────────────────────────────────────────


class TestFirst:
    def test_first_in_summarise(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> summarise(
            y=first(f.x)
        )
        assert out.get_column("y").to_list() == [10]

    def test_first_scalar(self):
        assert first([10, 20, 30]) == 10

    def test_first_empty_default(self):
        assert first([], default=999) == 999


# ── last ────────────────────────────────────────────────────────────────


class TestLast:
    def test_last_in_summarise(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> summarise(
            y=last(f.x)
        )
        assert out.get_column("y").to_list() == [30]

    def test_last_scalar(self):
        assert last([10, 20, 30]) == 30

    def test_last_empty_default(self):
        assert last([], default=999) == 999

    def test_last_with_2(self):
        x = range(10)
        y = range(9, -1, -1)
        out = last(x, y)
        assert out == 0
