"""Tests for dplyr expression-level helper functions: n, between,
coalesce, na_if, nth, first, last, if_else, case_when.
"""

import pytest
import polars as pl
from datar import f
from datar.data import starwars
from datar.base import rnorm, NULL, NA
from datar.dplyr import (
    n,
    between,
    coalesce,
    na_if,
    nth,
    first,
    last,
    if_else,
    case_when,
    mutate,
    summarise,
    filter_,
    group_by,
    pull,
)
from datar.tibble import tibble
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── n() ─────────────────────────────────────────────────────────────────


class TestN:
    def test_n_in_summarise(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(cnt=n())
        assert out.get_column("cnt").to_list() == [3]

    def test_n_in_mutate(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(cnt=n())
        assert out.get_column("cnt").to_list() == [3, 3, 3]

    def test_n_grouped(self):
        df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
        out = (
            df
            >> group_by(f.g)
            >> summarise(cnt=n())
        )
        out = out.sort("g")
        assert out.get_column("cnt").to_list() == [2, 1]


# ── between ─────────────────────────────────────────────────────────────


class TestBetween:
    def test_between_in_filter(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        out = df >> filter_(
            between(f.x, 2, 4)
        )
        assert out.get_column("x").to_list() == [2, 3, 4]

    def test_between_in_filter2(self):
        df = filter_(starwars, between(f.height, 100, 150), __ast_fallback="normal")
        assert df.shape == (5, 11)

    def test_between_scalar(self):
        assert between(3, 1, 5) is True
        assert between(0, 1, 5) is False

    def test_between_range(self):
        result = between(range(1, 13), 7, 9)
        assert result == [False] * 6 + [True] * 3 + [False] * 3

    def test_between_exclusive(self):
        assert between(1, 1, 5, inclusive="right") is False  # 1 < 1 is False
        assert between(1, 1, 5, inclusive="neither") is False

    def test_between_rnorm(self):
        x = rnorm(100)
        result = x[between(x, -1, 1)]
        assert all(-1 <= val <= 1 for val in result)
        assert len(result) < 100


# ── coalesce (dplyr expr version) ───────────────────────────────────────


class TestCoalesceExpr:
    def test_coalesce_in_mutate(self):
        df = _df({"a": [1, None, 3], "b": [4, 5, None]})
        out = df >> mutate(
            y=coalesce(f.a, f.b)
        )
        assert out.get_column("y").to_list() == [1, 5, 3]

    def test_coalesce_in_mutate2(self):
        df = tibble(x=[5,4,3,NA,2,NA,1,NA])
        out = df >> mutate(y=coalesce(f.x, 0)) >> pull(f.y)
        assert list(out) == [5, 4, 3, 0, 2, 0, 1, 0]


# ── na_if (dplyr expr version) ──────────────────────────────────────────


class TestNaIfExpr:
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

    def test_na_if_with_range(self):
        out = na_if(range(5), list(range(4,-1,-1)))
        assert out == [0, 1, None, 3, 4]


# ── nth (dplyr version) ─────────────────────────────────────────────────


class TestNthExpr:
    def test_nth_in_summarise(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> summarise(
            y=nth(f.x, 0)
        )
        assert out.get_column("y").to_list() == [10]


# ── first (dplyr version) ───────────────────────────────────────────────


class TestFirstExpr:
    def test_first_in_summarise(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> summarise(
            y=first(f.x)
        )
        assert out.get_column("y").to_list() == [10]


# ── last (dplyr version) ────────────────────────────────────────────────


class TestLastExpr:
    def test_last_in_summarise(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> summarise(
            y=last(f.x)
        )
        assert out.get_column("y").to_list() == [30]


# ── if_else (dplyr Tibble version) ──────────────────────────────────────


class TestIfElseDplyr:
    def test_if_else_chain(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=if_else(f.x > 2, "high", "low"),
            __ast_fallback="normal",
            __backend="polars",
        )
        vals = out.get_column("y").to_list()
        assert vals == ["low", "low", "high", "high"]


# ── case_when (dplyr Tibble version) ────────────────────────────────────


class TestCaseWhenDplyr:
    def test_case_when_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=case_when((f.x == 1, "one"), (f.x == 2, "two"), True, "other"),
            __ast_fallback="normal",
            __backend="polars",
        )
        vals = out.get_column("y").to_list()
        assert vals == ["one", "two", "other", "other"]
