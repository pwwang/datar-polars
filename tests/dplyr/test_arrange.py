"""Tests for arrange verb — ported from tidyverse test-arrange.r

https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-arrange.r
"""

import pytest
import polars as pl
from datar import f
from datar.base import c, factor, letters
from datar.data import mtcars
from datar.dplyr import arrange, desc, group_by, group_vars, across, filter_
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal, assert_iterable_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


def _gvars(df) -> list:
    return group_vars(df)


# ---------------------------------------------------------------------------
# basic arrange
# ---------------------------------------------------------------------------

class TestArrangeAscending:
    def test_arrange_single_column(self):
        df = _df({"x": [3, 1, 2]})
        out = df >> arrange(f.x)
        assert out.get_column("x").to_list() == [1, 2, 3]

    def test_arrange_preserves_shape(self):
        df = _df({"x": [3, 1, 4, 1, 5], "y": [1, 2, 3, 4, 5]})
        out = df >> arrange(f.x)
        assert out.shape == df.shape

    def test_arrange_na_last(self):
        df = _df({"x": [4.0, 3.0, None]})
        out = df >> arrange(f.x)
        vals = out.get_column("x").to_list()
        # polars defaults to nulls first
        assert vals == [None, 3.0, 4.0]


class TestArrangeDesc:
    def test_arrange_descending_single(self):
        df = _df({"x": [1, 3, 2]})
        out = df >> arrange(
            desc(f.x),
        )
        assert out.get_column("x").to_list() == [3, 2, 1]

    def test_arrange_desc_na_last(self):
        df = _df({"x": [4.0, 3.0, None]})
        out = df >> arrange(
            desc(f.x),
        )
        vals = out.get_column("x").to_list()
        # polars defaults nulls first for desc (unlike R which puts NA last)
        assert vals[0] is None
        assert vals[1] == 4.0
        assert vals[2] == 3.0


class TestArrangeWithAcross:
    def test_arrange_with_across(self):
        df = _df({"x": [1, 1, 2, 2], "y": [2, 1, 4, 3], "z": [10, 40, 30, 20]})
        out = df >> arrange(
            across(c[f.y:]),
            __ast_fallback="piping",
            __backend="polars",
        )
        xv = out.get_column("x").to_list()
        yv = out.get_column("y").to_list()
        zv = out.get_column("z").to_list()
        assert xv == [1, 1, 2, 2]
        assert yv == [1, 2, 3, 4]
        assert zv == [40, 10, 20, 30]


class TestArrangeMultipleColumns:
    def test_arrange_two_columns(self):
        df = _df({"x": [1, 1, 2, 2], "y": [2, 1, 4, 3]})
        out = df >> arrange(
            f.x, f.y,
        )
        xv = out.get_column("x").to_list()
        yv = out.get_column("y").to_list()
        assert xv == [1, 1, 2, 2]
        assert yv == [1, 2, 3, 4]

    def test_arrange_mixed_directions(self):
        df = _df({"x": [1, 1, 2, 2], "y": [1, 2, 3, 4]})
        out = df >> arrange(
            f.x, desc(f.y),
        )
        xv = out.get_column("x").to_list()
        yv = out.get_column("y").to_list()
        assert xv == [1, 1, 2, 2]
        assert yv == [2, 1, 4, 3]


# ---------------------------------------------------------------------------
# empty and edge cases
# ---------------------------------------------------------------------------

class TestArrangeEmpty:
    def test_empty_returns_self(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> arrange(__backend="polars")
        assert_df_equal(out, df)

    def test_empty_df_arrange(self):
        df = _df({"x": [], "y": []})
        out = df >> arrange(f.x)
        assert out.shape == (0, 2)

    def test_empty_df_no_args(self):
        df = _df({"x": [], "y": []})
        out = df >> arrange(__backend="polars")
        assert out.shape == (0, 2)


# ---------------------------------------------------------------------------
# arrange with grouping
# ---------------------------------------------------------------------------

class TestArrangeByGroup:
    def test_arrange_ignores_group(self):
        df = _df({"g": [2, 1, 2, 1], "x": [4, 3, 2, 1]})
        gf = df >> group_by(f.g)
        out = gf >> arrange(f.x)
        assert out.get_column("x").to_list() == [1, 2, 3, 4]

    def test_arrange_by_group_true(self):
        df = _df({"g": [2, 1, 2, 1], "x": [4, 3, 2, 1]})
        gf = df >> group_by(f.g)
        out = gf >> arrange(
            f.x, _by_group=True,
        )
        # Within group g=1: x=[3,1] -> sorted = [1,3]
        # Within group g=2: x=[4,2] -> sorted = [2,4]
        xv = out.get_column("x").to_list()
        gv = out.get_column("g").to_list()
        # Groups should appear in order: g=1 rows then g=2 rows
        g1_vals = [x for g, x in zip(gv, xv) if g == 1]
        g2_vals = [x for g, x in zip(gv, xv) if g == 2]
        assert g1_vals == [1, 3]
        assert g2_vals == [2, 4]

    def test_arrange_preserves_groups(self):
        df = _df({"g": [1, 1, 2, 2], "x": [4, 3, 2, 1]})
        gf = df >> group_by(f.g)
        out = gf >> arrange(f.x)
        assert _gvars(out) == ["g"]

    def test_arrange_group_by_mtchars(self):
        by_cyl = mtcars >> group_by(f.cyl)
        out = by_cyl >> arrange(desc(f.wt))
        # Within each cyl group, wt should be in descending order
        out8 = out >> filter_(f.cyl == 8)
        assert out8.get_column("wt").to_list() == sorted(
            out8.get_column("wt").to_list(), reverse=True
        )
        out6 = out >> filter_(f.cyl == 6)
        assert out6.get_column("wt").to_list() == sorted(
            out6.get_column("wt").to_list(), reverse=True
        )
        out4 = out >> filter_(f.cyl == 4)
        assert out4.get_column("wt").to_list() == sorted(
            out4.get_column("wt").to_list(), reverse=True
        )

# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TestArrangeErrors:
    def test_arrange_nonexistent_column(self):
        df = _df({"x": [1, 2]})
        # For eager DataFrames, error is raised at arrange time
        with pytest.raises(Exception):
            df >> arrange(f.z)


# --- desc -------------------------------------------------------------------
def test_desc():
    out = desc([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert_iterable_equal(out, [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])

    out = desc(range(1, 11))
    assert_iterable_equal(out, [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
