"""Tests for select verb — ported from tidyverse test-select.r

https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-select.r
"""

import pytest
import polars as pl
from datar import f
from datar.base import c
from datar.dplyr import select, group_by, group_vars, starts_with, ends_with
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


def _gvars(df) -> list:
    return group_vars(df)


# ---------------------------------------------------------------------------
# select columns
# ---------------------------------------------------------------------------

class TestSelectColumns:
    def test_select_single_column(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = df >> select(f.x)
        assert list(out.collect_schema().names()) == ["x"]
        assert out.get_column("x").to_list() == [1, 2, 3]

    def test_select_multiple_columns(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        out = df >> select(f.x, f.z)
        assert list(out.collect_schema().names()) == ["x", "z"]

    def test_select_with_strings(self):
        df = _df({"cyl": [1, 2], "am": [3, 4]})
        out = df >> select("cyl", "am")
        assert list(out.collect_schema().names()) == ["cyl", "am"]

    def test_select_with_helpers(self):
        df = _df({"name": ["a", "b"], "height": [1, 2], "mass": [3, 4], "hair_color": ["x", "y"]})
        out = df >> select(starts_with("h"))
        assert list(out.collect_schema().names()) == ["height", "hair_color"]
        out = df >> select(ends_with("t"))
        assert list(out.collect_schema().names()) == ["height"]
        out = df >> select(ends_with("t") & starts_with("h"))
        assert list(out.collect_schema().names()) == ["height"]


class TestSelectRenames:
    def test_rename_column_with_kwarg(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> select(
            new_x=f.x,
        )
        assert list(out.collect_schema().names()) == ["new_x"]
        assert out.get_column("new_x").to_list() == [1, 2, 3]

    def test_rename_doesnt_preserve_old_name(self):
        df = _df({"a": [1], "b": [2]})
        out = df >> select(
            foo=f.a,
        )
        assert list(out.collect_schema().names()) == ["foo"]

    def test_select_arg_dont_match_internal_args(self):
        df = _df({"a": [1]})
        out = df >> select(
            var=f.a,
        )
        assert list(out.collect_schema().names()) == ["var"]


class TestSelectNoArgs:
    def test_no_args_returns_empty(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> select(__backend="polars")
        assert len(out.collect_schema().names()) == 0

    def test_none_arg_treated_as_empty(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> select(
            None, f.x, None,
        )
        assert list(out.collect_schema().names()) == ["x"]


# ---------------------------------------------------------------------------
# select with grouping
# ---------------------------------------------------------------------------

class TestSelectWithGroupVars:
    def test_preserves_grouping_variables(self):
        df = _df({"g": [1, 2, 3], "x": [3, 2, 1]})
        gf = df >> group_by(f.g)
        out = gf >> select(f.x)
        assert "g" in out.collect_schema().names()
        assert "x" in out.collect_schema().names()
        assert _gvars(out) == ["g"]

    def test_groups_always_included_first(self):
        df = _df({"g": [1, 2], "x": [3, 4], "y": [5, 6]})
        gf = df >> group_by(f.g)
        out = gf >> select(f.y)
        assert list(out.collect_schema().names()) == ["g", "y"]

    def test_select_renamed_groups(self):
        df = _df({"g": [1, 2, 3], "x": [3, 2, 1]})
        gf = df >> group_by(f.g)
        out = gf >> select(
            h=f.g,
        )
        assert _gvars(out) == ["h"]


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

class TestSelectEmpty:
    def test_empty_df_select(self):
        df = _df({"x": [], "y": []})
        out = df >> select(f.x)
        assert out.shape == (0, 1)
        assert list(out.collect_schema().names()) == ["x"]

    def test_empty_df_no_args(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> select(__backend="polars")
        assert len(out.collect_schema().names()) == 0


class TestSelectAllColumns:
    def test_select_all_columns(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        out = df >> select(f.x, f.y)
        assert_df_equal(out, df)

    def test_reorder_columns(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        out = df >> select(f.y, f.x)
        assert list(out.collect_schema().names()) == ["y", "x"]

    def test_select_range_with_c_syntax(self):
        df = _df({"name": ["a"], "height": [1], "mass": [2], "hair_color": ["x"]})
        out = df >> select(c[f.name:f.mass])
        assert list(out.collect_schema().names()) == ["name", "height", "mass"]


class TestSelectErrors:
    def test_select_nonexistent_column(self):
        df = _df({"x": [1]})
        with pytest.raises(KeyError):
            df >> select(f.z)
