"""Tests for distinct verb — ported from tidyverse test-distinct.R

https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-distinct.R
"""

import pytest
import polars as pl
from datar import f
from datar.dplyr import distinct, group_by, group_vars, mutate
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


def _gvars(df) -> list:
    return group_vars(df)


# ---------------------------------------------------------------------------
# basic distinct
# ---------------------------------------------------------------------------

class TestDistinctAllUnique:
    def test_distinct_no_args_deduplicates_all(self):
        df = _df({"x": [1, 1, 2, 2], "y": [1, 2, 1, 2]})
        out = df >> distinct(__backend="polars")
        assert out.shape == (4, 2)

    def test_distinct_duplicate_rows_removed(self):
        df = _df({"x": [1, 1, 2], "y": [1, 1, 3]})
        out = df >> distinct(__backend="polars")
        assert out.shape == (2, 2)
        assert out.get_column("x").to_list() == [1, 2]
        assert out.get_column("y").to_list() == [1, 3]

    def test_distinct_all_unique_already(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = df >> distinct(__backend="polars")
        assert_df_equal(out, df)

    def test_distinct_preserves_order(self):
        df = _df({"x": [2, 1, 3, 1], "y": [4, 5, 6, 5]})
        out = df >> distinct(__backend="polars")
        assert out.get_column("x").to_list() == [2, 1, 3]


class TestDistinctSpecificColumns:
    def test_distinct_by_single_col(self):
        df = _df({"x": [1, 1, 2], "y": [3, 4, 5]})
        out = df >> distinct(f.x)
        assert out.shape == (2, 1)
        assert list(out.collect_schema().names()) == ["x"]
        assert sorted(out.get_column("x").to_list()) == [1, 2]

    def test_distinct_by_multiple_cols(self):
        df = _df({"x": [1, 1, 2, 2], "y": [1, 2, 1, 2]})
        out = df >> distinct(
            f.x, f.y,
        )
        assert out.shape == (4, 2)

    def test_distinct_by_col_string(self):
        df = _df({"x": [1, 1, 2], "y": [3, 4, 5]})
        out = df >> distinct("x")
        assert list(out.collect_schema().names()) == ["x"]
        assert sorted(out.get_column("x").to_list()) == [1, 2]

    def test_distinct_doesnt_duplicate_cols(self):
        df = _df({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = df >> distinct(
            f.a, f.a,
        )
        assert list(out.collect_schema().names()) == ["a"]

    def test_distinct_by_expr(self):
        df = _df({"x": [1, 1, 2, 2], "y": [1, 2, 3, 4]})
        out = df >> distinct(
            diff=f.x - f.y,
        )
        assert list(out.collect_schema().names()) == ["diff"]
        assert out.get_column("diff").to_list() == [0, -1, -2]


# ---------------------------------------------------------------------------
# distinct with keep_all
# ---------------------------------------------------------------------------

class TestDistinctKeepAll:
    def test_distinct_keep_all_true(self):
        df = _df({"x": [1, 1, 1], "y": [3, 2, 1]})
        out = df >> distinct(
            f.x, _keep_all=True,
        )
        assert out.shape == (1, 2)
        assert list(out.collect_schema().names()) == ["x", "y"]
        assert out.get_column("x").to_list() == [1]
        # Keeps the first matching row
        assert out.get_column("y").to_list() == [3]

    def test_distinct_keep_all_default_false(self):
        df = _df({"x": [1, 1], "y": [3, 4]})
        out = df >> distinct(f.x)
        assert list(out.collect_schema().names()) == ["x"]


# ---------------------------------------------------------------------------
# distinct with grouping
# ---------------------------------------------------------------------------

class TestDistinctGrouped:
    def test_distinct_grouping_cols_always_included(self):
        df = _df({"g": [1, 2, 2], "x": [1, 2, 3]})
        gf = df >> group_by(f.g)
        out = gf >> distinct(f.x)
        assert "g" in out.collect_schema().names()
        assert "x" in out.collect_schema().names()

    def test_distinct_preserves_grouping(self):
        df = _df({"g": [1, 1, 2, 2], "x": [1, 2, 3, 4]})
        gf = df >> group_by(f.g)
        out = gf >> distinct(__backend="polars")
        assert _gvars(out) == ["g"]

    def test_distinct_grouped_vs_ungrouped_equivalent(self):
        df = _df({"g": [1, 2], "x": [1, 2]})
        out1 = df >> distinct(__backend="polars") >> group_by(
            f.g,
        )
        out2 = df >> group_by(
            f.g,
        ) >> distinct(__backend="polars")
        # Both should have same data values
        assert sorted(out1.get_column("g").to_list()) == sorted(
            out2.get_column("g").to_list()
        )


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

class TestDistinctEmpty:
    def test_distinct_empty_df(self):
        df = _df({"x": [], "y": []})
        out = df >> distinct(__backend="polars")
        assert out.shape == (0, 2)

    def test_distinct_empty_df_with_cols(self):
        df = _df({"x": [], "y": []})
        out = df >> distinct(f.x)
        assert out.shape == (0, 1)
        assert list(out.collect_schema().names()) == ["x"]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TestDistinctErrors:
    def test_distinct_nonexistent_col(self):
        df = _df({"x": [1, 2]})
        with pytest.raises(KeyError):
            df >> distinct(f.z)
