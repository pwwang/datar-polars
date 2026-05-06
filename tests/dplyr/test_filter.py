"""Tests for filter verb — ported from tidyverse test-filter.r

https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-filter.r
"""

import pytest
import polars as pl
from datar import f
from datar.base import max_
from datar.dplyr import filter_, group_by, group_vars, mutate, ungroup, row_number
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal, assert_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


def _gvars(df) -> list:
    return group_vars(df)


def _nrow(df) -> int:
    return df.shape[0]


# ---------------------------------------------------------------------------
# basic filtering
# ---------------------------------------------------------------------------

class TestSimpleSymbols:
    def test_filters_by_bool_column(self):
        df = _df({"x": [1, 2, 3, 4], "test": [True, False, True, False]})
        out = df >> filter_(f.test)
        assert out.get_column("x").to_list() == [1, 3]

    def test_filters_by_expression(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> filter_(f.x > 2)
        assert out.get_column("x").to_list() == [3, 4]

    def test_multiple_conditions(self):
        df = _df({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]})
        out = df >> filter_(
            f.x > 1, f.y > 1,
        )
        assert out.get_column("x").to_list() == [2, 3]

    def test_chained_conditions(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> filter_(
            f.x > 1, f.x < 4,
        )
        assert out.get_column("x").to_list() == [2, 3]

    def test_bitwise_and_condition(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> filter_(
            (f.x > 1) & (f.x < 4),
        )
        assert out.get_column("x").to_list() == [2, 3]


class TestNoArgs:
    def test_returns_input_unchanged(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> filter_(__backend="polars")
        assert_df_equal(out, df)

    def test_empty_df_no_args(self):
        df = _df({"x": []})
        out = df >> filter_(__backend="polars")
        assert out.shape == (0, 1)


class TestDiscardsNA:
    def test_na_values_dropped_by_condition(self):
        df = _df({"x": [1, 2, None, 4]})
        out = df >> filter_(
            f.x > 2,
        )
        assert out.get_column("x").to_list() == [4]

    def test_null_values_dropped(self):
        df = _df({"x": [None, None, 3, 4]})
        out = df >> filter_(
            f.x > 2,
        )
        assert out.get_column("x").to_list() == [3, 4]


class TestRowNumber:
    def test_row_number_empty_result(self):
        df = _df({"a": [1, 2, 3]})
        out = df >> filter_(
            row_number() == 4,
        )
        assert _nrow(out) == 0

    def test_row_number_first(self):
        df = _df({"a": [1, 2, 3]})
        out = df >> filter_(
            row_number() == 1,
        )
        assert _nrow(out) == 1
        assert out.get_column("a").to_list() == [1]


class TestEmptyDF:
    def test_empty_df_filter(self):
        df = _df({})
        out = df >> filter_(
            False,
        )
        assert _nrow(out) == 0
        assert len(out.collect_schema().names()) == 0


class TestTrueTrue:
    def test_both_true_returns_all(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        out = df >> filter_(
            True, True,
        )
        assert out.shape == df.shape

    def test_true_scalar(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> filter_(
            True,
        )
        assert_df_equal(out, df)

    def test_false_scalar_empty(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> filter_(
            False,
        )
        assert _nrow(out) == 0


class TestTwoConds:
    def test_multiple_comma_conditions(self):
        df = _df({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        out1 = df >> filter_(
            f.x > 2, f.y > 2,
        )
        out2 = df >> filter_(
            (f.x > 2) & (f.y > 2),
        )
        assert_df_equal(out1, out2)


# ---------------------------------------------------------------------------
# grouping
# ---------------------------------------------------------------------------

class TestPreservesGrouping:
    def test_group_vars_after_filter(self):
        df = _df({"g": [1, 1, 1, 2, 2], "x": [1, 2, 3, 4, 5]})
        gf = df >> group_by(f.g)
        out = gf >> filter_(
            f.x > 2,
        )
        assert _gvars(out) == ["g"]

    def test_filter_then_mutate_keeps_grouping(self):
        df = _df({"g": [1, 1, 1, 2, 2], "x": [1, 2, 3, 4, 5]})
        gf = df >> group_by(f.g)
        out = gf >> filter_(
            f.x > 2,
        )
        out2 = out >> mutate(
            y=f.x * 2,
        )
        assert _gvars(out2) == ["g"]


class TestGroupedFilter:
    def test_filter_within_group(self):
        df = _df({"g": [1, 1, 2, 2], "x": [1, 3, 1, 3]})
        gf = df >> group_by(f.g)
        out = gf >> filter_(
            f.x >= 3,
        )
        # Each group keeps rows where x >= 3
        assert out.shape[0] >= 2
        assert all(v >= 3 for v in out.get_column("x").to_list())

    def test_grouped_filter_chained(self):
        df = _df({"g": [1, 1, 1, 2, 2], "x": [1, 2, 3, 4, 5]})
        out = (
            df
            >> group_by(f.g)
            >> filter_(f.x > 2)
            >> ungroup(__backend="polars")
        )
        assert _gvars(out) == []
        assert all(v > 2 for v in out.get_column("x").to_list())

    def test_filter_max(self):
        df = _df({"g": [1, 1, 2, 2], "x": [1, 3, 1, 4]})
        gf = df >> group_by(f.g)
        out = gf >> filter_(
            f.x == max_(f.x),
        )
        # Each group keeps rows where x is the max in that group
        assert out.shape == (2, 2)
        assert set(out.get_column("x").to_list()) == {3, 4}


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TestErrors:
    def test_wrong_size_raises(self):
        df = _df({"x": [1, 2, 3]})
        with pytest.raises(ValueError):
            df >> filter_(
                [True, False],
            )

    def test_named_args_raises(self):
        df = _df({"x": [1, 2]})
        with pytest.raises(TypeError):
            df >> filter_(
                x=1,
            )
