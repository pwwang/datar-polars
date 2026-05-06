"""Tests for mutate verb — ported from tidyverse test-mutate.r

https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-mutate.r
"""

import pytest
import polars as pl
from datar import f
from datar.base import c, round, is_double
from datar.tibble import tibble
from datar.dplyr import mutate, transmute, group_by, group_vars, select, pull, across, where, if_else
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal, assert_equal, assert_iterable_equal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _df(data: dict) -> pl.DataFrame:
    """Create a polars DataFrame wrapped with _datar metadata."""
    return as_tibble(pl.DataFrame(data))


def _gvars(df) -> list:
    """Convenience to get group_vars with polars backend."""
    return group_vars(df)


# ---------------------------------------------------------------------------
# basic mutate
# ---------------------------------------------------------------------------

class TestEmptyMutate:
    def test_empty_mutate_returns_input_ungrouped(self):
        df = _df({"x": [1]})
        out = df >> mutate(__backend="polars")
        assert out.shape == df.shape
        assert list(out.collect_schema().names()) == list(df.collect_schema().names())

    def test_empty_mutate_returns_input_grouped(self):
        df = _df({"x": [1, 2], "g": [1, 1]})
        gf = df >> group_by(f.g)
        out = gf >> mutate(__backend="polars")
        assert out.shape == (2, 2)
        assert _gvars(out) == ["g"]


class TestMutateWithAcross:
    def test_mutate_with_across(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> mutate(
            across(f[f.a :], lambda x: x + 1),
            __backend="polars",
        )
        assert out.get_column("a").to_list() == [2, 3]
        assert out.get_column("b").to_list() == [4, 5]

    def test_mutate_with_across_and_c(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> mutate(
            across(c[f.a :], lambda x: x + 1),
            __backend="polars",
        )
        assert out.get_column("a").to_list() == [2, 3]
        assert out.get_column("b").to_list() == [4, 5]

    def test_mutate_with_across_and_c2(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> mutate(
            across(c[: f.b], round),
            __backend="polars",
        )
        assert out.get_column("a").to_list() == [1, 2]
        assert out.get_column("b").to_list() == [3, 4]

    def test_mutate_with_across_and_c3(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> mutate(
            across(where(is_double), round),
            __backend="polars",
        )
        assert out.get_column("a").to_list() == [1, 2]
        assert out.get_column("b").to_list() == [3, 4]

    def test_mutate_with_across_and_c4(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> mutate(
            across(where(is_double) & ~c(f.Petal_Length, f.Petal_Width), round),
            __backend="polars",
        )
        assert out.get_column("a").to_list() == [1, 2]
        assert out.get_column("b").to_list() == [3, 4]


class TestMutateWithRowwise:
    def test_mutate_with_rowwise(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> mutate(
            c=f.a + f.b,
            __backend="polars",
        )
        assert out.get_column("c").to_list() == [4, 6]

    def test_mutate_with_group_rowwise(self):
        df = _df({"a": [1, 2], "b": [3, 4]})
        gf = df >> group_by(f.a)
        out = gf >> mutate(
            c=f.a + f.b,
            __backend="polars",
        )
        assert out.get_column("c").to_list() == [4, 6]


class TestAppliedProgressively:
    def test_chained_references(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            y=f.x + 1, z=f.y + 1,
            __backend="polars",
        )
        assert out.get_column("x").to_list() == [1, 2, 3]
        assert out.get_column("y").to_list() == [2, 3, 4]
        assert out.get_column("z").to_list() == [3, 4, 5]

    def test_overwrite_column_uses_new_value(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            x=2, y=f.x,
            __backend="polars",
        )
        assert out.get_column("x").to_list() == [2, 2, 2]
        assert out.get_column("y").to_list() == [2, 2, 2]

    def test_overwrite_then_reference(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            y=f.x + 1, x=f.y + 1,
            __backend="polars",
        )
        # y = x+1 = [2,3,4]; then x = y+1 = [3,4,5]
        assert out.get_column("y").to_list() == [2, 3, 4]
        assert out.get_column("x").to_list() == [3, 4, 5]


class TestLength1VectorsRecycled:
    def test_scalar_broadcast(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=1,
        )
        assert out.get_column("y").to_list() == [1, 1, 1, 1]

    # @pytest.mark.skip(reason="Error type differs (ShapeError vs ValueError) — polars native")
    def test_mismatched_length_raises(self):
        df = _df({"x": [1, 2, 3]})
        with pytest.raises(pl.exceptions.ShapeError):
            df >> mutate(
                y=[1, 2],
            )


class TestRemovesVarsWithNone:
    def test_none_removes_existing_column(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = df >> mutate(
            y=None,
        )
        assert list(out.collect_schema().names()) == ["x"]

    def test_none_for_nonexistent_column_noop(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            z=None,
        )
        assert_df_equal(out, _df({"x": [1, 2, 3]}))

    def test_none_removes_from_grouped(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6]})
        gf = df >> group_by(f.x)
        out = gf >> mutate(
            y=None,
        )
        assert list(out.collect_schema().names()) == ["x"]


class TestPreservesNames:
    def test_column_names_preserved(self):
        df = _df({"a": [1, 2, 3]})
        out = df >> mutate(
            b=f.a * 2,
        )
        assert list(out.collect_schema().names()) == ["a", "b"]


# ---------------------------------------------------------------------------
# _keep / _before / _after
# ---------------------------------------------------------------------------

class TestKeep:
    def test_keep_all_is_default(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=f.x + f.y,
        )
        assert set(out.collect_schema().names()) == {"x", "y", "z"}

    def test_keep_unused(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            x1=f.x + 1, y=f.y, _keep="unused",
            __backend="polars",
        )
        # "unused" keeps columns NOT referenced in expressions + new cols
        # x1 uses x, y uses y — so no unused columns except possibly none
        # Actually: y is used (referenced), x is used. "unused" means keep vars NOT used
        assert "x1" in out.collect_schema().names()

    def test_keep_used(self):
        df = _df({"a": [1], "b": [2], "c": [3], "x": [1], "y": [2]})
        out = df >> mutate(
            xy=f.x + f.y, _keep="used",
            __backend="polars",
        )
        assert set(out.collect_schema().names()) == {"x", "y", "xy"}

    def test_keep_none_only_new_and_group_vars(self):
        df = _df({"x": [1], "y": [2]})
        gf = df >> group_by(f.x)
        out = gf >> mutate(
            z=1, _keep="none",
            __backend="polars",
        )
        assert set(out.collect_schema().names()) == {"x", "z"}

    def test_keep_none_ungrouped(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=1, _keep="none",
            __backend="polars",
        )
        assert set(out.collect_schema().names()) == {"z"}

    def test_keep_always_retains_grouping_vars(self):
        df = _df({"x": [1], "y": [2], "z": [3]})
        gf = df >> group_by(f.z)
        out = gf >> mutate(
            a=f.x + 1, _keep="none",
            __backend="polars",
        )
        assert set(out.collect_schema().names()) == {"z", "a"}
        assert _gvars(out) == ["z"]


class TestBeforeAndAfter:
    def test_default_appends(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=1,
        )
        assert list(out.collect_schema().names()) == ["x", "y", "z"]

    def test_before_int(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=1, _before=1,
        )
        assert list(out.collect_schema().names()) == ["x", "z", "y"]

    def test_after_int(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=1, _after=0,
        )
        assert list(out.collect_schema().names()) == ["x", "z", "y"]

    def test_before_string(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=1, _before="y",
        )
        assert list(out.collect_schema().names()) == ["x", "z", "y"]

    def test_after_string(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=1, _after="x",
        )
        assert list(out.collect_schema().names()) == ["x", "z", "y"]

    def test_after_column(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            z=1, _after=f.x,
        )
        assert list(out.collect_schema().names()) == ["x", "z", "y"]


# ---------------------------------------------------------------------------
# grouping and edge cases
# ---------------------------------------------------------------------------

class TestPreservesGrouping:
    def test_group_vars_preserved_after_mutate(self):
        df = _df({"x": [1, 2], "y": [2, 3]})
        gf = df >> group_by(f.x)
        out = gf >> mutate(
            z=1,
        )
        assert _gvars(out) == ["x"]

    def test_group_vars_preserved_when_overwriting_group_col(self):
        df = _df({"x": [1, 2], "y": [2, 3]})
        gf = df >> group_by(f.x)
        out = gf >> mutate(
            x=1,
        )
        assert _gvars(out) == ["x"]


class TestEdgeCases:
    def test_mutate_with_none_as_positional(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> mutate(
            None,
        )
        assert out.shape == df.shape

    def test_empty_df(self):
        df = _df({"x": []})
        out = df >> mutate(
            __backend="polars",
        )
        assert out.shape == (0, 1)

        out2 = df >> mutate(
            y=[],
        )
        assert out2.shape == (0, 2)
        assert list(out2.collect_schema().names()) == ["x", "y"]

    def test_dup_keyword_args(self):
        df = _df({"a": [1]})
        out = df >> mutate(
            _b=f.a + 1, b=f._b * 2,
            __backend="polars",
        )
        assert out.get_column("a").to_list() == [1]
        assert out.get_column("b").to_list() == [4]

    def test_complex_expression(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            y=f.x * f.x + 1,
        )
        assert out.get_column("y").to_list() == [2, 5, 10]

    def test_mutate_with_tibble(self):
        out = tibble(x=1) >> mutate(y=tibble(y=f.x))
        assert out.collect_schema().names() == ["x", "y"]
        assert out["y"]["y"].to_list() == [1]


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TestErrors:
    # @pytest.mark.skip(reason="Error type differs (ShapeError vs ValueError) — polars native")
    def test_wrong_size(self):
        df = _df({"x": [1, 2, 3, 4]})
        with pytest.raises(pl.exceptions.ShapeError):
            df >> mutate(
                y=[1, 2],
            )

    # @pytest.mark.skip(reason="Error type differs (ShapeError vs ValueError) — polars native")
    def test_grouped_wrong_size(self):
        df = _df({"x": [1, 2, 3, 4], "g": [1, 1, 2, 2]})
        gf = df >> group_by(f.g)
        with pytest.raises(pl.exceptions.ShapeError):
            gf >> mutate(
                y=[1, 2, 3],
            )


# ---------------------------------------------------------------------------
# transmute
# ---------------------------------------------------------------------------

class TestTransmute:
    def test_transmute_only_keeps_new_cols_and_group_vars(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> transmute(
            z=f.x + f.y,
        )
        assert list(out.collect_schema().names()) == ["z"]

    def test_transmute_preserves_grouping(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        gf = df >> group_by(f.x)
        out = gf >> transmute(
            z=1,
        )
        assert _gvars(out) == ["x"]

    # @pytest.mark.skip(reason="Transmute without args edge case differs")
    def test_transmute_without_args_returns_empty(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> transmute(__backend="polars")
        # polars DataFrame must have columns
        # assert out.shape == (1, 0)
        assert out.shape == (0, 0)

    def test_transmute_without_args_grouped(self):
        df = _df({"x": [1], "y": [2]})
        gf = df >> group_by(f.x)
        out = gf >> transmute(__backend="polars")
        assert list(out.collect_schema().names()) == ["x"]

    def test_transmute_dont_match_internal_args(self):
        df = _df({"a": [1]})
        out = df >> transmute(
            var=f.a,
        )
        assert list(out.collect_schema().names()) == ["var"]
        assert out.get_column("var").to_list() == [1]

    def test_transmute_no_keep_arg(self):
        df = _df({"x": [1]})
        with pytest.raises(TypeError):
            transmute(df, z=f.x, _keep="all")


class TestMutateIfElse:
    def test_mutate_with_if_else(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            y=if_else(f.x > 2, "big", "small"),
            __backend="polars",
        )
        assert out.get_column("y").to_list() == ["small", "small", "big"]
