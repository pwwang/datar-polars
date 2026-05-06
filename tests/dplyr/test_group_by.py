"""Tests for group_by, ungroup, rowwise — ported from tidyverse test-group-by.r

https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-group-by.r
"""

import pytest
import polars as pl
from datar import f
from datar.base import factor, rnorm
from datar.dplyr import group_by, ungroup, rowwise, group_vars, group_by_drop_default, group_keys, group_rows, mutate
from datar.tibble import tibble
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


def _gvars(df) -> list:
    return group_vars(df)


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------

class TestGroupBySingleColumn:
    def test_group_by_single(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = df >> group_by(f.x)
        assert _gvars(out) == ["x"]

    def test_group_by_string(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> group_by("x")
        assert _gvars(out) == ["x"]

    def test_group_by_preserves_data(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = df >> group_by(f.x)
        assert out.shape == df.shape
        assert list(out.collect_schema().names()) == list(df.collect_schema().names())

    def test_does_not_affect_input_data(self):
        df = _df({"x": [1, 2, 3]})
        _ = df >> group_by(f.x)
        assert df.get_column("x").to_list() == [1, 2, 3]


class TestGroupByMultipleColumns:
    def test_group_by_multiple(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})
        out = df >> group_by(
            f.x, f.y,
        )
        assert _gvars(out) == ["x", "y"]

    def test_group_by_multiple_strings(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        out = df >> group_by(
            "x", "y",
        )
        assert _gvars(out) == ["x", "y"]


class TestGroupByAdd:
    def test_group_by_add_appends(self):
        df = _df({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        gf = df >> group_by(f.x)
        out = gf >> group_by(
            f.y, _add=True,
        )
        assert _gvars(out) == ["x", "y"]

    def test_group_by_add_no_duplicates(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        gf = df >> group_by(f.x)
        out = gf >> group_by(
            f.x, _add=True,
        )
        assert _gvars(out) == ["x"]

    def test_group_by_no_add_replaces(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        gf = df >> group_by(f.x)
        out = gf >> group_by(f.y)
        assert _gvars(out) == ["y"]


# ---------------------------------------------------------------------------
# group_vars
# ---------------------------------------------------------------------------

class TestGroupVars:
    def test_group_vars_returns_correct(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        gf = df >> group_by(f.y)
        assert _gvars(gf) == ["y"]

    def test_group_vars_ungrouped_returns_empty(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        assert _gvars(df) == []


# ---------------------------------------------------------------------------
# ungroup
# ---------------------------------------------------------------------------

class TestUngroup:
    def test_ungroup_clears_all(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        gf = df >> group_by(f.x)
        out = gf >> ungroup(__backend="polars")
        assert _gvars(out) == []

    def test_ungroup_some_columns(self):
        df = _df({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
        gf = df >> group_by(
            f.x, f.y,
        )
        out = gf >> ungroup(
            f.x,
        )
        assert _gvars(out) == ["y"]

    def test_ungroup_string_col(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        gf = df >> group_by(f.x)
        out = gf >> ungroup("x")
        assert _gvars(out) == []

    def test_ungroup_nonexistent_col_error(self):
        df = _df({"x": [1, 2]})
        gf = df >> group_by(f.x)
        with pytest.raises(KeyError):
            gf >> ungroup(f.z)


# ---------------------------------------------------------------------------
# rowwise
# ---------------------------------------------------------------------------

class TestRowwise:
    def test_rowwise_sets_flag(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> rowwise(__backend="polars")
        assert out._datar.get("rowwise") is True
        assert _gvars(out) == []

    def test_rowwise_with_cols_sets_groups(self):
        df = _df({"x": [1, 2, 3], "y": [4, 5, 6]})
        out = df >> rowwise(f.x)
        assert out._datar.get("rowwise") is True
        assert _gvars(out) == ["x"]

    def test_rowwise_over_grouped_df(self):
        df = _df({"g": [1, 2], "x": [1, 2]})
        gf = df >> group_by(f.g)
        out = gf >> rowwise(__backend="polars")
        assert out._datar.get("rowwise") is True

    def test_group_by_after_rowwise_replaces(self):
        df = _df({"x": [1, 2], "y": [3, 4]})
        rf = df >> rowwise(f.x)
        out = rf >> group_by(f.y)
        assert _gvars(out) == ["y"]

    def test_rowwise(self):
        params = tibble(
            sim=[1, 2, 3],
            n=[1, 2, 3],
            mean=[1, 2, 1],
            sd=[1, 4, 2]
        )

        out = params >> rowwise(f.sim) >> mutate(z=rnorm(f.n, f.mean, f.sd))
        assert out.shape == (3, 5)


# ---------------------------------------------------------------------------
# group_by_drop_default
# ---------------------------------------------------------------------------

class TestGroupByDropDefault:
    def test_drop_default_true(self):
        df = _df({"x": [1, 2]})
        gf = df >> group_by(f.x)
        assert group_by_drop_default(
            gf,
        )

    def test_drop_default_false(self):
        df = _df({"x": factor(['a', 'b'], levels=['a', 'b', 'c'])})
        gf = df >> group_by(
            f.x, _drop=False,
        )
        keys = group_keys(gf)
        assert keys.shape == (3, 1)

        rows = group_rows(gf)
        assert len(rows) == 3
        assert rows[0] == [0]
        assert rows[1] == [1]
        assert rows[2] == []


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

class TestGroupByEmpty:
    def test_group_by_empty_df(self):
        df = _df({"g": [], "x": []})
        out = df >> group_by(f.g)
        assert _gvars(out) == ["g"]
        assert out.shape == (0, 2)

    def test_group_by_no_args(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> group_by(__backend="polars")
        assert _gvars(out) == []


class TestGroupByErrors:
    def test_group_by_nonexistent_col(self):
        df = _df({"x": [1]})
        with pytest.raises(KeyError):
            df >> group_by(f.z)

    def test_ungroup_no_groups_is_noop(self):
        df = _df({"x": [1, 2]})
        out = df >> ungroup(__backend="polars")
        assert _gvars(out) == []
