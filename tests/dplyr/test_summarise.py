"""Tests for summarise and reframe verbs — ported from tidyverse test-summarise.R

https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-summarise.R
"""

import pytest
import polars as pl
from datar import f
from datar.data import mtcars
from datar.base import sum_, mean, quantile, c, intersect, sd
from datar.dplyr import summarise, reframe, group_by, group_vars, ungroup, across
from datar.dplyr import mutate
from datar.tibble import tibble
from datar_polars.tibble import as_tibble
from pandas.core.groupby import SeriesGroupBy
from pipda import register_func

from ..conftest import assert_df_equal, assert_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


def _gvars(df) -> list:
    return group_vars(df)


# ---------------------------------------------------------------------------
# summarise
# ---------------------------------------------------------------------------

class TestSummariseUngrouped:
    def test_summarise_single_expression(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(
            y=sum_(f.x),
        )
        assert out.shape == (1, 1)
        assert list(out.collect_schema().names()) == ["y"]
        assert out.get_column("y").to_list() == [6]

    def test_summarise_literal_value(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(
            val=42,
        )
        assert out.get_column("val").to_list() == [42]

    def test_summarise_multi_expression(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(
            s=sum_(f.x), c=10,
        )
        assert out.shape == (1, 2)
        assert out.get_column("s").to_list() == [6]
        assert out.get_column("c").to_list() == [10]

    def test_summarise_no_args_returns_empty(self):
        df = _df({"x": [1, 2]})
        out = df >> summarise(__backend="polars")
        assert out.shape == (1, 0)


class TestSummariseGrouped:
    def test_summarise_one_per_group(self):
        df = _df({"g": [1, 1, 2, 2], "x": [10, 20, 30, 40]})
        gf = df >> group_by(f.g)
        out = gf >> summarise(
            s=sum_(f.x),
        )
        # Each group gets its own row
        assert out.shape == (2, 2)
        assert list(out.collect_schema().names()) == ["g", "s"]
        gv = out.get_column("g").to_list()
        sv = out.get_column("s").to_list()
        assert sorted(zip(gv, sv)) == [(1, 30), (2, 70)]

    def test_summarise_grouped_literal(self):
        df = _df({"g": [1, 2], "x": [10, 20]})
        gf = df >> group_by(f.g)
        out = gf >> summarise(
            val=1,
        )
        assert out.shape == (2, 2)
        assert out.get_column("val").to_list() == [1, 1]

    def test_summarise_peels_grouping_layer(self):
        df = _df({"x": [1, 2, 3, 4], "y": [1, 1, 2, 2]})
        gf = df >> group_by(f.x, f.y)
        out = gf >> summarise(
            s=sum_(f.y),
        )
        # One layer peeled off (2 groups -> 1 group)
        assert _gvars(out) == ["x"]

    def test_summarise_groups_keep(self):
        df = _df({"x": [1, 2], "y": [1, 2]})
        gf = df >> group_by(f.x, f.y)
        out = gf >> summarise(
            z=1, _groups="keep",
        )
        assert _gvars(out) == ["x", "y"]

    def test_summarise_groups_drop(self):
        df = _df({"x": [1, 2], "y": [1, 2]})
        gf = df >> group_by(f.x, f.y)
        out = gf >> summarise(
            z=1, _groups="drop",
        )
        assert _gvars(out) == []

    def test_summarise_with_quantile(self):
        out = mtcars >> \
            group_by(f.cyl) >> \
            summarise(qs=quantile(f.disp, c(0.25, 0.75)), prob=c(0.25, 0.75))

        assert out.collect_schema().names() == ["cyl", "qs", "prob"]
        assert out.shape == (6, 3)
        assert set(out.get_column("cyl").to_list()) == {4, 6, 8}
        assert set(out.get_column("prob").to_list()) == {0.25, 0.75}

    def test_summarise_reuses_kwargs(self):
        out = mtcars >> \
            group_by(f.cyl) >> \
            summarise(_disp_m2=mean(f.disp), disp_m2=f._disp_m2 * 2)

        # Temporary variables (beginning with _) are not included in output, but can be reused in later expressions.
        assert out.collect_schema().names() == ["cyl", "disp_m2"]
        assert out.shape == (3, 2)

    def test_summarise_reuses_original_column_name(self):
        out = mtcars >> \
            group_by(f.cyl) >> \
            summarise(disp=mean(f.disp), sd=sd(f.disp))

        assert out.collect_schema().names() == ["cyl", "disp", "sd"]
        assert out.shape == (3, 3)

# ---------------------------------------------------------------------------
# summarise edge cases
# ---------------------------------------------------------------------------

class TestSummariseEmptyDF:
    def test_summarise_empty_df(self):
        df = _df({"x": []})
        out = df >> summarise(
            y=1,
        )
        assert out.shape == (1, 1)
        assert out.get_column("y").to_list() == [1]

    def test_summarise_empty_grouped(self):
        df = _df({"g": [], "x": []})
        gf = df >> group_by(f.g)
        out = gf >> summarise(
            y=1,
        )
        assert out.shape == (0, 2)
        assert list(out.collect_schema().names()) == ["g", "y"]


class TestSummariseSkip:
    def test_summarise_with_mean(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(
            avg=f.x.mean(),
        )
        assert out.get_column("avg").to_list() == [2.0]

    def test_summarise_with_sum(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(
            total=f.x.sum(),
        )
        assert out.get_column("total").to_list() == [6]


# ---------------------------------------------------------------------------
# reframe
# ---------------------------------------------------------------------------

class TestReframe:
    def test_reframe_ungrouped(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> reframe(
            y=sum_(f.x),
        )
        assert out.shape == (1, 1)
        assert out.get_column("y").to_list() == [6]

    def test_reframe_grouped(self):
        df = _df({"g": [1, 1, 2, 2], "x": [10, 20, 30, 40]})
        gf = df >> group_by(f.g)
        out = gf >> reframe(
            total=sum_(f.x),
        )
        assert out.shape == (2, 2)
        gv = out.get_column("g").to_list()
        tv = out.get_column("total").to_list()
        assert sorted(zip(gv, tv)) == [(1, 30), (2, 70)]

    def test_reframe_grouped_literal(self):
        df = _df({"g": [1, 2], "x": [1, 2]})
        gf = df >> group_by(f.g)
        out = gf >> reframe(
            v=42,
        )
        assert out.shape == (2, 2)
        assert out.get_column("v").to_list() == [42, 42]

    def test_reframe_with_intersect(self):
        tbl = c('a', 'b', 'd', 'f')
        df = _df({"g": [1, 1, 1, 2, 2, 2, 2], "x": c('e', 'a', 'b', 'c', 'f', 'd', 'a')})
        out = df >> reframe(x = intersect(f.x, tbl))
        assert out.shape == (4, 1)
        assert set(out.get_column("x").to_list()) == {'a', 'b', 'd', 'f'}

    def test_reframe_splices_helper_tibble(self):
        @register_func
        def quantile_df(x, probs=[0.25, 0.5, 0.75]):
            return tibble(
                val=quantile(x, probs, na_rm=True),
                quant=[probs] if isinstance(x, SeriesGroupBy) else probs,
            )

        out = mtcars >> reframe(quantile_df(f.disp))
        assert out.collect_schema().names() == ["val", "quant"]
        assert out.shape == (3, 2)
        assert out.get_column("quant").to_list() == [0.25, 0.5, 0.75]

    def test_reframe_grouped_splices_helper_tibble(self):
        @register_func
        def quantile_df(x, probs=[0.25, 0.5, 0.75]):
            return tibble(
                val=quantile(x, probs, na_rm=True),
                quant=[probs] if isinstance(x, SeriesGroupBy) else probs,
            )

        out = mtcars >> group_by(f.cyl) >> reframe(quantile_df(f.disp))
        assert out.collect_schema().names() == ["cyl", "val", "quant"]
        assert out.shape == (9, 3)
        assert set(out.get_column("quant").to_list()) == {0.25, 0.5, 0.75}

    def test_reframe_across_splices_helper_tibble(self):
        @register_func
        def quantile_df(x, probs=[0.25, 0.5, 0.75]):
            return tibble(
                val=quantile(x, probs, na_rm=True),
                quant=[probs] if isinstance(x, SeriesGroupBy) else probs,
            )

        out = mtcars >> group_by(f.cyl) >> reframe(across(c(f.disp, f.hp), quantile_df))
        assert out.collect_schema().names() == [
            "cyl",
            "disp_val",
            "disp_quant",
            "hp_val",
            "hp_quant",
        ]
        assert out.shape == (9, 5)
        assert set(out.get_column("disp_quant").to_list()) == {0.25, 0.5, 0.75}


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

class TestSummariseErrors:
    def test_summarise_nonexistent_column(self):
        df = _df({"x": [1]})
        with pytest.raises(KeyError):
            df >> summarise(
                z=f.notexist,
            )

    def test_summarise_none_args(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(
            None,
        )
        assert out.shape == (1, 0)
