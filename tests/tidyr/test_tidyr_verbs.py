import pytest
"""Tests for tidyr verbs — polars backend.

Covers: drop_na, replace_na, fill, pivot_longer, pivot_wider,
separate, unite, unnest.
"""

import polars as pl
from datar import f
from datar.data import relig_income, billboard, warpbreaks
from datar.base import NA, NULL, c, TRUE, mean
from datar.dplyr import starts_with
from datar.tidyr import (
    drop_na,
    replace_na,
    fill,
    pivot_longer,
    pivot_wider,
    separate,
    separate_rows,
    unite,
    unnest,
    uncount,
    expand_grid,
)
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal, assert_iterable_equal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

BA = {}


def _df(data: dict) -> pl.DataFrame:
    "Create a polars DataFrame as Tibble with _datar metadata."
    # Convert NA (float NaN) to None so Polars infers proper nulls
    # instead of converting NaN to the literal string "NaN"
    import math

    cleaned = {}
    for k, v in data.items():
        cleaned[k] = [None if (isinstance(x, float) and math.isnan(x)) else x for x in v]
    return as_tibble(pl.DataFrame(cleaned, strict=False))


# ===========================================================================
# drop_na
# ===========================================================================

class TestDropNA:
    "Tests for drop_na()"

    def test_drop_na_all_columns_default(self):
        "Rows with any null are dropped."
        df = _df({"x": [1, None, 3], "y": ["a", "b", None]})
        out = df >> drop_na(**BA)
        assert out.shape == (1, 2)
        assert out.get_column("x").to_list() == [1]

    def test_drop_na_specific_column(self):
        "Only check specified columns for nulls."
        df = _df({"x": [1, None, 3], "y": ["a", "b", "c"]})
        out = df >> drop_na("x", **BA)
        assert out.shape == (2, 2)
        assert out.get_column("x").to_list() == [1, 3]

    def test_drop_na_how_all(self):
        "Drop rows where all are NA."
        df = _df({
            "x": [1, None, None, 4],
            "y": [None, 2, None, None],
        })
        out = df >> drop_na(_how="all", **BA)
        # Only row 2 has all None; row 1 has partial None, kept.
        assert out.shape == (3, 2)

    def test_drop_na_no_nulls(self):
        "Data with no nulls is unchanged."
        df = _df({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        out = df >> drop_na(**BA)
        assert out.shape == (3, 2)


# ===========================================================================
# replace_na
# ===========================================================================

class TestReplaceNA:
    "Tests for replace_na()"

    def test_replace_na_scalar(self):
        "Replace all nulls with scalar."
        df = _df({"x": [1, None, 3], "y": [None, "b", None]})
        out = df >> replace_na(0, **BA)
        assert out.get_column("x").to_list() == [1, 0, 3]

    def test_replace_na_dict_per_column(self):
        "Replace nulls with per-column values."
        df = _df({"x": [1, None, 3], "y": ["a", None, "c"]})
        out = df >> replace_na({"x": 0, "y": "z"}, **BA)
        assert out.get_column("x").to_list() == [1, 0, 3]
        assert out.get_column("y").to_list() == ["a", "z", "c"]

    def test_replace_na_no_nulls(self):
        "Data with no nulls unchanged."
        df = _df({"x": [1, 2, 3]})
        out = df >> replace_na(0, **BA)
        assert out.get_column("x").to_list() == [1, 2, 3]

    def test_replace_na_str(self):
        "Replace NA in string column."
        df = _df({"x": ["a", NA, "c"]})
        out = df >> replace_na("missing", **BA)
        assert out.get_column("x").to_list() == ["a", "missing", "c"]

    def test_replace_null_str(self):
        "Replace NA in string column."
        df = _df({"x": ["a", NULL, "c"]})
        out = df >> replace_na("missing", **BA)
        assert out.get_column("x").to_list() == ["a", "missing", "c"]

        out = df.x >> replace_na("missing", **BA)
        assert out.to_list() == ["a", "missing", "c"]


# ===========================================================================
# fill
# ===========================================================================

class TestFill:
    "Tests for fill()"

    def test_fill_down(self):
        "Fill missing values downward."
        df = _df({"x": [1, None, None, 4, None]})
        out = df >> fill(f.x, _direction="down", **BA)
        assert out.get_column("x").to_list() == [1, 1, 1, 4, 4]

    def test_fill_up(self):
        "Fill missing values upward."
        df = _df({"x": [None, None, 3, None, 5]})
        out = df >> fill(f.x, _direction="up", **BA)
        assert out.get_column("x").to_list() == [3, 3, 3, 5, 5]

    def test_fill_downup(self):
        "Fill down, then up."
        df = _df({"x": [None, 2, None, None, 5]})
        out = df >> fill(f.x, _direction="downup", **BA)
        assert out.get_column("x").to_list() == [2, 2, 2, 2, 5]

    def test_fill_updown(self):
        "Fill up, then down."
        df = _df({"x": [1, None, None, 4, None]})
        out = df >> fill(f.x, _direction="updown", **BA)
        assert out.get_column("x").to_list() == [1, 4, 4, 4, 4]

    def test_fill_all_columns(self):
        "Fill all columns when none specified."
        df = _df({"x": [1, None], "y": [None, 2]})
        out = df >> fill(_direction="down", **BA)
        assert out.get_column("x").to_list() == [1, 1]
        assert out.get_column("y").to_list() == [None, 2]

    def test_fill_works_with_NAs(self):
        "Fill should work with NA values from replace_na()."
        df = _df({"x": [1, NA, 3]})
        out = df >> replace_na(0, **BA) >> fill(f.x, _direction="down", **BA)
        assert out.get_column("x").to_list() == [1, 0, 3]

# ===========================================================================
# pivot_longer
# ===========================================================================

class TestPivotLonger:
    "Tests for pivot_longer()"

    def test_basic_pivot_longer(self):
        "Wide to long with default names."
        df = _df({"id": [1, 2], "x": [10, 20], "y": [30, 40]})
        out = df >> pivot_longer(["x", "y"], **BA)
        assert out.shape == (4, 3)
        assert sorted(out.get_column("name").unique().to_list()) == ["x", "y"]

    def test_pivot_longer_custom_names(self):
        "Custom names_to and values_to."
        df = _df({"id": [1, 2], "a": [1, 2], "b": [3, 4]})
        out = df >> pivot_longer(
            ["a", "b"],
            names_to="var",
            values_to="val",
            **BA,
        )
        assert "var" in out.collect_schema().names()
        assert "val" in out.collect_schema().names()
        assert out.get_column("var").to_list() == ["a", "b", "a", "b"]

    def test_pivot_longer_values_drop_na(self):
        "Drop rows where value is null."
        df = _df({"id": [1], "x": [None], "y": [2]})
        out = df >> pivot_longer(["x", "y"], values_drop_na=True, **BA)
        assert out.shape == (1, 3)
        assert out.get_column("value").to_list() == [2]

    def test_pivot_longer_relig_income(self):
        out = relig_income >> \
            pivot_longer(~f.religion, names_to="income", values_to="count")

        assert out.collect_schema().names() == ["religion", "income", "count"]
        assert out.shape == (180, 3)

    def test_pivot_longer_billboard(self):
        out = billboard >> \
            pivot_longer(
            cols = starts_with("wk"),
            names_to = "week",
            names_prefix = "wk",
            values_to = "rank",
            values_drop_na = TRUE
        )
        assert out.shape == (5307, 5)

# ===========================================================================
# pivot_wider
# ===========================================================================

class TestPivotWider:
    "Tests for pivot_wider()"

    def test_basic_pivot_wider(self):
        "Long to wide."
        df = _df({
            "id": [1, 1, 2, 2],
            "name": ["x", "y", "x", "y"],
            "value": [10, 20, 30, 40],
        })
        out = df >> pivot_wider(
            id_cols="id",
            names_from="name",
            values_from="value",
            **BA,
        )
        assert "id" in out.collect_schema().names()
        assert out.shape == (2, 3)

    def test_pivot_wider_names_prefix(self):
        "Add prefix to new column names."
        df = _df({
            "id": [1, 2],
            "name": ["x", "x"],
            "value": [10, 20],
        })
        out = df >> pivot_wider(
            id_cols="id",
            names_from="name",
            values_from="value",
            names_prefix="p_",
            **BA,
        )
        assert "p_x" in out.collect_schema().names()

    def test_pivot_wider_values_fill(self):
        "Fill missing values in wide form."
        df = _df({
            "id": [1, 1],
            "name": ["x", "y"],
            "value": [10, 20],
        })
        out = df >> pivot_wider(
            id_cols="id",
            names_from="name",
            values_from="value",
            values_fill=0,
            **BA,
        )
        # All present, no fill needed, but should work
        assert out.shape == (1, 3)

    def test_pivot_wider_warpbreaks(self):
        out = warpbreaks >> pivot_wider(
            names_from=f.wool,
            values_from=f.breaks,
            values_fn = mean,
        )
        assert out.shape == (3, 3)
        assert set(out["tension"]) == {"H", "L", "M"}


# ===========================================================================
# separate
# ===========================================================================

class TestSeparate:
    "Tests for separate()"

    def test_separate_basic(self):
        "Split a column on default separator."
        df = _df({"x": ["a_b", "c_d", "e_f"]})
        out = df >> separate("x", into=["first", "second"], **BA)
        assert "first" in out.collect_schema().names()
        assert "second" in out.collect_schema().names()
        assert out.get_column("first").to_list() == ["a", "c", "e"]

    def test_separate_custom_sep(self):
        "Split with custom separator."
        df = _df({"x": ["a-b", "c-d"]})
        out = df >> separate("x", into=["left", "right"], sep="-", **BA)
        assert out.get_column("left").to_list() == ["a", "c"]

    def test_separate_keep_original(self):
        "Keep original column with remove=False."
        df = _df({"x": ["a_b", "c_d"]})
        out = df >> separate("x", into=["first", "second"], remove=False, **BA)
        assert "x" in out.collect_schema().names()
        assert "first" in out.collect_schema().names()

    def test_separate_single_piece(self):
        "Single output column."
        df = _df({"x": ["a_b", "c_d"]})
        out = df >> separate("x", into="first", **BA)
        assert "first" in out.collect_schema().names()
        assert out.get_column("first").to_list() == ["a", "c"]

    def test_separate_ignore_column(self):
        df = _df({"x": [NA, "x.y", "x.z", "y.z"]})
        out = df >> separate(f.x, c(NA, "B"))
        assert out.get_column("B").to_list() == [None, "y", "z", "z"]

    def test_separate_ignore_column_with_na(self):
        df = _df({"x": ["x", "x y", "x y z", NA]})
        out = df >> separate("x", into=["a", "b"], **BA)
        assert out.get_column("a").to_list() == ["x", "x", "x", None]
        assert out.get_column("b").to_list() == [None, "y", "y", None]

        out = df >> separate(f.x, c("a", "b"), extra="drop", fill="right")
        assert out.get_column("a").to_list() == ["x", "x", "x", None]
        assert out.get_column("b").to_list() == [None, "y", "y", None]

    def test_separate_merge(self):
        df = _df({"x": ["x", "x y", "x y z", NA]})
        out = df >> separate(f.x, c("a", "b"), extra="merge", fill="left")
        assert out.get_column("a").to_list() == [None, "x", "x", None]
        assert out.get_column("b").to_list() == ["x", "y", "y z", None]

    def test_separate_convert(self):
        df = _df({"x": ["x:1", "x:2", "y:4", "z", NA]})
        out = df >> separate("x", into=["a", "b"], sep=":", convert={"b": float}, **BA)
        assert out.get_column("a").to_list() == ["x", "x", "y", "z", None]
        assert out.get_column("b").to_list() == [1.0, 2.0, 4.0, None, None]


class TestSeparateRows:

    def test_separate_rows_basic(self):
        "Separate rows on default separator."
        df = _df(
            {
                "x": [1,2,3],
                "y": ["a", "d,e,f", "g,h"],
                "z": ["1", "2,3,4", "5,6"],
            }
        )
        out = df >> separate_rows(f.y, f.z, convert={'z': int}, **BA)
        assert out.shape == (6, 3)
        assert out.get_column("y").to_list() == ["a", "d", "e", "f", "g", "h"]
        assert out.get_column("z").to_list() == [1, 2, 3, 4, 5, 6]


# ===========================================================================
# unite
# ===========================================================================

class TestUnite:
    "Tests for unite()"

    def test_unite_basic(self):
        "Combine two columns into one."
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> unite("ab", "a", "b", **BA)
        assert "ab" in out.collect_schema().names()
        assert out.get_column("ab").to_list() == ["1_3", "2_4"]

    def test_unite_custom_sep(self):
        "Custom separator."
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> unite("ab", "a", "b", sep="-", **BA)
        assert out.get_column("ab").to_list() == ["1-3", "2-4"]

    def test_unite_keep_original(self):
        "Keep original columns with remove=False."
        df = _df({"a": [1, 2], "b": [3, 4]})
        out = df >> unite("ab", "a", "b", remove=False, **BA)
        assert "a" in out.collect_schema().names()
        assert "b" in out.collect_schema().names()
        assert "ab" in out.collect_schema().names()

    def test_unite_na_rm(self):
        "Remove NA values from concatenation."
        df = _df({"a": [1, None], "b": [3, 4]})
        out = df >> unite("ab", "a", "b", na_rm=True, **BA)
        vals = out.get_column("ab").to_list()
        assert vals[0] in ("1_3", "3_1")  # order may vary
        assert vals[1] == "4" or vals[1] == "_4" or vals[1] == "4_"

    def test_unite_removal_false(self):
        df = expand_grid(x=c('a', NA), y=c('b', NA))
        out = df >> unite("z", [0, 1], na_rm=True, remove=False)
        assert out.collect_schema().names() == ["z", "x", "y"]
        assert out.get_column("z").to_list() == ["a_b", "a", "b", ""]
        assert out.get_column("x").to_list() == ["a", "a", None, None]
        assert out.get_column("y").to_list() == ["b", None, "b", None]


# ===========================================================================
# unnest
# ===========================================================================

class TestUnnest:
    "Tests for unnest()"

    def test_unnest_basic(self):
        df = _df(
            {
                "a": [c("a", "b"), "c"],
                "b": [[1, 2], 3],
                "c": [11, 22],
            }
        )
        out = df >> unnest(c(f.a, f.b))
        assert out.collect_schema().names() == ["a", "b", "c"]
        assert out.shape == (3, 3)

    def test_unnest_list_column(self):
        "Unnest a list column."
        df = _df({
            "id": [1, 2],
            "vals": [[10, 20], [30]],
        })
        out = df >> unnest("vals", **BA)
        assert out.shape == (3, 2)
        assert sorted(out.get_column("vals").to_list()) == [10, 20, 30]

    def test_unnest_multiple_columns(self):
        "Unnest multiple list columns."
        df = _df({
            "id": [1, 2],
            "a": [[1, 2], [3]],
            "b": [[4, 5], [6]],
        })
        out = df >> unnest("a", "b", **BA)
        assert out.shape == (3, 3)

    def test_unnest_keep_empty(self):
        "Keep rows with empty lists."
        df = _df({
            "id": [1, 2],
            "vals": [[10], []],
        })
        out = df >> unnest("vals", keep_empty=True, **BA)
        assert out.shape == (2, 2)


class TestUncount:

    def test_uncount_basic(self):
        "Repeat rows according to count."
        df = _df({"id": [1, 2], "n": [2, 3]})
        out = df >> uncount("n", **BA)
        assert out.shape == (5, 1)
        assert out.get_column("id").to_list() == [1, 1, 2, 2, 2]

    def test_uncount_zero(self):
        "Rows with zero count are dropped."
        df = _df({"id": [1, 2], "n": [0, 3]})
        out = df >> uncount("n", **BA)
        assert out.shape == (3, 1)
        assert out.get_column("id").to_list() == [2, 2, 2]

    def test_uncount_no_count_column(self):
        "Error if count column missing."
        df = _df({"id": [1, 2]})
        with pytest.raises(ValueError, match="weights"):
            df >> uncount("n", **BA)

    def test_uncount_non_integer(self):
        "Error if count column not integer."
        df = _df({"id": [1, 2], "n": [1.5, 3.0]})
        with pytest.raises(ValueError, match="integer"):
            df >> uncount("n", **BA)

    def test_uncount_with_id(self):
        "Uncount with id column should repeat id correctly."
        df = _df({"x": ["a", "b"], "n": [1, 2]})
        out = df >> uncount("n", _id="id", **BA)
        assert out.get_column("x").to_list() == ["a", "b", "b"]
        assert out.get_column("id").to_list() == [0, 1, 1]

    def test_uncount_with_expr(self):
        "Uncount with expression for weights."
        df = _df({"x": ["a", "b"], "n": [1, 2]})
        out = df >> uncount(f.n + 1, _id="id", **BA)
        assert out.get_column("x").to_list() == ["a", "a", "b", "b", "b"]
        assert out.get_column("id").to_list() == [0, 0, 1, 1, 1]

        out = df >> uncount(2//f.n, _id="id", **BA)
        assert out.get_column("x").to_list() == ["a", "a", "b"]
        assert out.get_column("id").to_list() == [0, 0, 1]

class TestExpandGrid:

    def test_expand_grid_basic(self):
        "Expand two vectors into a grid."
        df = expand_grid(x=[1, 2], y=["a", "b"], **BA)
        assert df.shape == (4, 2)
        assert sorted(df.get_column("x").unique().to_list()) == [1, 2]
        assert sorted(df.get_column("y").unique().to_list()) == ["a", "b"]

    def test_expand_grid_with_missing(self):
        df = expand_grid(x=c("a", NA), y=c("b", NA))
        assert df.shape == (4, 2)
        assert set(df.get_column("x").to_list()) == {"a", None}
        assert set(df.get_column("y").to_list()) == {"b", None}
