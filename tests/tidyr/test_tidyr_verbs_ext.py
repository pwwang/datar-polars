"""Additional tidyr tests for the polars backend."""

import pytest
import polars as pl

from datar import f
from datar.base import FALSE, NA, TRUE, c, NULL, factor, rnorm, runif
from datar.data import iris
from datar.dplyr import group_by, group_vars
from datar.tibble import tibble
from datar.tidyr import (
    drop_na, replace_na, fill, separate, unite, chop, unchop, nesting, unpack,
    pivot_longer, pivot_wider, nest, unnest, expand, complete, full_seq, extract
)
from ..conftest import assert_iterable_equal, assert_equal


def _col(df, name):
    """Get column values from a DataFrame or LazyFrame."""
    if hasattr(df, "collect"):
        df = df.collect()
    return list(df[name])


# ---- drop_na ------------------------------------------------------------


def test_drop_na_empty_call():
    df = tibble(x=[1, 2, None], y=["a", None, "b"])
    out = drop_na(df)
    assert out.collect().height == 1


def test_drop_na_specified_vars():
    df = tibble(x=[1, 2, None], y=["a", None, "b"])
    out = drop_na(df, f.x).collect()
    assert list(out["x"]) == [1, 2]
    assert list(out["y"]) == ["a", None]


def test_drop_na_groups_preserved():
    df = tibble(g=["A", "A", "B"], x=[1, 2, None], y=["a", None, "b"])
    gdf = group_by(df, f.g)
    out = drop_na(gdf, f.y)
    assert group_vars(out) == group_vars(gdf)


# ---- replace_na ---------------------------------------------------------


def test_replace_na_empty_call():
    x = [1, None]
    out = replace_na(x)
    assert list(out) == [1, None]


def test_replace_na_values_replaced():
    x = [1, None]
    out = replace_na(x, 0)
    assert list(out) == [1, 0]


def test_replace_na_df():
    df = tibble(x=[1, None])
    out = replace_na(df, {"x": 0}).collect()
    assert list(out["x"]) == [1, 0]


# ---- fill ---------------------------------------------------------------


def test_fill_missings_filled_correctly():
    df = tibble(x=[None, 1, None, 2, None, None])
    out = fill(df, f.x).collect()
    assert list(out["x"]) == [None, 1, 1, 2, 2, 2]


def test_fill_up():
    df = tibble(x=[None, 1, None, 2, None, None])
    out = fill(df, f.x, _direction="up").collect()
    assert list(out["x"]) == [1, 1, 2, 2, None, None]


def test_fill_respects_grouping():
    df = tibble(x=[1, 1, 2], y=[1, None, None])
    out = (df >> group_by(f.x) >> fill(f.y)).collect()
    assert list(out["y"]) == [1, 1, None]


# ---- separate -----------------------------------------------------------


def test_separate_basic():
    df = tibble(x=["a_b", "c_d", "e_f"])
    out = separate(df, f.x, into=["first", "second"], sep="_").collect()
    assert list(out["first"]) == ["a", "c", "e"]
    assert list(out["second"]) == ["b", "d", "f"]


# ---- unite --------------------------------------------------------------


def test_unite_basic():
    df = tibble(a=["a", "b"], b=["c", "d"])
    out = unite(df, "ab", f.a, f.b, sep="_").collect()
    assert list(out["ab"]) == ["a_c", "b_d"]


# ---- pivot_longer -------------------------------------------------------


def test_pivot_longer_basic():
    df = tibble(id=[1, 2], x_a=[1, 2], y_a=[3, 4])
    out = pivot_longer(df, [f.x_a, f.y_a], names_to="name", values_to="value").collect()
    assert list(out["name"]) == ["x_a", "y_a", "x_a", "y_a"]
    assert list(out["value"]) == [1, 3, 2, 4]


# ---- pivot_wider --------------------------------------------------------


def test_pivot_wider_basic():
    df = tibble(
        id=[1, 1, 2, 2],
        name=["x", "y", "x", "y"],
        value=[1, 3, 2, 4],
    )
    out = pivot_wider(df, names_from=f.name, values_from=f.value).collect()
    assert list(out["x"]) == [1, 2]
    assert list(out["y"]) == [3, 4]


# ---- nest / unnest ------------------------------------------------------


def test_nest_basic():
    df = tibble(g=[1, 1, 2, 2], x=[1, 2, 3, 4], y=[5, 6, 7, 8])
    out = nest(df, data=[f.x, f.y]).collect()
    assert "data" in out.collect_schema().names()
    assert "g" in out.collect_schema().names()
    assert out.height == 2


def test_nest():
    df = tibble(x = c(1, 1, 1, 2, 2, 3), y = c[1:6:1], z = c[6:1:-1])
    df = df >> nest(data = [f.y, f.z], __ast_fallback="piping")
    assert list(df["data"][0].struct.unnest()["y"]) == [1, 2, 3]
    assert list(df["data"][0].struct.unnest()["z"]) == [6, 5, 4]
    assert list(df["data"][1].struct.unnest()["y"]) == [4, 5]
    assert list(df["data"][1].struct.unnest()["z"]) == [3, 2]
    assert list(df["data"][2].struct.unnest()["y"]) == [6]
    assert list(df["data"][2].struct.unnest()["z"]) == [1]


def test_nest_iris():
    out = iris >> nest(data=~f.Species, __ast_fallback="piping")
    assert out["data"][0].struct.unnest().shape == (50, 4)
    assert out["data"][1].struct.unnest().shape == (50, 4)
    assert out["data"][2].struct.unnest().shape == (50, 4)


# ---- chop / unchop ------------------------------------------------------


def test_chop():
    df = tibble(x = c(1, 1, 1, 2, 2, 3), y = c[1:6:1], z = c[6:1:-1])
    df = df >> chop(c(f.y, f.z), __ast_fallback="piping")
    assert list(df["y"][0]) == [1, 2, 3]
    assert list(df["z"][0]) == [6, 5, 4]
    assert list(df["y"][1]) == [4, 5]
    assert list(df["z"][1]) == [3, 2]
    assert list(df["y"][2]) == [6]
    assert list(df["z"][2]) == [1]
    assert list(df["x"]) == [1, 2, 3]


def test_unchop():
    df = tibble(x = c[1:4], y = [[], [1], [1,2], [1,2,3]])
    df = df >> unchop(f.y, __ast_fallback="piping")
    assert list(df["x"]) == [2, 3, 3, 4, 4, 4]
    assert list(df["y"]) == [1, 1, 2, 1, 2, 3]


def test_unchop_empty():
    df = tibble(x = c[1:4], y = [[], [1], [1,2], [1,2,3]])
    df = df >> unchop(f.y, keep_empty=True, __ast_fallback="piping")
    assert list(df["x"]) == [1, 2, 3, 3, 4, 4, 4]
    assert list(df["y"]) == [None, 1, 1, 2, 1, 2, 3]


def test_unchop_mixed_types():
    df = tibble(x = 1, y = ["a", [1,2,3]])
    df = df >> unchop(f.y, __ast_fallback="piping")
    assert list(df["x"]) == [1] * 4
    assert list(df["y"]) == ["a", 1, 2, 3]

    with pytest.raises(TypeError):
        df >> unchop(f.y, dtypes=int, __ast_fallback="piping")


def test_unchop_nested_df():
    df = tibble(x=c[1:3], y=[NULL, tibble(x=1), tibble(y=c[1:3])])
    df = df >> unchop(f.y, __ast_fallback="piping")
    assert df.collect_schema().names() == ["x", "y"]
    assert list(df["x"]) == [2, 3, 3, 3]
    assert list(df["y"]["x"]) == [1, None, None, None]
    assert list(df["y"]["y"]) == [None, 1, 2, 3]


# ---- complete ----------------------------------------------------------------


def test_complete():
    df = tibble(x=[1, 2], y=[3, 4])
    out = complete(df, f.x, f.y)
    assert list(out["x"]) == [1, 1, 2, 2]
    assert list(out["y"]) == [3, 4, 3, 4]


def test_complete_with_nesting():
    df = tibble(
        group = c(c[1:2:1], 1),
        item_id = c(c[1:2:1], 2),
        item_name = c("a", "b", "b"),
        value1 = c[1:3:1],
        value2 = c[4:6:1]
    )
    out = df >> complete(
        f.group,
        nesting(f.item_id, f.item_name),
        __ast_fallback="piping",
        __backend="polars",
    )
    assert list(out["group"]) == [1, 1, 2, 2]
    assert list(out["item_id"]) == [1, 2, 1, 2]
    assert list(out["item_name"]) == ["a", "b", "a", "b"]
    assert list(out["value1"]) == [1, 3, None, 2]
    assert list(out["value2"]) == [4, 6, None, 5]


def test_complete_with_nesting_and_fill():
    df = tibble(
        group = c(c[1:2:1], 1),
        item_id = c(c[1:2:1], 2),
        item_name = c("a", "b", "b"),
        value1 = c[1:3:1],
        value2 = c[4:6:1]
    )
    out = df >> complete(
        f.group,
        nesting(f.item_id, f.item_name),
        fill={"value1": 0},
        __ast_fallback="piping",
        __backend="polars",
    )
    assert list(out["group"]) == [1, 1, 2, 2]
    assert list(out["item_id"]) == [1, 2, 1, 2]
    assert list(out["item_name"]) == ["a", "b", "a", "b"]
    assert list(out["value1"]) == [1, 3, 0, 2]
    assert list(out["value2"]) == [4, 6, None, 5]


# ---- expand ----------------------------------------------------------------


def test_expand():
    fruits = tibble(
        type   = c("apple", "orange", "apple", "orange", "orange", "orange"),
        year   = c(2010, 2010, 2012, 2010, 2010, 2012),
        size  =  factor(
            c("XS", "S",  "M", "S", "S", "M"),
            levels = c("XS", "S", "M", "L")
        ),
        weights = rnorm(6)
    )
    out = expand(fruits, f.type, f.size)
    assert out.shape == (8, 2)


def test_expand_nesting_single_col():
    fruits = tibble(
        type   = c("apple", "orange", "apple", "orange", "orange", "orange"),
        year   = c(2010, 2010, 2012, 2010, 2010, 2012),
        size  =  factor(
            c("XS", "S",  "M", "S", "S", "M"),
            levels = c("XS", "S", "M", "L")
        ),
        weights = rnorm(6)
    )
    out = expand(
        fruits,
        nesting(f.type),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert out.shape == (2, 1)
    assert list(out["type"]) == ["apple", "orange"]


def test_expand_nesting_multi_col():
    fruits = tibble(
        type   = c("apple", "orange", "apple", "orange", "orange", "orange"),
        year   = c(2010, 2010, 2012, 2010, 2010, 2012),
        size  =  factor(
            c("XS", "S",  "M", "S", "S", "M"),
            levels = c("XS", "S", "M", "L")
        ),
        weights = rnorm(6)
    )
    out = expand(
        fruits,
        nesting(f.type, f.size),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert out.shape == (4, 2)
    assert list(out["type"]) == ["apple", "orange", "apple", "orange"]
    assert list(out["size"]) == ["XS", "S", "M", "M"]


def test_expand_with_full_seq():
    fruits = tibble(
        type   = c("apple", "orange", "apple", "orange", "orange", "orange"),
        year   = c(2010, 2010, 2012, 2010, 2010, 2012),
        size  =  factor(
            c("XS", "S",  "M", "S", "S", "M"),
            levels = c("XS", "S", "M", "L")
        ),
        weights = rnorm(6)
    )
    out = expand(
        fruits,
        f.type,
        f.size,
        full_seq(f.year, 1),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert out.shape == (24, 3)


# ---- extract ----------------------------------------------------------------


def test_extract():
    df = tibble(x=["a1", "b2", "c3"])
    out = extract(df, f.x, into=["letter", "number"], regex=r"([a-z])(\d)")
    assert list(out["letter"]) == ["a", "b", "c"]
    assert list(out["number"]) == ["1", "2", "3"]


def test_extract_concat_dup_names():
    df = tibble(x="abcd")
    out = extract(df, f.x, into=["a", "b", "a", "b"], regex=r"(.)(.)(.)(.)")
    assert out.collect_schema().names() == ["a", "b"]
    assert list(out["a"]) == ["ac"]
    assert list(out["b"]) == ["bd"]


# ---- unpack ----------------------------------------------------------------


def test_unpack():
    df = tibble(
        x = c[1:3],
        y = tibble(a = c[1:3], b = c[3:1]),
        z = tibble(X = c("a", "b", "c"), Y = runif(3), Z = c(TRUE, FALSE, NA))
    )
    out = df >> unpack(c(1,2))
    assert out.collect_schema().names() == ["x", "a", "b", "X", "Y", "Z"]
    assert list(out["x"]) == [1, 2, 3]
    assert list(out["a"]) == [1, 2, 3]
    assert list(out["b"]) == [3, 2, 1]
    assert list(out["X"]) == ["a", "b", "c"]
    assert list(out["Y"]) == list(df["z"]["Y"])
    assert list(out["Z"]) == [True, False, None]
