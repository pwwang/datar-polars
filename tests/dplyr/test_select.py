# tests grabbed from:
# https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-select.r
import pytest

from polars import DataFrame
from polars.testing import assert_frame_equal
from pipda import register_verb
from datar import f
from datar.data import iris, mtcars
from datar.base import (
    colnames,
    set_colnames,
    # ncol,
    nrow,
    c,
    NA,
    mean,
    rnorm,
)
# from datar.base import rnorm
from datar.tibble import tibble
from datar.dplyr import (
    group_vars,
    group_by,
    ungroup,
    summarise,
    select,
    starts_with,
    matches,
    where,
    any_of,
    all_of,
    ends_with,
    contains,
    num_range,
)
# from datar_pandas.pandas import DataFrame
# from pipda import register_verb

from ..conftest import assert_iterable_equal, assert_equal


def test_preserves_grouping():
    df = tibble(g=[1, 2, 3], x=[3, 2, 1])
    gf = group_by(df, f.g)

    out = select(gf, h=f.g)
    assert_equal(group_vars(out), ["h"])


def test_grouping_variables_preserved_with_a_message(caplog):
    df = tibble(g=[1, 2, 3], x=[3, 2, 1]) >> group_by(f.g)
    res = select(df, f.x)
    assert "Adding missing grouping variables" in caplog.text
    assert res._df.columns() == ["g", "x"]


def test_non_syntactic_grouping_variable_is_preserved():
    df = tibble(**{"a b": [1]}) >> group_by("a b") >> select()
    assert df._df.columns() == ["a b"]
    df = tibble(**{"a b": [1]}) >> group_by(f["a b"]) >> select()
    assert df._df.columns() == ["a b"]


def test_select_doesnot_fail_if_some_names_missing():
    df1 = tibble(x=range(1, 11), y=range(1, 11), z=range(1, 11))
    df2 = df1.copy()
    df2.columns = ["x", "y", ""]

    out1 = select(df1, f.x)
    assert_iterable_equal(out1["x"], range(1, 11))
    out2 = select(df2, f.x)
    assert_iterable_equal(out2["x"], range(1, 11))


# # Special cases -------------------------------------------------
def test_with_no_args_returns_nothing():
    empty = select(iris)
    assert_iterable_equal(empty.shape, (0, 0))

    empty = select(iris, **{})
    assert_iterable_equal(empty.shape, (0, 0))


def test_excluding_all_vars_returns_nothing():
    out = select(iris, ~c[f.Sepal_Length:])
    assert out.shape == (0, 0)

    out = iris >> select(starts_with("x"))
    assert out.shape == (0, 0)

    out = iris >> select(~matches("."))
    assert out.shape == (0, 0)


def test_negating_empty_match_returns_everything():
    df = tibble(x=[1, 2, 3], y=[3, 2, 1])
    out = df >> select(~starts_with("xyz"))
    assert_frame_equal(out, df)


# # Select variables -----------------------------------------------
def test_can_be_before_group_by():
    df = tibble(
        id=c(1, 1, 2, 2, 2, 3, 3, 4, 4, 5),
        year=c(2013, 2013, 2012, 2013, 2013, 2013, 2012, 2012, 2013, 2013),
        var1=rnorm(10)
    )
    dfagg = df >> group_by(
        f.id, f.year
    ) >> select(
        f.id, f.year, f.var1
    ) >> summarise(var1=mean(f.var1))

    assert_iterable_equal(colnames(dfagg), ["id", "year", "var1"])
    assert_equal(nrow(dfagg), 8)


def test_arguments_to_select_dont_match_vars_select_arguments():
    df = tibble(a=1)
    out = select(df, var=f.a)
    assert out.frame_equal(tibble(var=1))

    out = select(group_by(df, f.a), var=f.a)
    exp = group_by(tibble(var=1), f.var)
    assert_frame_equal(ungroup(out), ungroup(exp))
    assert_iterable_equal(group_vars(out), group_vars(exp))

    out = select(df, exclude=f.a)
    assert out.frame_equal(tibble(exclude=1))
    out = select(df, include=f.a)
    assert out.frame_equal(tibble(include=1))

    out = select(group_by(df, f.a), exclude=f.a)
    exp = group_by(tibble(exclude=1), f.exclude)
    assert_frame_equal(ungroup(out), ungroup(exp))
    assert_iterable_equal(group_vars(out), group_vars(exp))

    out = select(group_by(df, f.a), include=f.a)
    exp = group_by(tibble(include=1), f.include)
    assert_frame_equal(ungroup(out), ungroup(exp))
    assert_iterable_equal(group_vars(out), group_vars(exp))


def test_can_select_with_list_of_strs():
    out = select(mtcars, "cyl", "disp", c("cyl", "am", "drat"))
    # https://github.com/pwwang/datar/issues/23
    # exp = mtcars[c("cyl", "disp", "am", "drat")]
    exp = mtcars[["cyl", "disp", "am", "drat"]]
    assert out.frame_equal(exp)


def test_treats_null_inputs_as_empty():
    out = select(mtcars, None, f.cyl, None)
    exp = select(mtcars, f.cyl)
    assert out.frame_equal(exp)


def test_can_select_with_strings():
    variabls = dict(foo="cyl", bar="am")
    out = select(mtcars, **variabls)
    exp = select(mtcars, foo=f.cyl, bar=f.am)
    assert out.frame_equal(exp)


def test_works_on_empty_names():
    df = tibble(x=1, y=2, z=3) >> set_colnames(c("x", "y", ""))
    out = select(df, f.x)
    assert_iterable_equal(out['x'], [1])

    df >>= set_colnames(c("", "y", "z"))
    out = select(df, f.y)
    assert_iterable_equal(out['y'], [2])


def test_works_on_na_names():
    df = tibble(x=1, y=2, z=3) >> set_colnames(c("x", "y", NA))
    out = select(df, f.x)
    assert_iterable_equal(out['x'], [1])

    df >>= set_colnames(c(NA, "y", "z"))
    out = select(df, f.y)
    assert_iterable_equal(out['y'], [2])


def test_keeps_attributes():
    df = tibble(x=1)
    df.datar.meta["a"] = "b"
    out = select(df, f.x)
    assert out.datar.meta["a"] == "b"


def test_tidyselect_funs():
    # tidyselect.where
    def isupper(ser):
        return ser.name.isupper()

    df = tibble(x=1, X=2, y=3, Y=4)
    out = df >> select(where(isupper))
    assert out.columns == ["X", "Y"]

    @register_verb(DataFrame, dependent=True)
    def islower(_data, series):
        return [series.name.islower(), True]

    out = df >> select(where(islower))
    assert out.columns == ["x", "y"]

    out = df >> select(where(lambda x: False))
    assert out.shape == (0, 0)

    out = df >> select(ends_with("y"))
    assert out.columns == ["y", "Y"]
    out = df >> select(contains("y"))
    assert out.columns == ["y", "Y"]

    with pytest.raises(KeyError):
        df >> select(all_of(["x", "a"]))

    out = df >> select(any_of(["x", "y"]))
    assert out.columns == ["x", "y"]
    out = df >> select(any_of(["x", "a"]))
    assert out.columns == ["x"]

    out = num_range("a", 3, width=2)
    assert out == ["a00", "a01", "a02"]

    df = tibble(tibble(XY=1), Y=2)
    out = df >> select(contains("X"))
    assert out.columns == ["XY"]
