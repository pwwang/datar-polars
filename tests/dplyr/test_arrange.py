# https://github.com/tidyverse/dplyr/blob/master/tests/testthat/test-arrange.r

import pytest

from polars import Series, NotFoundError, ShapeError
from polars.testing import assert_frame_equal
from datar import f
from datar.tibble import tibble
from datar.base import NA, rep, c
from datar.dplyr import (
    arrange,
    desc,
    group_by,
    group_vars,
    group_rows,
    across,
    ungroup,
)
from datar_polars.tibble import TibbleGrouped

from ..conftest import assert_iterable_equal, assert_equal, assert_


def test_empty_returns_self():
    df = tibble(x=range(1, 11), y=range(1, 11))
    gf = df >> group_by(f.x)

    assert_(arrange(df).frame_equal(df))

    out = arrange(gf)
    assert_frame_equal(ungroup(out), ungroup(gf))
    assert_equal(group_vars(out), group_vars(gf))


def test_sort_empty_df():
    df = tibble()
    out = df >> arrange()
    assert_frame_equal(out, df)


def test_na_end():
    df = tibble(x=c(4, 3, NA))  # NA makes it float
    out = df >> arrange(f.x)
    assert_iterable_equal(out['x'], [3, 4, None])
    out = df >> arrange(desc(f.x))
    assert_iterable_equal(out['x'], [4, 3, None])


def test_errors():
    x = Series(values=[1], name="x")

    df = tibble(x=x)
    with pytest.raises(NotFoundError):
        df >> arrange(f.y)

    with pytest.raises(ShapeError):
        df >> arrange(rep(f.x, 2))


def test_df_cols():
    df = tibble(x=[1, 2, 3], y=tibble(z=[3, 2, 1]))
    out = df >> arrange(f.y)
    expect = tibble(x=[3, 2, 1], y=tibble(z=[1, 2, 3]))
    assert out.frame_equal(expect)


# polars does not support complex
# def test_complex_cols():
#     df = tibble(x=[1, 2, 3], y=[3 + 2j, 2 + 2j, 1 + 2j],
#           _dtypes={"y": "complex"})
#     out = df >> arrange(f.y)
#     assert_iterable_equal(out['x'], [3, 2, 1])


def test_ignores_group():
    df = tibble(g=[2, 1] * 2, x=[4, 3, 2, 1])
    gf = df >> group_by(f.g)
    out = gf >> arrange(f.x)
    assert out.frame_equal(df[[3, 2, 1, 0], :])

    out = gf >> arrange(f.x, _by_group=True)
    exp = df[[3, 1, 2, 0], :]
    assert_frame_equal(out, exp)


def test_update_grouping():
    df = tibble(g=[2, 2, 1, 1], x=[1, 3, 2, 4])
    res = df >> group_by(f.g) >> arrange(f.x)
    assert isinstance(res, TibbleGrouped)
    out = group_rows(res)
    assert_iterable_equal(out[0], [0, 2])
    assert_iterable_equal(out[1], [1, 3])


def test_across():
    df = tibble(x=[1, 3, 2, 1], y=[4, 3, 2, 1])

    out = df >> arrange(across())
    expect = df >> arrange(f.x, f.y)
    assert out.frame_equal(expect)

    out = df >> arrange(across(None, desc))
    expect = df >> arrange(desc(f.x), desc(f.y))
    assert_frame_equal(out, expect)

    out = df >> arrange(across(f.x))
    expect = df >> arrange(f.x)
    assert out.frame_equal(expect)

    out = df >> arrange(across(f.y))
    expect = df >> arrange(f.y)
    assert out.frame_equal(expect)
