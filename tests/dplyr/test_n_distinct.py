import pytest
import polars as pl

from datar import f
from datar.base import c, NA, sum_, identity
from datar.dplyr import n_distinct, summarise, group_by, pull
from datar.tibble import tibble
from ..conftest import assert_iterable_equal, assert_equal


def test_n_distinct_gives_correct_results():
    assert n_distinct([1, 2, 2, 3]) == 3
    assert n_distinct(pl.Series([1, 2, 2, 3])) == 3


def test_n_distinct_treats_na_correctly():
    assert n_distinct(c(1.0, NA, NA), na_rm=False) == 2
    assert n_distinct(pl.Series([1.0, None, None]), na_rm=True) == 1


def test_n_distinct_scalar():
    assert n_distinct(4) == 1
    assert n_distinct(NA, na_rm=True) == 0
    assert n_distinct([1, 2, 3, 4]) == 4


def test_n_distinct_in_summarise():
    d = tibble(x=[1, 2, 3, 4])
    res = d >> summarise(
        y=sum_(f.x),
        n5=n_distinct(f.x),
    )
    assert list(res.collect()["n5"]) == [4]


def test_n_distinct_with_groups():
    res = (
        tibble(g=[1, 1, 1, 1, 2, 2], x=[1, 2, 3, 1, 1, 2])
        >> group_by(f.g)
        >> summarise(
            y=sum_(f.x),
            n5=n_distinct(f.x),
        )
    )
    assert list(res.collect()["n5"]) == [3, 2]
