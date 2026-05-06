import pytest
import polars as pl

from datar import f
from datar.base import cummax, cummin, cumsum, NA, is_na
from datar.tibble import tibble
from datar.dplyr import (
    mutate, min_rank, desc, pull, group_by, row_number,
    lead, lag,
)
from ..conftest import assert_iterable_equal


def test_desc_correctly_handled_by_window_functions():
    df = tibble(x=list(range(1, 11)), y=list(range(1, 11)))
    out = mutate(df, rank=min_rank(desc(f.x))) >> pull(to="list")
    assert out == list(range(10, 0, -1))


def test_cum_sum_min_max_works():
    df = tibble(x=list(range(1, 11)), y=list(range(1, 11)))
    res = mutate(df, csumx=cumsum(f.x), cminx=cummin(f.x), cmaxx=cummax(f.x))


def test_lag_handles_default_argument_in_mutate():
    blah = tibble(x1=[5, 10, 20, 27, 35, 58, 5, 6])
    blah = mutate(blah, x2=f.x1 - lag(f.x1, n=1, default=0))


def test_min_rank_handles_columns_full_of_nas():
    test = tibble(Name=list("abcde"), ID=[1] * 5, expression=[NA] * 5)
    data = group_by(test, f.ID) >> mutate(rank=min_rank(f.expression))
