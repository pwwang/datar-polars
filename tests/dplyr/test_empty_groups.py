import pytest
import polars as pl

from datar import f
from datar.dplyr import group_by, group_size, summarise, n, mutate, arrange
from datar.tibble import tibble


def test_mutate_keeps_groups():
    df = tibble(x=[1, 2, 3, 4], g=[1, 1, 2, 2]) >> group_by(f.g)
    gsize = group_size(mutate(df, z=2))
    assert gsize == [2, 2]


def test_summarise_returns_a_row_for_groups():
    df = tibble(x=[1, 2, 3, 4], g=[1, 1, 2, 2])
    summarised = df >> group_by(f.g) >> summarise(z=n())
    rows = summarised.collect().height
    assert rows == 2


def test_arrange_keeps_groups():
    df = tibble(x=[1, 2, 3, 4], g=[1, 1, 2, 2]) >> group_by(f.g)
    gsize = group_size(arrange(df))
    assert gsize == [2, 2]
