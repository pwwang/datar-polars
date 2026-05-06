import pytest
import polars as pl

from datar import f
from datar.base import cumsum
from datar.dplyr import order_by, with_order, mutate
from datar.tibble import tibble
from ..conftest import assert_iterable_equal


def test_order_by():
    df = tibble(x=list(range(1, 6)))
    out = df >> mutate(y=order_by(list(range(5, 0, -1)), cumsum(f.x)))
    assert_iterable_equal(out.collect()["y"], [15, 14, 12, 9, 5])


def test_with_order():
    x = [1, 2, 3, 4, 5]
    out = with_order(list(range(5, 0, -1)), cumsum, x)
    assert_iterable_equal(out, [15, 14, 12, 9, 5])
