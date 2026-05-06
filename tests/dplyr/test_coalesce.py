import pytest
import polars as pl

from datar.base import NA, c
from datar.dplyr import coalesce
from datar.tibble import tibble
from ..conftest import assert_iterable_equal, assert_equal


def test_missing_replaced():
    x = [NA, 1]
    out = coalesce(x, 1)
    assert list(out) == [1, 1]


def test_common_type():
    out = coalesce(NA, 1)
    assert out == 1

    out = coalesce(None, 1)
    assert out == 1


def test_multiple_replaces():
    x1 = c(1, NA, NA)
    x2 = c(NA, 2, NA)
    x3 = c(NA, NA, 3)
    out = coalesce(x1, x2, x3)
    assert list(out) == [1, 2, 3]


def test_no_rep():
    x = c(1, 2, NA, NA, 5)
    out = coalesce(x)
    assert list(out) == list(x)
