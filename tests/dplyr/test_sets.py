import pytest
import polars as pl

from datar.base import intersect, union, setdiff, setequal
from ..conftest import assert_iterable_equal, assert_equal


def test_works_with_vectors():
    assert_iterable_equal(intersect([1, 2, 3], [3, 4]), [3])
    assert_iterable_equal(union([1, 2, 3], [3, 4]), [1, 2, 3, 4])
    assert_iterable_equal(setdiff([1, 2, 3], [3, 4]), [1, 2])


def test_set_equality():
    assert setequal([1, 2, 3], [1, 2, 3])
    assert not setequal([1, 2], [2, 3])


def test_with_series():
    s1 = pl.Series([1, 2, 3])
    s2 = pl.Series([3, 4])
    assert_iterable_equal(intersect(s1, s2), [3])
    assert_iterable_equal(union(s1, s2), [1, 2, 3, 4])
    assert_iterable_equal(setdiff(s1, s2), [1, 2])
