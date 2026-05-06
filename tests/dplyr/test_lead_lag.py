import pytest
import polars as pl

from datar.dplyr import lead, lag


def test_lead_basic():
    x = [1, 2, 3]
    out = lead(x)
    assert list(out) == [2, 3, None]

    out = lead(x, n=2)
    assert list(out) == [3, None, None]

    out = lead(x, default=0)
    assert list(out) == [2, 3, 0]


def test_lag_basic():
    x = [1, 2, 3]
    out = lag(x)
    assert list(out) == [None, 1, 2]

    out = lag(x, n=2)
    assert list(out) == [None, None, 1]

    out = lag(x, default=0)
    assert list(out) == [0, 1, 2]


def test_lead_series():
    s = pl.Series([1, 2, 3])
    out = lead(s)
    assert list(out) == [2, 3, None]


def test_lag_series():
    s = pl.Series([1, 2, 3])
    out = lag(s)
    assert list(out) == [None, 1, 2]


def test_lead_scalar():
    out = lead(1)
    assert list(out) == [None]


def test_lag_scalar():
    out = lag(1)
    assert list(out) == [None]


def test_lead_with_default():
    out = lead([1, 2, 3], default=99)
    assert list(out) == [2, 3, 99]


def test_lag_with_default():
    out = lag([1, 2, 3], default=99)
    assert list(out) == [99, 1, 2]


def test_lead_errors():
    with pytest.raises(ValueError, match="integer"):
        lead([1, 2], n="a")
