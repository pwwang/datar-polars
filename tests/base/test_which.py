"""Tests for base which functions: which, which_min, which_max."""
import polars as pl
import pytest
from datar.base import which, which_min, which_max
from datar.dplyr import mutate
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# -- which ---------------------------------------------------------------


class TestWhich:
    def test_which_series_bool(self):
        s = pl.Series("x", [True, False, True, False])
        result = which(s)
        assert result == [1, 3]

    def test_which_series_numeric(self):
        s = pl.Series("x", [0, 1, 0, 2, 0])
        result = which(s)
        assert result == [2, 4]

    def test_which_series_empty(self):
        s = pl.Series("x", [False, False, False])
        result = which(s)
        assert result == []

    def test_which_list(self):
        result = which([True, False, True])
        assert result == [1, 3]

    def test_which_in_mutate(self):
        from datar import f
        df = _df({"x": [3, 1, 6, 2, 5]})
        out = df >> mutate(
            y=which(f.x > 3)
        )
        assert out is not None


# -- which_min -----------------------------------------------------------


class TestWhichMin:
    def test_which_min_series(self):
        s = pl.Series("x", [3, 1, 5, 1, 2])
        result = which_min(s)
        assert result == 2  # first minimum, 1-based

    def test_which_min_series_unique(self):
        s = pl.Series("x", [5, 3, 1, 4, 2])
        result = which_min(s)
        assert result == 3

    def test_which_min_list(self):
        result = which_min([5, 3, 1, 4, 2])
        assert result == 3

    def test_which_min_in_mutate(self):
        from datar import f
        df = _df({"x": [3, 1, 5]})
        out = df >> mutate(
            y=which_min(f.x)
        )
        assert out is not None


# -- which_max -----------------------------------------------------------


class TestWhichMax:
    def test_which_max_series(self):
        s = pl.Series("x", [3, 5, 1, 5, 2])
        result = which_max(s)
        assert result == 2  # first maximum, 1-based

    def test_which_max_series_unique(self):
        s = pl.Series("x", [5, 3, 1, 4, 2])
        result = which_max(s)
        assert result == 1

    def test_which_max_list(self):
        result = which_max([1, 5, 3])
        assert result == 2

    def test_which_max_in_mutate(self):
        from datar import f
        df = _df({"x": [3, 1, 5]})
        out = df >> mutate(
            y=which_max(f.x)
        )
        assert out is not None
