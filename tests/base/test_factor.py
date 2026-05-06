"""Tests for factor functions: factor, ordered, levels, nlevels,
droplevels, is_factor, is_ordered, as_factor, as_ordered.
"""

import polars as pl
from datar import f
from datar.base import (
    c,
    factor,
    ordered,
    levels,
    nlevels,
    droplevels,
    is_factor,
    is_ordered,
    as_factor,
    as_ordered,
)
from datar.dplyr import mutate
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# -- as_factor -------------------------------------------------------------


class TestAsFactor:
    def test_as_factor_series(self):
        s = pl.Series("x", ["a", "b", "a", "c"])
        result = as_factor(s)
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))
        assert result.to_list() == ["a", "b", "a", "c"]

    def test_as_factor_already_factor(self):
        s = pl.Series("x", ["a", "b"], dtype=pl.Categorical)
        result = as_factor(s)
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))

    def test_as_factor_numeric(self):
        s = pl.Series("x", [1, 2, 1, 3])
        result = as_factor(s)
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))

    def test_as_factor_in_mutate(self):
        df = _df({"x": ["a", "b", "a", "c"]})
        out = df >> mutate(y=as_factor(f.x))
        assert out is not None

    def test_as_factor_list(self):
        result = as_factor(["a", "b", "a"])
        assert isinstance(result, pl.Series)
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))
        assert result.to_list() == ["a", "b", "a"]

    def test_as_factor_c(self):
        result = as_factor(c("a", "b", "a"))
        assert isinstance(result, pl.Series)
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))
        assert result.to_list() == ["a", "b", "a"]

    def test_as_factor_should_not_accumulate_levels(self):
        x = c("a", "z", "g")
        af = as_factor(x)
        assert levels(af) == ["a", "z", "g"]
        y = c("1.1", "11", "2.2", "22")
        afy = as_factor(y)
        assert levels(afy) == ["1.1", "11", "2.2", "22"]


# -- as_ordered ------------------------------------------------------------


class TestAsOrdered:
    def test_as_ordered_series(self):
        s = pl.Series("x", ["low", "medium", "high"])
        result = as_ordered(s)
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))

    def test_as_ordered_in_mutate(self):
        df = _df({"x": ["low", "medium", "high"]})
        out = df >> mutate(
            y=as_ordered(f.x)
        )
        assert out is not None

    def test_as_ordered_c(self):
        result = as_ordered(c("low", "medium", "high"))
        assert isinstance(result, pl.Series)
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))
        assert result.to_list() == ["low", "medium", "high"]


# -- factor ----------------------------------------------------------------


class TestFactor:
    def test_factor_series(self):
        s = pl.Series("x", ["a", "b", "a", "c", "b"])
        result = factor(s)
        assert isinstance(result.dtype, pl.Enum)

    def test_factor_with_levels(self):
        s = pl.Series("x", ["a", "b", "a", "c", "b"])
        result = factor(s, levels=["a", "b", "c", "d"])
        assert isinstance(result.dtype, (pl.Categorical, pl.Enum))
        assert result.cat.get_categories().to_list() == ["a", "b", "c", "d"]

    def test_factor_list(self):
        result = factor(["a", "b", "a"])
        assert isinstance(result, pl.Series)
        assert isinstance(result.dtype, pl.Enum)


# -- ordered ---------------------------------------------------------------


class TestOrdered:
    def test_ordered_series(self):
        s = pl.Series("x", ["low", "medium", "high", "low"])
        result = ordered(s)
        assert isinstance(result.dtype, pl.Enum)


# -- levels ----------------------------------------------------------------


class TestLevels:
    def test_levels_series(self):
        s = pl.Series("x", ["a", "b", "c"], dtype=pl.Categorical)
        result = levels(s)
        assert "a" in result

    def test_levels_numeric(self):
        s = as_factor(pl.Series("x", [1, 2, 3]))
        result = levels(s)
        assert result is not None


# -- nlevels ---------------------------------------------------------------


class TestNlevels:
    def test_nlevels_series(self):
        s = pl.Series("x", ["a", "b", "c"], dtype=pl.Categorical)
        result = nlevels(s)
        assert result == 3

    def test_nlevels_with_duplicates(self):
        s = pl.Series("x", ["a", "b", "a", "b"], dtype=pl.Categorical)
        result = nlevels(s)
        assert result == 2


# -- is_factor -------------------------------------------------------------


class TestIsFactor:
    def test_is_factor_true(self):
        s = pl.Series("x", ["a", "b"], dtype=pl.Categorical)
        assert is_factor(s)

    def test_is_factor_false(self):
        s = pl.Series("x", ["a", "b"])
        assert not is_factor(s)


# -- is_ordered ------------------------------------------------------------


class TestIsOrdered:
    def test_is_ordered_true(self):
        s = pl.Series("x", ["a", "b"], dtype=pl.Categorical)
        assert is_ordered(s)

    def test_is_ordered_false(self):
        s = pl.Series("x", [1.0, 2.0])
        assert not is_ordered(s)


# -- droplevels ------------------------------------------------------------


class TestDroplevels:
    def test_droplevels_series(self):
        s = pl.Series("x", ["a", "b", "a"], dtype=pl.Categorical)
        result = droplevels(s)
        assert result is not None

    def test_droplevels(self):
        fct = factor(["a", "b"], levels=["a", "b", "c"])
        result = droplevels(fct)
        assert result.cat.get_categories().to_list() == ["a", "b"]
        assert levels(result) == ["a", "b"]
