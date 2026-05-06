"""Tests for cumulative functions: cumsum, cummax, cummin, cumprod."""

import polars as pl
from datar import f
from datar.base import cumsum, cummax, cummin, cumprod
from datar.dplyr import mutate
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# -- cumsum ----------------------------------------------------------------


class TestCumsum:
    def test_cumsum_series(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = cumsum(s)
        assert result.to_list() == [1, 3, 6, 10, 15]

    def test_cumsum_list(self):
        result = cumsum([1, 2, 3])
        import numpy as np
        assert list(result) == [1, 3, 6]

    def test_cumsum_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        out = df >> mutate(
            y=cumsum(f.x)
        )
        assert out is not None
        pdf = out.collect()
        assert pdf["y"].to_list() == [1, 3, 6, 10, 15]

    def test_cumsum_empty(self):
        s = pl.Series("x", [], dtype=pl.Float64)
        result = cumsum(s)
        assert result.to_list() == []

    def test_cumsum_with_nulls(self):
        s = pl.Series("x", [1, None, 3, None, 5])
        result = cumsum(s)
        assert result.to_list()[:1] == [1]


# -- cummax ----------------------------------------------------------------


class TestCummax:
    def test_cummax_series(self):
        s = pl.Series("x", [1, 3, 2, 5, 4])
        result = cummax(s)
        assert result.to_list() == [1, 3, 3, 5, 5]

    def test_cummax_list(self):
        result = cummax([3, 1, 4, 1, 5])
        import numpy as np
        assert list(result) == [3, 3, 4, 4, 5]

    def test_cummax_in_mutate(self):
        df = _df({"x": [3, 1, 4, 1, 5]})
        out = df >> mutate(
            y=cummax(f.x)
        )
        pdf = out.collect()
        assert pdf["y"].to_list() == [3, 3, 4, 4, 5]


# -- cummin ----------------------------------------------------------------


class TestCummin:
    def test_cummin_series(self):
        s = pl.Series("x", [3, 1, 4, 1, 5])
        result = cummin(s)
        assert result.to_list() == [3, 1, 1, 1, 1]

    def test_cummin_list(self):
        result = cummin([3, 1, 4, 1, 5])
        import numpy as np
        assert list(result) == [3, 1, 1, 1, 1]

    def test_cummin_in_mutate(self):
        df = _df({"x": [3, 1, 4, 1, 5]})
        out = df >> mutate(
            y=cummin(f.x)
        )
        pdf = out.collect()
        assert pdf["y"].to_list() == [3, 1, 1, 1, 1]


# -- cumprod ---------------------------------------------------------------


class TestCumprod:
    def test_cumprod_series(self):
        s = pl.Series("x", [1, 2, 3, 4])
        result = cumprod(s)
        assert result.to_list() == [1, 2, 6, 24]

    def test_cumprod_list(self):
        result = cumprod([2, 3, 4])
        import numpy as np
        assert list(result) == [2, 6, 24]

    def test_cumprod_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=cumprod(f.x)
        )
        pdf = out.collect()
        assert pdf["y"].to_list() == [1, 2, 6, 24]
