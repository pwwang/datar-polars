"""Tests for table functions: table, tabulate."""

import polars as pl
from datar.base import table, tabulate
from datar_polars.tibble import Tibble


# -- table -----------------------------------------------------------------


class TestTable:
    def test_table_series(self):
        s = pl.Series("x", ["a", "b", "a", "c", "b", "a"])
        result = table(s)
        assert isinstance(result, Tibble)
        pdf = result.collect() if hasattr(result, "collect") else result
        vals = pdf.to_dict(as_series=False)
        assert "a" in vals.get(pdf.columns[0], [])

    def test_table_list(self):
        result = table(["a", "b", "a", "c", "b"])
        assert isinstance(result, Tibble)

    def test_table_two_series(self):
        x = pl.Series("x", ["a", "b", "a", "b"])
        y = pl.Series("y", [1, 1, 2, 2])
        result = table(x, y)
        assert isinstance(result, Tibble)

    def test_table_empty(self):
        s = pl.Series("x", [], dtype=pl.Utf8)
        result = table(s)
        assert isinstance(result, Tibble)


# -- tabulate --------------------------------------------------------------


class TestTabulate:
    def test_tabulate_series(self):
        s = pl.Series("x", [1, 2, 1, 3, 2, 1])
        result = tabulate(s)
        assert isinstance(result, list)

    def test_tabulate_list(self):
        result = tabulate([1, 2, 2, 3, 3, 3])
        assert isinstance(result, list)

    def test_tabulate_with_nbins(self):
        s = pl.Series("x", [1, 2, 1, 3])
        result = tabulate(s, nbins=5)
        assert len(result) == 5
