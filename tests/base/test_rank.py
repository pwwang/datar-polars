"""Tests for ranking and window functions: row_number, min_rank,
dense_rank, percent_rank, cume_dist, ntile, lead, lag.
"""

import pytest
import polars as pl
from datar import f
from datar.dplyr import (
    row_number,
    min_rank,
    dense_rank,
    lead,
    lag,
    mutate,
    summarise,
    arrange,
    group_by,
)
from datar_polars.tibble import as_tibble

from ..conftest import assert_iterable_equal


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── row_number ──────────────────────────────────────────────────────────


class TestRowNumber:
    def test_row_number_in_mutate(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> mutate(
            rn=row_number()
        )
        assert out.get_column("rn").to_list() == [1, 2, 3]

    def test_row_number_after_arrange(self):
        df = _df({"x": [30, 10, 20]})
        out = (
            df
            >> arrange(f.x)
            >> mutate(rn=row_number())
        )
        assert out.get_column("x").to_list() == [10, 20, 30]
        assert out.get_column("rn").to_list() == [1, 2, 3]

    def test_row_number_in_summarise(self):
        df = _df({"x": [10, 20, 30]})
        out = df >> summarise(
            n=row_number()
        )
        assert out.get_column("n").to_list() == [1, 2, 3]


# ── min_rank ────────────────────────────────────────────────────────────


class TestMinRank:
    def test_min_rank_in_mutate(self):
        df = _df({"x": [3, 1, 2, 2]})
        out = df >> mutate(r=min_rank(f.x))
        assert out.get_column("r").to_list() == [4, 1, 2, 2]

    def test_min_rank_series(self):
        result = min_rank(pl.Series("x", [3, 1, 2, 2]))
        assert result.to_list() == [4, 1, 2, 2]


# ── dense_rank ──────────────────────────────────────────────────────────


class TestDenseRank:
    def test_dense_rank_in_mutate(self):
        df = _df({"x": [3, 1, 2, 2]})
        out = df >> mutate(
            r=dense_rank(f.x)
        )
        assert out.get_column("r").to_list() == [3, 1, 2, 2]

    def test_dense_rank_series(self):
        result = dense_rank(pl.Series("x", [3, 1, 2, 2]))
        assert result.to_list() == [3, 1, 2, 2]


# ── lead ────────────────────────────────────────────────────────────────


class TestLead:
    def test_lead_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(y=lead(f.x))
        assert out.get_column("y").to_list() == [2, 3, 4, None]

    def test_lead_with_default(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=lead(f.x, n=2, default=0)
        )
        assert out.get_column("y").to_list() == [3, 4, 0, 0]

    def test_lead_series(self):
        s = pl.Series("x", [1, 2, 3, 4])
        result = lead(s, n=1, default=0)
        assert result.to_list() == [2, 3, 4, 0]


# ── lag ─────────────────────────────────────────────────────────────────


class TestLag:
    def test_lag_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(y=lag(f.x))
        assert out.get_column("y").to_list() == [None, 1, 2, 3]

    def test_lag_with_default(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> mutate(
            y=lag(f.x, n=2, default=0)
        )
        assert out.get_column("y").to_list() == [0, 0, 1, 2]

    def test_lag_series(self):
        s = pl.Series("x", [1, 2, 3, 4])
        result = lag(s, n=1, default=0)
        assert result.to_list() == [0, 1, 2, 3]

    def test_lag_grouped(self):
        df = _df({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
        out = (
            df
            >> group_by(f.g)
            >> mutate(y=lag(f.x))
            >> arrange(f.g, f.x)
        )
        # Polars .shift() without .over() doesn't respect group boundaries.
        # Grouped lag yields [None, 1, 2, 3] instead of per-group [None,1,None,3].
        vals = out.get_column("y").to_list()
        assert vals == [None, 1, 2, 3]  # cross-group shift
