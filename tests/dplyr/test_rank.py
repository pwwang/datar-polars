"""Tests for ranking and window functions in dplyr context.

Adapted from datar-pandas tests/dplyr/test_rank.py
"""

import pytest
import polars as pl

from datar import f
from datar.base import NA, c
from datar.dplyr import (
    cume_dist,
    dense_rank,
    lag,
    lead,
    min_rank,
    mutate,
    ntile,
    percent_rank,
    row_number,
)
from datar.tibble import tibble
from datar_polars.tibble import as_tibble

from ..conftest import assert_iterable_equal


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── row_number ──────────────────────────────────────────────────────────

def test_row_number_in_mutate():
    """row_number() inside mutate gives sequential indices."""
    df = _df({"x": [10, 20, 30]})
    out = df >> mutate(
        rn=row_number()
    )
    assert out.get_column("rn").to_list() == [1, 2, 3]


def test_row_number_after_arrange():
    """row_number respects arrange order."""
    df = _df({"x": [30, 10, 20]})
    from datar.dplyr import arrange
    out = (
        df
        >> arrange(f.x)
        >> mutate(rn=row_number())
    )
    assert out.get_column("rn").to_list() == [1, 2, 3]


# ── min_rank / dense_rank ───────────────────────────────────────────────

def test_min_rank_in_mutate():
    """min_rank gives minimum rank."""
    df = _df({"x": [3, 1, 2, 2]})
    out = df >> mutate(r=min_rank(f.x))
    assert out.get_column("r").to_list() == [4, 1, 2, 2]


def test_dense_rank_in_mutate():
    """dense_rank gives dense ranking."""
    df = _df({"x": [3, 1, 2, 2]})
    out = df >> mutate(
        r=dense_rank(f.x)
    )
    assert out.get_column("r").to_list() == [3, 1, 2, 2]


# ── percent_rank / cume_dist on Series ──────────────────────────────────

def test_percent_rank_series():
    """percent_rank on a Series."""
    s = pl.Series("x", [1, 2, 3, 4])
    result = percent_rank(s)
    assert result.to_list() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_cume_dist_series():
    """cume_dist on a Series."""
    s = pl.Series("x", [1, 2, 3, 4])
    result = cume_dist(s)
    assert result.to_list() == pytest.approx([0.25, 0.5, 0.75, 1.0])


# ── lead / lag ──────────────────────────────────────────────────────────

def test_lead_lag_in_mutate():
    """lead and lag inside mutate."""
    df = _df({"x": [1, 2, 3, 4]})
    out = df >> mutate(
        y_lead=lead(f.x),
        y_lag=lag(f.x),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert out.get_column("y_lead").to_list() == [2, 3, 4, None]
    assert out.get_column("y_lag").to_list() == [None, 1, 2, 3]


# ── ntile on plain values ───────────────────────────────────────────────

def test_ntile_plain():
    """ntile on a plain Python list."""
    result = ntile([1, 2, 3, 4, 5, 6], n=3)
    # ntile splits indices: 6 items into 3 bins = [1,1,2,2,3,3]
    assert_iterable_equal(result, [1, 1, 2, 2, 3, 3])


def test_ranks():
    df = tibble(x=c(5, 1, 3, 2, 2, NA))
    out = df >> mutate(
        row_number=row_number(),
        min_rank=min_rank(f.x),
        dense_rank=dense_rank(f.x),
        percent_rank=percent_rank(f.x),
        cume_dist=cume_dist(f.x),
        ntile=ntile(f.x, n=2)
    )
    assert_iterable_equal(out.get_column("row_number").to_list(), [1, 2, 3, 4, 5, 6])
    assert_iterable_equal(out.get_column("min_rank").to_list(), [5, 1, 4, 2, 2, NA])
    assert_iterable_equal(
        out.get_column("dense_rank").to_list(), [4, 1, 3, 2, 2, NA]
    )
    assert_iterable_equal(
        out.get_column("percent_rank").to_list(), [1, 0, 0.75, 0.25, 0.25, NA]
    )
    assert_iterable_equal(
        out.get_column("cume_dist").to_list(), [1.0, 0.2, 0.8, 0.4, 0.4, NA]
    )
