"""Tests for dplyr count verbs: count, tally, add_count, add_tally.
"""

import pytest
import polars as pl
from datar import f
from datar.data import starwars
from datar.base import round_
from datar.dplyr import count, tally, add_count, add_tally, group_by
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── count ───────────────────────────────────────────────────────────────


class TestCount:
    def test_count_ungrouped(self):
        df = _df({"x": [1, 2, 2, 3]})
        out = df >> count(__backend="polars")
        assert out.shape == (1, 1)
        assert out.get_column("n").to_list() == [4]

    def test_count_by_column(self):
        df = _df({"x": [1, 2, 2, 3]})
        out = df >> count(f.x)
        out = out.sort("x")
        assert out.get_column("x").to_list() == [1, 2, 3]
        assert out.get_column("n").to_list() == [1, 2, 1]

    def test_count_with_new_column(self):
        df = _df({"x": [1, 2, 2, 3]})
        out = df >> count(count=f.x)
        out = out.sort("x")
        assert out.get_column("x").to_list() == [1, 2, 3]
        assert out.get_column("count").to_list() == [1, 2, 1]

    def test_count_sorted(self):
        df = _df({"x": [2, 2, 1, 3, 1]})
        out = df >> count(f.x, sort=True)
        vals = out.get_column("n").to_list()
        # sorted by count desc
        assert vals == sorted(vals, reverse=True)

    def test_count_with_weight(self):
        df = _df({"x": [1, 2, 2], "w": [0.5, 1, 2]})
        out = df >> count(f.x, wt=f.w)
        out = out.sort("x")
        assert out.get_column("n").to_list() == [0.5, 3.0]

    def test_count_starwars(self):
        out = starwars >> count(f.sex, f.gender, sort=True, __ast_fallback="piping")
        assert out["n"].to_list()[:3] == [60, 16, 5]

    def test_count_starwars2(self):
        # unexpected, count change the behavior later
        starwars >> count(f.species)
        out = starwars >> count(f.sex, f.gender, sort=True, __ast_fallback="piping")
        assert out["n"].to_list()[:3] == [60, 16, 5]

    def test_count_starwars3(self):
        out = starwars >> count(birth_decade=round_(f.birth_year, -1), __ast_fallback="piping")
        assert out["birth_decade"].to_list()[:3] == [20.0, 110.0, 30.0]
        assert out["n"].to_list()[:3] == [6, 1, 4]


# ── tally ───────────────────────────────────────────────────────────────


class TestTally:
    def test_tally_ungrouped(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> tally(__backend="polars")
        assert out.shape == (1, 1)
        assert out.get_column("n").to_list() == [3]

    def test_tally_grouped(self):
        df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
        out = (
            df
            >> group_by(f.g)
            >> tally(__backend="polars")
        )
        out = out.sort("g")
        assert out.get_column("n").to_list() == [2, 1]

    def test_tally_weighted(self):
        df = _df({"x": [1, 2, 3], "w": [0.5, 1.0, 1.5]})
        out = df >> tally(wt=f.w)
        assert out.get_column("n").to_list() == [3.0]


# ── add_count ───────────────────────────────────────────────────────────


class TestAddCount:
    def test_add_count_ungrouped(self):
        df = _df({"x": [1, 2, 2]})
        out = df >> add_count(__backend="polars")
        assert out.get_column("n").to_list() == [3, 3, 3]

    def test_add_count_by_column(self):
        df = _df({"x": [1, 2, 2, 3]})
        out = df >> add_count(f.x)
        out = out.sort("x")
        assert out.get_column("n").to_list() == [1, 2, 2, 1]


# ── add_tally ───────────────────────────────────────────────────────────


class TestAddTally:
    def test_add_tally_ungrouped(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> add_tally(__backend="polars")
        assert out.get_column("n").to_list() == [3, 3, 3]

    def test_add_tally_grouped(self):
        df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
        out = (
            df
            >> group_by(f.g)
            >> add_tally(__backend="polars")
        )
        out = out.sort("g", "x")
        assert out.get_column("n").to_list() == [2, 2, 1]
