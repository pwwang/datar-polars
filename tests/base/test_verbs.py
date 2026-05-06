"""Tests for base arithmetic/aggregation functions: sum_, mean, median,
min_, max_, prod, abs_, sqrt, round_, ceiling, floor, sd, var.
"""

import pytest
import polars as pl
from datar import f
from datar.base import (
    sum_,
    mean,
    median,
    min_,
    max_,
    prod,
    abs_,
    sqrt,
    round_,
    ceiling,
    floor,
    sd,
    var,
    diag,
)
from datar.dplyr import mutate, summarise, group_by
from datar.tibble import tibble
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal, assert_equal, assert_iterable_equal


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── sum_ ────────────────────────────────────────────────────────────────


class TestSum:
    def test_sum_in_mutate(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(y=sum_(f.x))
        assert out.get_column("y").to_list() == [6, 6, 6]

    def test_sum_in_summarise(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(y=sum_(f.x))
        assert out.get_column("y").to_list() == [6]

    def test_sum_scalar(self):
        assert sum_([1, 2, 3, 4]) == 10

    def test_sum_grouped(self):
        df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
        out = (
            df
            >> group_by(f.g)
            >> summarise(y=sum_(f.x))
        )
        out = out.sort("g")
        assert out.get_column("y").to_list() == [3, 3]


# ── mean ────────────────────────────────────────────────────────────────


class TestMean:
    def test_mean_in_mutate(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(y=mean(f.x))
        assert out.get_column("y").to_list() == [2.0, 2.0, 2.0]

    def test_mean_in_summarise(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(y=mean(f.x))
        assert out.get_column("y").to_list() == [2.0]

    def test_mean_scalar(self):
        assert mean([1, 2, 3]) == 2.0

    def test_mean_grouped(self):
        df = _df({"g": ["a", "a", "b"], "x": [1, 3, 5]})
        out = (
            df
            >> group_by(f.g)
            >> summarise(y=mean(f.x))
        )
        out = out.sort("g")
        assert out.get_column("y").to_list() == [2.0, 5.0]


# ── median ──────────────────────────────────────────────────────────────


class TestMedian:
    def test_median_in_mutate(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(y=median(f.x))
        assert out.get_column("y").to_list() == [2.0, 2.0, 2.0]

    def test_median_in_summarise(self):
        df = _df({"x": [1, 3, 5]})
        out = df >> summarise(y=median(f.x))
        assert out.get_column("y").to_list() == [3.0]

    def test_median_scalar(self):
        assert median([1, 3, 5]) == 3.0

    def test_median_grouped(self):
        df = _df({"g": ["a", "a", "b"], "x": [1, 5, 10]})
        out = (
            df
            >> group_by(f.g)
            >> summarise(y=median(f.x))
        )
        out = out.sort("g")
        assert out.get_column("y").to_list() == [3.0, 10.0]


# ── min_ ────────────────────────────────────────────────────────────────


class TestMin:
    def test_min_in_mutate(self):
        df = _df({"x": [3, 1, 2]})
        out = df >> mutate(y=min_(f.x))
        assert out.get_column("y").to_list() == [1, 1, 1]

    def test_min_in_summarise(self):
        df = _df({"x": [3, 1, 2]})
        out = df >> summarise(y=min_(f.x))
        assert out.get_column("y").to_list() == [1]

    def test_min_scalar(self):
        assert min_([3, 1, 2]) == 1


# ── max_ ────────────────────────────────────────────────────────────────


class TestMax:
    def test_max_in_mutate(self):
        df = _df({"x": [3, 1, 2]})
        out = df >> mutate(y=max_(f.x))
        assert out.get_column("y").to_list() == [3, 3, 3]

    def test_max_in_summarise(self):
        df = _df({"x": [3, 1, 2]})
        out = df >> summarise(y=max_(f.x))
        assert out.get_column("y").to_list() == [3]

    def test_max_scalar(self):
        assert max_([3, 1, 2]) == 3


# ── prod ────────────────────────────────────────────────────────────────


class TestProd:
    def test_prod_in_summarise(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(y=prod(f.x))
        assert out.get_column("y").to_list() == [6]

    def test_prod_scalar(self):
        assert prod([1, 2, 3, 4]) == 24


# ── abs_ ────────────────────────────────────────────────────────────────


class TestAbs:
    def test_abs_in_mutate(self):
        df = _df({"x": [-1, 2, -3]})
        out = df >> mutate(y=abs_(f.x))
        assert out.get_column("y").to_list() == [1, 2, 3]

    def test_abs_scalar(self):
        assert abs_(-5) == 5


# ── sqrt ────────────────────────────────────────────────────────────────


class TestSqrt:
    def test_sqrt_in_mutate(self):
        df = _df({"x": [1, 4, 9]})
        out = df >> mutate(y=sqrt(f.x))
        assert out.get_column("y").to_list() == [1.0, 2.0, 3.0]

    def test_sqrt_scalar(self):
        assert sqrt(16) == 4.0


# ── round_ ──────────────────────────────────────────────────────────────


class TestRound:
    def test_round_in_mutate(self):
        df = _df({"x": [1.234, 2.567, 3.891]})
        out = df >> mutate(y=round_(f.x, digits=1))
        assert out.get_column("y").to_list() == [1.2, 2.6, 3.9]

    def test_round_default_digits(self):
        df = _df({"x": [1.4, 2.6]})
        out = df >> mutate(y=round_(f.x))
        assert out.get_column("y").to_list() == [1.0, 3.0]

    def test_round_scalar(self):
        assert round_(3.14159, digits=2) == 3.14


# ── ceiling ─────────────────────────────────────────────────────────────


class TestCeiling:
    def test_ceiling_in_mutate(self):
        df = _df({"x": [1.2, 2.7, 3.0]})
        out = df >> mutate(y=ceiling(f.x))
        assert out.get_column("y").to_list() == [2.0, 3.0, 3.0]

    def test_ceiling_scalar(self):
        assert ceiling(2.3) == 3.0


# ── floor ───────────────────────────────────────────────────────────────


class TestFloor:
    def test_floor_in_mutate(self):
        df = _df({"x": [1.2, 2.7, 3.0]})
        out = df >> mutate(y=floor(f.x))
        assert out.get_column("y").to_list() == [1.0, 2.0, 3.0]

    def test_floor_scalar(self):
        assert floor(2.9) == 2.0


# ── sd ──────────────────────────────────────────────────────────────────


class TestSd:
    def test_sd_in_summarise(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(y=sd(f.x))
        assert out.get_column("y").to_list() == [1.0]

    def test_sd_grouped(self):
        df = _df({"g": ["a", "a", "b", "b"], "x": [1, 3, 5, 7]})
        out = (
            df
            >> group_by(f.g)
            >> summarise(y=sd(f.x))
        )
        out = out.sort("g")
        vals = out.get_column("y").to_list()
        assert abs(vals[0] - 1.4142135623730951) < 0.01  # sd of [1,3] = sqrt(2)


# ── var ─────────────────────────────────────────────────────────────────


class TestVar:
    def test_var_in_summarise(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> summarise(y=var(f.x))
        assert out.get_column("y").to_list() == [1.0]

    def test_var_scalar(self):
        import numpy as np

        assert var([1, 2, 3, 4]) == pytest.approx(np.var([1, 2, 3, 4], ddof=1))


# ── diag ───────────────────────────────────────────────────────────────


class TestDiag:
    def test_diag_scalar(self):
        assert diag([1, 2, 3]).tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
