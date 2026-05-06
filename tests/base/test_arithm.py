"""Tests for base arithm functions: pmin, pmax, mod, sign, signif, trunc."""

import math

import polars as pl
import pytest
from datar import f
from datar.base import pmin, pmax, mod, sign, signif, trunc
from datar.dplyr import mutate
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── pmin ────────────────────────────────────────────────────────────────


class TestPmin:
    def test_pmin_scalars(self):
        assert pmin(3, 1, 2) == 1
        assert pmin(-5, 0, 5) == -5

    def test_pmin_series(self):
        s1 = pl.Series("a", [3, 1, 5])
        s2 = pl.Series("b", [2, 4, 0])
        result = pmin(s1, s2)
        assert result.to_list() == [2, 1, 0]

    def test_pmin_three_series(self):
        s1 = pl.Series("a", [3, 1, 5])
        s2 = pl.Series("b", [2, 4, 0])
        s3 = pl.Series("c", [1, 2, 3])
        result = pmin(s1, s2, s3)
        assert result.to_list() == [1, 1, 0]

    def test_pmin_empty(self):
        assert pmin(__backend="polars") is None

    def test_pmin_lists(self):
        assert pmin([3, 1, 5], [2, 4, 0]) == [2, 1, 0]

    def test_pmin_in_mutate(self):
        df = _df({"x": [3, 1, 5], "y": [2, 4, 0]})
        out = df >> mutate(z=pmin(f.x, f.y))
        assert out.get_column("z").to_list() == [2, 1, 0]


# ── pmax ────────────────────────────────────────────────────────────────


class TestPmax:
    def test_pmax_scalars(self):
        assert pmax(3, 1, 2) == 3
        assert pmax(-5, 0, 5) == 5

    def test_pmax_series(self):
        s1 = pl.Series("a", [3, 1, 5])
        s2 = pl.Series("b", [2, 4, 0])
        result = pmax(s1, s2)
        assert result.to_list() == [3, 4, 5]

    def test_pmax_three_series(self):
        s1 = pl.Series("a", [3, 1, 5])
        s2 = pl.Series("b", [2, 4, 0])
        s3 = pl.Series("c", [1, 2, 3])
        result = pmax(s1, s2, s3)
        assert result.to_list() == [3, 4, 5]

    def test_pmax_empty(self):
        assert pmax(__backend="polars") is None

    def test_pmax_in_mutate(self):
        df = _df({"x": [3, 1, 5], "y": [2, 4, 0]})
        out = df >> mutate(
            z=pmax(f.x, f.y)
        )
        assert out.get_column("z").to_list() == [3, 4, 5]


# ── mod ─────────────────────────────────────────────────────────────────


class TestMod:
    def test_mod_positive_scalar(self):
        assert mod(5) == 5

    def test_mod_negative_scalar(self):
        assert mod(-5) == 5

    def test_mod_zero(self):
        assert mod(0) == 0

    def test_mod_float(self):
        assert mod(-3.14) == 3.14

    def test_mod_complex(self):
        result = mod(3 + 4j)
        assert result == 5.0

    def test_mod_series(self):
        s = pl.Series("x", [-3, 0, 4, -1])
        result = mod(s)
        assert result.to_list() == [3, 0, 4, 1]

    def test_mod_in_mutate(self):
        df = _df({"x": [-3, 0, 4, -1]})
        out = df >> mutate(y=mod(f.x))
        assert out.get_column("y").to_list() == [3, 0, 4, 1]


# ── sign ────────────────────────────────────────────────────────────────


class TestSign:
    def test_sign_positive_scalar(self):
        assert sign(42) == 1

    def test_sign_negative_scalar(self):
        assert sign(-7) == -1

    def test_sign_zero_scalar(self):
        assert sign(0) == 0

    def test_sign_series(self):
        s = pl.Series("x", [3, -1, 0, 4, -2])
        result = sign(s)
        assert result.to_list() == [1, -1, 0, 1, -1]

    def test_sign_in_mutate(self):
        df = _df({"x": [3, -1, 0, 4, -2]})
        out = df >> mutate(y=sign(f.x))
        assert out.get_column("y").to_list() == [1, -1, 0, 1, -1]


# ── signif ──────────────────────────────────────────────────────────────


class TestSignif:
    def test_signif_default_digits(self):
        # 123.456 -> 123.456 rounded to 6 significant digits -> 123.456
        assert pytest.approx(signif(123.456)) == 123.456

    def test_signif_three_digits(self):
        assert pytest.approx(signif(123.456, digits=3)) == 123.0

    def test_signif_two_digits(self):
        assert pytest.approx(signif(0.0012345, digits=2)) == 0.0012

    def test_signif_zero(self):
        assert signif(0, digits=4) == 0

    def test_signif_series(self):
        s = pl.Series("x", [123.456, 0.0012345, 0.0])
        result = signif(s, digits=3)
        vals = result.to_list()
        assert vals == pytest.approx([123.0, 0.00123, 0.0])

    def test_signif_in_mutate(self):
        df = _df({"x": [123.456, 78.9]})
        out = df >> mutate(
            y=signif(f.x, digits=2)
        )
        assert out.get_column("y").to_list() == pytest.approx([120.0, 79.0])


# ── trunc ───────────────────────────────────────────────────────────────


class TestTrunc:
    def test_trunc_positive_scalar(self):
        assert trunc(3.7) == 3

    def test_trunc_negative_scalar(self):
        assert trunc(-3.7) == -3

    def test_trunc_zero(self):
        assert trunc(0.0) == 0

    def test_trunc_series(self):
        s = pl.Series("x", [3.7, -3.7, 0.5, -0.5, 2.0])
        result = trunc(s)
        assert result.to_list() == [3.0, -3.0, 0.0, 0.0, 2.0]  # polars truncate returns Float

    def test_trunc_in_mutate(self):
        df = _df({"x": [3.7, -3.7, 0.5]})
        out = df >> mutate(y=trunc(f.x))
        # polars truncate returns Float64
        assert out.get_column("y").to_list() == [3.0, -3.0, 0.0]
