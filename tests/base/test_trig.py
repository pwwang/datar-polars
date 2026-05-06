"""Tests for base trig/math functions: log, log2, log10, exp, log1p,
acos, acosh, asin, asinh, atan, atanh, atan2,
cos, cosh, cospi, sin, sinh, sinpi, tan, tanh, tanpi.
"""

import math

import polars as pl
import pytest
from datar import f
from datar.base import (
    log,
    log2,
    log10,
    exp,
    log1p,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    atan2,
    cos,
    cosh,
    cospi,
    sin,
    sinh,
    sinpi,
    tan,
    tanh,
    tanpi,
)
from datar.dplyr import mutate
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── log ─────────────────────────────────────────────────────────────────


class TestLog:
    def test_log_scalar_default(self):
        assert log(math.e) == 1.0

    def test_log_scalar_base2(self):
        assert log(8, base=2) == 3.0

    def test_log_scalar_base10(self):
        assert log(100, base=10) == 2.0

    def test_log_series(self):
        s = pl.Series("x", [1.0, math.e, math.e**2])
        result = log(s)
        assert result.to_list() == pytest.approx([0.0, 1.0, 2.0])

    def test_log_in_mutate(self):
        df = _df({"x": [1.0, math.e]})
        out = df >> mutate(y=log(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 1.0])


# ── log2 ────────────────────────────────────────────────────────────────


class TestLog2:
    def test_log2_scalar(self):
        assert log2(8) == 3.0

    def test_log2_series(self):
        s = pl.Series("x", [1.0, 2.0, 4.0, 8.0])
        result = log2(s)
        assert result.to_list() == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_log2_in_mutate(self):
        df = _df({"x": [1.0, 2.0, 4.0]})
        out = df >> mutate(y=log2(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 1.0, 2.0])


# ── log10 ───────────────────────────────────────────────────────────────


class TestLog10:
    def test_log10_scalar(self):
        assert log10(100) == 2.0

    def test_log10_series(self):
        s = pl.Series("x", [1.0, 10.0, 100.0])
        result = log10(s)
        assert result.to_list() == pytest.approx([0.0, 1.0, 2.0])

    def test_log10_in_mutate(self):
        df = _df({"x": [1.0, 10.0, 100.0]})
        out = df >> mutate(y=log10(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 1.0, 2.0])


# ── exp ─────────────────────────────────────────────────────────────────


class TestExp:
    def test_exp_scalar(self):
        assert exp(0) == 1.0

    def test_exp_scalar_one(self):
        assert exp(1) == pytest.approx(math.e)

    def test_exp_series(self):
        s = pl.Series("x", [0.0, 1.0, 2.0])
        result = exp(s)
        assert result.to_list() == pytest.approx([1.0, math.e, math.e**2])

    def test_exp_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=exp(f.x))
        assert out.get_column("y").to_list() == pytest.approx([1.0, math.e])


# ── log1p ───────────────────────────────────────────────────────────────


class TestLog1p:
    def test_log1p_scalar(self):
        assert log1p(0) == 0.0
        assert log1p(math.e - 1) == pytest.approx(1.0)

    def test_log1p_series(self):
        s = pl.Series("x", [0.0, math.e - 1])
        result = log1p(s)
        assert result.to_list() == pytest.approx([0.0, 1.0])

    def test_log1p_in_mutate(self):
        df = _df({"x": [0.0, math.e - 1]})
        out = df >> mutate(y=log1p(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 1.0])


# ── cos ─────────────────────────────────────────────────────────────────


class TestCos:
    def test_cos_scalar(self):
        assert cos(0) == 1.0
        assert cos(math.pi) == pytest.approx(-1.0)

    def test_cos_series(self):
        s = pl.Series("x", [0.0, math.pi])
        result = cos(s)
        assert result.to_list() == pytest.approx([1.0, -1.0])

    def test_cos_in_mutate(self):
        df = _df({"x": [0.0, math.pi]})
        out = df >> mutate(y=cos(f.x))
        assert out.get_column("y").to_list() == pytest.approx([1.0, -1.0])


# ── sin ─────────────────────────────────────────────────────────────────


class TestSin:
    def test_sin_scalar(self):
        assert sin(0) == 0.0
        assert sin(math.pi / 2) == pytest.approx(1.0)

    def test_sin_series(self):
        s = pl.Series("x", [0.0, math.pi / 2])
        result = sin(s)
        assert result.to_list() == pytest.approx([0.0, 1.0])

    def test_sin_in_mutate(self):
        df = _df({"x": [0.0, math.pi / 2]})
        out = df >> mutate(y=sin(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 1.0])


# ── tan ─────────────────────────────────────────────────────────────────


class TestTan:
    def test_tan_scalar(self):
        assert tan(0) == 0.0
        assert tan(math.pi / 4) == pytest.approx(1.0)

    def test_tan_series(self):
        s = pl.Series("x", [0.0, math.pi / 4])
        result = tan(s)
        assert result.to_list() == pytest.approx([0.0, 1.0])

    def test_tan_in_mutate(self):
        df = _df({"x": [0.0, math.pi / 4]})
        out = df >> mutate(y=tan(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 1.0])


# ── cosh ────────────────────────────────────────────────────────────────


class TestCosh:
    def test_cosh_scalar(self):
        assert cosh(0) == 1.0

    def test_cosh_series(self):
        s = pl.Series("x", [0.0, 1.0])
        result = cosh(s)
        assert result.to_list() == pytest.approx([1.0, math.cosh(1.0)])

    def test_cosh_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=cosh(f.x))
        assert out.get_column("y").to_list() == pytest.approx([1.0, math.cosh(1.0)])


# ── sinh ────────────────────────────────────────────────────────────────


class TestSinh:
    def test_sinh_scalar(self):
        assert sinh(0) == 0.0

    def test_sinh_series(self):
        s = pl.Series("x", [0.0, 1.0])
        result = sinh(s)
        assert result.to_list() == pytest.approx([0.0, math.sinh(1.0)])

    def test_sinh_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=sinh(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.sinh(1.0)])


# ── tanh ────────────────────────────────────────────────────────────────


class TestTanh:
    def test_tanh_scalar(self):
        assert tanh(0) == 0.0

    def test_tanh_series(self):
        s = pl.Series("x", [0.0, 1.0])
        result = tanh(s)
        assert result.to_list() == pytest.approx([0.0, math.tanh(1.0)])

    def test_tanh_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=tanh(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.tanh(1.0)])


# ── acos ────────────────────────────────────────────────────────────────


class TestAcos:
    def test_acos_scalar(self):
        assert acos(1) == 0.0
        assert acos(0) == pytest.approx(math.pi / 2)

    def test_acos_series(self):
        s = pl.Series("x", [1.0, 0.0])
        result = acos(s)
        assert result.to_list() == pytest.approx([0.0, math.pi / 2])

    def test_acos_in_mutate(self):
        df = _df({"x": [1.0, 0.0]})
        out = df >> mutate(y=acos(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.pi / 2])


# ── asin ────────────────────────────────────────────────────────────────


class TestAsin:
    def test_asin_scalar(self):
        assert asin(0) == 0.0
        assert asin(1) == pytest.approx(math.pi / 2)

    def test_asin_series(self):
        s = pl.Series("x", [0.0, 1.0])
        result = asin(s)
        assert result.to_list() == pytest.approx([0.0, math.pi / 2])

    def test_asin_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=asin(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.pi / 2])


# ── atan ────────────────────────────────────────────────────────────────


class TestAtan:
    def test_atan_scalar(self):
        assert atan(0) == 0.0
        assert atan(1) == pytest.approx(math.pi / 4)

    def test_atan_series(self):
        s = pl.Series("x", [0.0, 1.0])
        result = atan(s)
        assert result.to_list() == pytest.approx([0.0, math.pi / 4])

    def test_atan_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=atan(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.pi / 4])


# ── acosh ───────────────────────────────────────────────────────────────


class TestAcosh:
    def test_acosh_scalar(self):
        assert acosh(1) == 0.0

    def test_acosh_series(self):
        s = pl.Series("x", [1.0, 2.0])
        result = acosh(s)
        assert result.to_list() == pytest.approx([0.0, math.acosh(2.0)])

    def test_acosh_in_mutate(self):
        df = _df({"x": [1.0, 2.0]})
        out = df >> mutate(y=acosh(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.acosh(2.0)])


# ── asinh ───────────────────────────────────────────────────────────────


class TestAsinh:
    def test_asinh_scalar(self):
        assert asinh(0) == 0.0

    def test_asinh_series(self):
        s = pl.Series("x", [0.0, 1.0])
        result = asinh(s)
        assert result.to_list() == pytest.approx([0.0, math.asinh(1.0)])

    def test_asinh_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=asinh(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.asinh(1.0)])


# ── atanh ───────────────────────────────────────────────────────────────


class TestAtanh:
    def test_atanh_scalar(self):
        assert atanh(0) == 0.0

    def test_atanh_series(self):
        s = pl.Series("x", [0.0, 0.5])
        result = atanh(s)
        assert result.to_list() == pytest.approx([0.0, math.atanh(0.5)])

    def test_atanh_in_mutate(self):
        df = _df({"x": [0.0, 0.5]})
        out = df >> mutate(y=atanh(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, math.atanh(0.5)])


# ── atan2 ───────────────────────────────────────────────────────────────


class TestAtan2:
    def test_atan2_scalar(self):
        assert atan2(0, 1) == 0.0
        assert atan2(1, 0) == pytest.approx(math.pi / 2)

    def test_atan2_series(self):
        y = pl.Series("y", [0.0, 1.0, 1.0])
        x = pl.Series("x", [1.0, 1.0, math.sqrt(3)])
        result = atan2(y, x)
        # returns numpy array for Series inputs
        import numpy as np
        assert result.tolist() == pytest.approx([0.0, math.pi / 4, math.pi / 6])

    def test_atan2_in_mutate(self):
        df = _df({"y": [0.0, 1.0, 1.0], "x": [1.0, 1.0, math.sqrt(3)]})
        out = df >> mutate(z=atan2(f.y, f.x))
        expected = [0.0, math.pi / 4, math.pi / 6]
        assert out.get_column("z").to_list() == pytest.approx(expected)


# ── cospi ───────────────────────────────────────────────────────────────


class TestCospi:
    def test_cospi_scalar(self):
        assert cospi(0) == 1.0
        assert cospi(0.5) == pytest.approx(0.0)
        assert cospi(1) == pytest.approx(-1.0)

    def test_cospi_series(self):
        s = pl.Series("x", [0.0, 0.5, 1.0])
        result = cospi(s)
        assert result.to_list() == pytest.approx([1.0, 0.0, -1.0])

    def test_cospi_in_mutate(self):
        df = _df({"x": [0.0, 0.5, 1.0]})
        out = df >> mutate(y=cospi(f.x))
        assert out.get_column("y").to_list() == pytest.approx([1.0, 0.0, -1.0])


# ── sinpi ───────────────────────────────────────────────────────────────


class TestSinpi:
    def test_sinpi_scalar(self):
        assert sinpi(0) == 0.0
        assert sinpi(0.5) == pytest.approx(1.0)
        assert sinpi(1) == pytest.approx(0.0, abs=1e-8)

    def test_sinpi_series(self):
        s = pl.Series("x", [0.0, 0.5, 1.0])
        result = sinpi(s)
        assert result.to_list() == pytest.approx([0.0, 1.0, 0.0], abs=1e-8)

    def test_sinpi_in_mutate(self):
        df = _df({"x": [0.0, 0.5, 1.0]})
        out = df >> mutate(y=sinpi(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 1.0, 0.0], abs=1e-8)


# ── tanpi ───────────────────────────────────────────────────────────────


class TestTanpi:
    def test_tanpi_scalar(self):
        assert tanpi(0) == 0.0
        assert tanpi(1) == pytest.approx(0.0, abs=1e-8)

    def test_tanpi_series(self):
        s = pl.Series("x", [0.0, 1.0])
        result = tanpi(s)
        assert result.to_list() == pytest.approx([0.0, 0.0], abs=1e-8)

    def test_tanpi_in_mutate(self):
        df = _df({"x": [0.0, 1.0]})
        out = df >> mutate(y=tanpi(f.x))
        assert out.get_column("y").to_list() == pytest.approx([0.0, 0.0], abs=1e-8)
