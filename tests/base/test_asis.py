"""Tests for base type-check functions: is_na, is_finite, is_infinite,
is_null, is_numeric, is_integer, is_character.
"""

import math

import polars as pl
from datar import f
from datar.base import (
    is_na,
    is_finite,
    is_infinite,
    is_null,
    is_numeric,
    is_integer,
    is_character,
)
from datar.dplyr import mutate, filter_, summarise
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── is_na ───────────────────────────────────────────────────────────────


class TestIsNa:
    def test_is_na_in_filter(self):
        df = _df({"x": [1, None, 3, None]})
        out = df >> filter_(is_na(f.x))
        assert out.get_column("x").to_list() == [None, None]

    def test_is_na_in_mutate(self):
        df = _df({"x": [1, None, 3]})
        out = df >> mutate(y=is_na(f.x))
        assert out.get_column("y").to_list() == [False, True, False]

    def test_is_na_scalar(self):
        assert is_na(None)
        assert not is_na(5)


# ── is_finite ───────────────────────────────────────────────────────────


class TestIsFinite:
    def test_is_finite_in_mutate(self):
        df = _df({"x": [1.0, float("inf"), 3.0]})
        out = df >> mutate(y=is_finite(f.x))
        assert out.get_column("y").to_list() == [True, False, True]

    def test_is_finite_scalar(self):
        assert is_finite(3.0)
        assert not is_finite(float("inf"))


# ── is_infinite ─────────────────────────────────────────────────────────


class TestIsInfinite:
    def test_is_infinite_in_mutate(self):
        df = _df({"x": [1.0, float("inf"), 3.0]})
        out = df >> mutate(
            y=is_infinite(f.x)
        )
        assert out.get_column("y").to_list() == [False, True, False]

    def test_is_infinite_scalar(self):
        assert not is_infinite(3.0)
        assert is_infinite(float("inf"))


# ── is_null ─────────────────────────────────────────────────────────────


class TestIsNull:
    def test_is_null_in_filter(self):
        df = _df({"x": [1, None, 3]})
        out = df >> filter_(is_na(f.x))
        assert out.get_column("x").to_list() == [None]

    def test_is_null_scalar_true(self):
        assert is_null(None)

    def test_is_null_scalar_false(self):
        assert not is_null(42)
        assert not is_null("hello")


# ── is_numeric ──────────────────────────────────────────────────────────


class TestIsNumeric:
    def test_is_numeric_series(self):
        s = pl.Series("x", [1, 2, 3])
        assert is_numeric(s)

    def test_is_numeric_string_series(self):
        s = pl.Series("x", ["a", "b"])
        assert not is_numeric(s)

    def test_is_numeric_scalar(self):
        assert is_numeric(5)
        assert is_numeric(3.14)
        assert not is_numeric("hello")

    def test_is_numeric_sequence(self):
        assert is_numeric([1, 2, 3])
        assert not is_numeric([1, "a", 3])


# ── is_integer ──────────────────────────────────────────────────────────


class TestIsInteger:
    def test_is_integer_series(self):
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)
        assert is_integer(s)

    def test_is_integer_float_series(self):
        s = pl.Series("x", [1.0, 2.0], dtype=pl.Float64)
        assert not is_integer(s)

    def test_is_integer_scalar(self):
        assert is_integer(5)
        assert not is_integer(3.14)
        assert not is_integer(True)  # bool is not integer in R semantics

    def test_is_integer_sequence(self):
        assert is_integer([1, 2, 3])
        assert not is_integer([1, 2.5, 3])


# ── is_character ────────────────────────────────────────────────────────


class TestIsCharacter:
    def test_is_character_series(self):
        s = pl.Series("x", ["a", "b"])
        assert is_character(s)

    def test_is_character_numeric_series(self):
        s = pl.Series("x", [1, 2, 3])
        assert not is_character(s)

    def test_is_character_scalar(self):
        assert is_character("hello")
        assert not is_character(42)

    def test_is_character_sequence(self):
        assert is_character(["a", "b", "c"])
        assert not is_character([1, "a", 3])
