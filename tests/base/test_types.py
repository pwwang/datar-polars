"""Tests for base type conversion and checking functions.

Covers: as_character, as_double, as_integer, as_logical, as_numeric,
is_atomic, is_character, is_double, is_element, is_false, is_integer,
is_logical, is_true.
"""

import polars as pl
from datar import f
from datar.base import (
    as_character,
    as_double,
    as_integer,
    as_logical,
    as_numeric,
    gsub,
    is_atomic,
    is_character,
    is_double,
    is_element,
    is_false,
    is_integer,
    is_logical,
    is_true,
)
from datar.dplyr import mutate, filter_
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── as_character ────────────────────────────────────────────────────────


class TestAsCharacter:
    def test_as_character_expr(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            y=as_character(f.x),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").dtype == pl.Utf8
        assert out.get_column("y").to_list() == ["1", "2", "3"]

    def test_as_character_series(self):
        s = pl.Series("x", [1, 2, 3])
        result = as_character(s)
        assert result.dtype == pl.Utf8
        assert result.to_list() == ["1", "2", "3"]

    def test_as_character_scalar(self):
        assert as_character(42) == "42"
        assert as_character(3.14) == "3.14"
        assert as_character(True) == "True"


# ── as_double ───────────────────────────────────────────────────────────


class TestAsDouble:
    def test_as_double_expr(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            y=as_double(f.x),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").dtype == pl.Float64
        assert out.get_column("y").to_list() == [1.0, 2.0, 3.0]

    def test_as_double_scalar(self):
        assert as_double(5) == 5.0
        assert as_double("3.14") == 3.14


# ── as_integer ──────────────────────────────────────────────────────────


class TestAsInteger:
    def test_as_integer_expr(self):
        df = _df({"x": [1.5, 2.7, 3.9]})
        out = df >> mutate(
            y=as_integer(f.x),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").dtype == pl.Int64

    def test_as_integer_scalar(self):
        assert as_integer(3.7) == 3
        assert as_integer("42") == 42

    def test_as_integer_list(self):
        lst = [1.5, 2.7, 3.9]
        result = as_integer(lst)
        assert isinstance(result, pl.Series)
        assert result.to_list() == [1, 2, 3]


# ── as_logical ──────────────────────────────────────────────────────────


class TestAsLogical:
    def test_as_logical_expr(self):
        df = _df({"x": [0, 1, 2]})
        out = df >> mutate(
            y=as_logical(f.x),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").dtype == pl.Boolean
        assert out.get_column("y").to_list() == [False, True, True]

    def test_as_logical_scalar(self):
        assert as_logical(1) is True
        assert as_logical(0) is False
        assert as_logical("") is False


# ── as_numeric ──────────────────────────────────────────────────────────


class TestAsNumeric:
    def test_as_numeric_expr(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> mutate(
            y=as_numeric(f.x),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").dtype == pl.Float64

    def test_as_numeric_scalar(self):
        assert as_numeric(5) == 5.0
        assert as_numeric("3.14") == 3.14

    def test_as_numeric_str_series(self):
        s = pl.Series("x", ["1.5", "2.7", "3.9"])
        result = as_numeric(s)
        assert result.dtype == pl.Float64
        assert result.to_list() == [1.5, 2.7, 3.9]


# ── is_atomic ───────────────────────────────────────────────────────────


class TestIsAtomic:
    def test_is_atomic_series_true(self):
        s = pl.Series("x", [1, 2, 3])
        assert is_atomic(s)

    def test_is_atomic_expr_true(self):
        assert is_atomic(pl.col("x"))

    def test_is_atomic_scalar_types(self):
        assert is_atomic(5)
        assert is_atomic(3.14)
        assert is_atomic("hello")
        assert is_atomic(True)
        assert is_atomic(1 + 2j)

    def test_is_atomic_list_false(self):
        assert not is_atomic([1, 2, 3])


# ── is_character ────────────────────────────────────────────────────────


class TestIsCharacter:
    def test_is_character_utf8_series(self):
        s = pl.Series("x", ["a", "b"])
        assert is_character(s)

    def test_is_character_categorical_series(self):
        s = pl.Series("x", ["a", "b"]).cast(pl.Categorical)
        assert is_character(s)

    def test_is_character_numeric_series_false(self):
        s = pl.Series("x", [1, 2, 3])
        assert not is_character(s)

    def test_is_character_scalar(self):
        assert is_character("hello")
        assert not is_character(42)


# ── is_double ───────────────────────────────────────────────────────────


class TestIsDouble:
    def test_is_double_float64_series(self):
        s = pl.Series("x", [1.0, 2.0], dtype=pl.Float64)
        assert is_double(s)

    def test_is_double_float32_series(self):
        s = pl.Series("x", [1.0, 2.0], dtype=pl.Float32)
        assert is_double(s)

    def test_is_double_int_series_false(self):
        s = pl.Series("x", [1, 2], dtype=pl.Int64)
        assert not is_double(s)

    def test_is_double_scalar(self):
        assert is_double(3.14)
        assert not is_double(42)


# ── is_element ──────────────────────────────────────────────────────────


class TestIsElement:
    def test_is_element_expr(self):
        df = _df({"x": [1, 5, 3]})
        out = df >> mutate(
            y=is_element(f.x, [1, 2, 3]),
            __ast_fallback="normal",
            __backend="polars",
        )
        assert out.get_column("y").to_list() == [True, False, True]

    def test_is_element_scalar_true(self):
        assert is_element(3, [1, 2, 3])

    def test_is_element_scalar_false(self):
        assert not is_element(99, [1, 2, 3])

    def test_is_element_series_check(self):
        s = pl.Series("x", [1, 5, 3])
        result = is_element(s, [1, 2, 3])
        assert result.to_list() == [True, False, True]


# ── is_false ────────────────────────────────────────────────────────────


class TestIsFalse:
    def test_is_false_scalar(self):
        assert is_false(False)
        assert not is_false(0)
        assert not is_false(True)
        assert not is_false(1)

    def test_is_false_string(self):
        assert not is_false("hello")


# ── is_integer ──────────────────────────────────────────────────────────


class TestIsInteger:
    def test_is_integer_int64_series(self):
        s = pl.Series("x", [1, 2, 3], dtype=pl.Int64)
        assert is_integer(s)

    def test_is_integer_uint32_series(self):
        s = pl.Series("x", [1, 2, 3], dtype=pl.UInt32)
        assert is_integer(s)

    def test_is_integer_float_series_false(self):
        s = pl.Series("x", [1.0, 2.0], dtype=pl.Float64)
        assert not is_integer(s)

    def test_is_integer_scalar(self):
        assert is_integer(5)
        assert not is_integer(3.14)
        assert not is_integer(True)


# ── is_logical ──────────────────────────────────────────────────────────


class TestIsLogical:
    def test_is_logical_bool_series(self):
        s = pl.Series("x", [True, False, True])
        assert is_logical(s)

    def test_is_logical_int_series_false(self):
        s = pl.Series("x", [1, 0, 1])
        assert not is_logical(s)

    def test_is_logical_scalar(self):
        assert is_logical(True)
        assert is_logical(False)
        assert not is_logical(1)


# ── is_true ─────────────────────────────────────────────────────────────


class TestIsTrue:
    def test_is_true_scalar(self):
        assert is_true(True)
        assert not is_true(1)
        assert not is_true(3.14)
        assert not is_true(-1)
        assert not is_true(False)
        assert not is_true(0)
        assert not is_true(0.0)
