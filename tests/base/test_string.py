"""Tests for base string functions: paste, paste0, toupper, tolower,
nchar, nzchar.
"""
import pytest
import polars as pl
from datar import f
from datar.base import paste, paste0, toupper, tolower, nchar, nzchar
from datar.dplyr import mutate, filter_
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── toupper ─────────────────────────────────────────────────────────────


class TestToupper:
    def test_toupper_in_mutate(self):
        df = _df({"x": ["hello", "world"]})
        out = df >> mutate(y=toupper(f.x))
        assert out.get_column("y").to_list() == ["HELLO", "WORLD"]

    def test_toupper_scalar(self):
        assert toupper("hello") == "HELLO"


# ── tolower ─────────────────────────────────────────────────────────────


class TestTolower:
    def test_tolower_in_mutate(self):
        df = _df({"x": ["HELLO", "WORLD"]})
        out = df >> mutate(y=tolower(f.x))
        assert out.get_column("y").to_list() == ["hello", "world"]

    def test_tolower_scalar(self):
        assert tolower("HELLO") == "hello"


# ── nchar ───────────────────────────────────────────────────────────────


class TestNchar:
    def test_nchar_in_mutate(self):
        df = _df({"x": ["hi", "hello", "a"]})
        out = df >> mutate(y=nchar(f.x))
        assert out.get_column("y").to_list() == [2, 5, 1]

    def test_nchar_scalar(self):
        assert nchar("hello") == 5
        assert nchar("") == 0

    def test_nchar_zero_byte(self):
        # nchar should count zero-byte characters
        assert nchar("a\0b") == 2
        assert nchar("\0b") == 1
        with pytest.raises(ValueError, match="invalid zero-byte character"):
            nchar("\0")
        with pytest.raises(ValueError, match="invalid zero-byte character"):
            nchar("a\0")


# ── nzchar ──────────────────────────────────────────────────────────────


class TestNzchar:
    def test_nzchar_in_filter(self):
        df = _df({"x": ["", "hello", "", "world"]})
        out = df >> filter_(
            nzchar(f.x)
        )
        assert out.get_column("x").to_list() == ["hello", "world"]

    def test_nzchar_in_mutate(self):
        df = _df({"x": ["", "hi"]})
        out = df >> mutate(y=nzchar(f.x))
        assert out.get_column("y").to_list() == [False, True]

    def test_nzchar_scalar(self):
        assert not nzchar("")
        assert nzchar("hello")

    def test_nzchar_list(self):
        assert nzchar(["", "hello", ""]) == [False, True, False]


# ── paste ───────────────────────────────────────────────────────────────


class TestPaste:
    def test_paste_in_mutate_two_columns(self):
        df = _df({"a": ["x", "y"], "b": ["1", "2"]})
        out = df >> mutate(
            c=paste(f.a, f.b)
        )
        assert out.get_column("c").to_list() == ["x 1", "y 2"]

    def test_paste_custom_sep(self):
        df = _df({"a": ["x", "y"], "b": ["1", "2"]})
        out = df >> mutate(
            c=paste(f.a, f.b, sep="-")
        )
        assert out.get_column("c").to_list() == ["x-1", "y-2"]

    def test_paste_literal(self):
        df = _df({"a": ["x", "y"]})
        out = df >> mutate(
            c=paste(f.a, "suffix")
        )
        assert out.get_column("c").to_list() == ["x suffix", "y suffix"]

    def test_paste_scalar(self):
        # paste with scalars
        result = paste("a", "b")
        assert result == "a b"
        # returns a pl.Expr; evaluate directly
        df = _df({"dummy": [1]})
        out = df >> mutate(
            c=paste("a", "b", sep="-")
        )
        assert out.get_column("c").to_list() == ["a-b"]

    def test_paste_list(self):
        result = paste(["a", "b"], sep =".")
        assert result == ["a", "b"]

        # paste with lists
        result = paste(["a", "b"], ["c", "d"], sep=".")
        assert result == ["a.c", "b.d"]


# ── paste0 ──────────────────────────────────────────────────────────────


class TestPaste0:
    def test_paste0_in_mutate(self):
        df = _df({"a": ["x", "y"], "b": ["1", "2"]})
        out = df >> mutate(
            c=paste0(f.a, f.b)
        )
        assert out.get_column("c").to_list() == ["x1", "y2"]

    def test_paste0_literal(self):
        df = _df({"a": ["x", "y"]})
        out = df >> mutate(
            c=paste0(f.a, "_suffix")
        )
        assert out.get_column("c").to_list() == ["x_suffix", "y_suffix"]

    def test_paste0_scalar(self):
        df = _df({"dummy": [1]})
        out = df >> mutate(
            c=paste0("hello", "world")
        )
        assert out.get_column("c").to_list() == ["helloworld"]

    def test_paste0_list(self):
        result = paste0(["a", "c"], ["b", "d"], collapse="; ")
        assert result == "ab; cd"
