"""Tests for base seq functions: seq_along, seq_len, rep, rev, sample."""

import polars as pl
import pytest
from datar import f
from datar.base import seq_along, seq_len, rep, rev, sample, length, c
from datar.dplyr import mutate
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── seq_along ───────────────────────────────────────────────────────────


class TestSeqAlong:
    def test_seq_along_series(self):
        s = pl.Series("x", [10, 20, 30])
        result = seq_along(s)
        assert result.to_list() == [1, 2, 3]

    def test_seq_along_list(self):
        result = seq_along([10, 20, 30, 40])
        assert result == [1, 2, 3, 4]

    def test_seq_along_empty(self):
        s = pl.Series("x", [], dtype=pl.Int64)
        result = seq_along(s)
        assert result.to_list() == []

    def test_seq_along_scalar(self):
        result = seq_along(42)
        assert result == [1]

    def test_seq_along_in_mutate(self):
        df = _df({"x": [10, 20, 30]})
        from datar import f
        out = df >> mutate(
            y=seq_along(f.x)
        )
        vals = out.get_column("y").to_list()
        assert vals == [1, 2, 3]


# ── seq_len ─────────────────────────────────────────────────────────────


class TestSeqLen:
    def test_seq_len_int(self):
        result = seq_len(3)
        assert result == [0, 1, 2]

    def test_seq_len_zero(self):
        result = seq_len(0)
        assert result == []

    def test_seq_len_one(self):
        result = seq_len(1)
        assert result == [0]

    def test_seq_len_in_mutate(self):
        df = _df({"x": [1, 2, 3]})
        from datar import f
        out = df >> mutate(
            y=seq_len(f.x)
        )
        assert out is not None


# ── rep ─────────────────────────────────────────────────────────────────


class TestRep:
    def test_rep_series_times(self):
        s = pl.Series("x", [1, 2, 3])
        result = rep(s, times=2)
        assert result.to_list() == [1, 2, 3, 1, 2, 3]

    def test_rep_series_each(self):
        s = pl.Series("x", [1, 2, 3])
        result = rep(s, each=2)
        assert result.to_list() == [1, 1, 2, 2, 3, 3]

    def test_rep_series_length_out(self):
        s = pl.Series("x", [1, 2, 3])
        result = rep(s, times=3, length=5)
        assert result.to_list() == [1, 2, 3, 1, 2]

    def test_rep_list(self):
        result = rep([1, 2], times=3)
        assert result == [1, 2, 1, 2, 1, 2]

    def test_rep_scalar(self):
        result = rep(5, times=3)
        assert result == [5, 5, 5]

    def test_rep_in_mutate(self):
        df = _df({"x": [1, 2]})
        out = df >> mutate(
            y=rep(f.x, times=2)
        )
        assert out is not None

    def test_rep_with_c(self):
        result = rep(c(1, 2), times=c(1, 2))
        assert result == [1, 2, 2]


# ── rev ─────────────────────────────────────────────────────────────────


class TestRev:
    def test_rev_series(self):
        s = pl.Series("x", [1, 2, 3, 4])
        result = rev(s)
        assert result.to_list() == [4, 3, 2, 1]

    def test_rev_list(self):
        result = rev([1, 2, 3])
        assert result == [3, 2, 1]

    def test_rev_scalar(self):
        result = rev(42)
        assert result == 42

    def test_rev_empty(self):
        s = pl.Series("x", [], dtype=pl.Int64)
        result = rev(s)
        assert result.to_list() == []

    def test_rev_in_mutate(self):
        df = _df({"x": [1, 2, 3]})
        from datar import f
        out = df >> mutate(
            y=rev(f.x)
        )
        vals = out.get_column("y").to_list()
        assert vals == [3, 2, 1]


# ── sample ──────────────────────────────────────────────────────────────


class TestSample:
    def test_sample_series_no_replace(self):
        s = pl.Series("x", [1, 2, 3, 4, 5])
        result = sample(s, size=3)
        assert len(result) == 3
        assert set(result.to_list()).issubset({1, 2, 3, 4, 5})

    def test_sample_series_all(self):
        s = pl.Series("x", [1, 2, 3])
        result = sample(s)
        assert len(result) == 3
        assert set(result.to_list()) == {1, 2, 3}

    def test_sample_list(self):
        result = sample([1, 2, 3, 4, 5], size=2)
        assert len(result) == 2
        assert set(result).issubset({1, 2, 3, 4, 5})

    def test_sample_scalar(self):
        result = sample(42)
        assert result == [42]

    def test_sample_in_mutate(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        from datar import f
        out = df >> mutate(
            y=sample(f.x, size=3)
        )
        assert out is not None

    def test_sample_with_replace(self):
        x = sample(range(10), 1e5, replace=True)
        assert len(x) == 1e5
        assert len(set(x)) <= 10  # With replacement, we should have duplicates


# ── length ──────────────────────────────────────────────────────────────


class TestLength:
    def test_length_series(self):
        s = pl.Series("x", [1, 2, 3, 4])
        result = length(s)
        assert result == 4

    def test_length_list(self):
        result = length([1, 2, 3])
        assert result == 3

    def test_length_scalar(self):
        result = length(42)
        assert result == 1

    def test_length_string(self):
        result = length("hello")
        assert result == 1

    def test_length_empty_series(self):
        s = pl.Series("x", [], dtype=pl.Int64)
        result = length(s)
        assert result == 0
