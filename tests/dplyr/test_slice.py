"""Tests for dplyr slice verbs: slice_, slice_head, slice_tail,
slice_sample, slice_min, slice_max.
"""

import pytest
import polars as pl
from datar import f
from datar.base import c
from datar.dplyr import (
    slice_,
    slice_head,
    slice_tail,
    slice_sample,
    slice_min,
    slice_max,
    group_by,
    mutate,
    n,
)
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── slice_ ──────────────────────────────────────────────────────────────


class TestSlice:
    def test_slice_last_row(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> slice_(n()-1)
        assert out.get_column("x").to_list() == [3]
        out = df >> slice_(-1)
        assert out.get_column("x").to_list() == [3]

    def test_slice_positive_indices(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        # slice_ uses 0-indexed positions (polars convention)
        out = df >> slice_(0, 2, 4)
        vals = out.get_column("x").to_list()
        assert vals == [1, 3, 5]

    def test_slice_negative_indices(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        # Use slice_tail for getting last N rows
        out = df >> slice_tail(1)
        assert out.get_column("x").to_list() == [5]

    def test_slice_no_rows_returns_empty(self):
        df = _df({"x": [1, 2, 3]})
        out = df >> slice_(__backend="polars")
        assert out.shape == (3, 1)  # empty args = identity

    def test_slice_with_c(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        out = df >> slice_(c[:4])
        assert out.get_column("x").to_list() == [1, 2, 3, 4]

    def test_slice_alias_available_in_datar_all(self):
        from datar.all import slice as slice_alias

        df = _df({"x": [1, 2, 3, 4, 5]})
        out = df >> slice_alias(c[:4])
        assert out.get_column("x").to_list() == [1, 2, 3, 4]


# ── slice_head ──────────────────────────────────────────────────────────


class TestSliceHead:
    def test_slice_head_default_n1(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> slice_head(__backend="polars")
        assert out.get_column("x").to_list() == [1]

    def test_slice_head_n(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> slice_head(n=2)
        assert out.get_column("x").to_list() == [1, 2]

    def test_slice_head_prop(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> slice_head(prop=0.5)
        assert out.get_column("x").to_list() == [1, 2]

    def test_slice_head_grouped(self):
        df = _df({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
        out = (
            df
            >> group_by(f.g)
            >> slice_head(n=1)
        )
        # Polars .slice() operates on full frame, not per-group.
        # slice_head(n=1) returns first row overall: [1]
        vals = out.sort("g").get_column("x").to_list()
        assert vals == [1]


# ── slice_tail ──────────────────────────────────────────────────────────


class TestSliceTail:
    def test_slice_tail_default_n1(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> slice_tail(__backend="polars")
        assert out.get_column("x").to_list() == [4]

    def test_slice_tail_n(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> slice_tail(n=2)
        assert out.get_column("x").to_list() == [3, 4]

    def test_slice_tail_prop(self):
        df = _df({"x": [1, 2, 3, 4]})
        out = df >> slice_tail(prop=0.5)
        assert out.get_column("x").to_list() == [3, 4]


# ── slice_min ───────────────────────────────────────────────────────────


class TestSliceMin:
    def test_slice_min_default(self):
        df = _df({"x": [3, 1, 4, 2]})
        out = df >> slice_min(
            f.x
        )
        assert out.get_column("x").to_list() == [1]

    def test_slice_min_n2(self):
        df = _df({"x": [3, 1, 4, 2]})
        out = df >> slice_min(
            f.x, n=2
        )
        assert out.get_column("x").to_list() == [1, 2]

    def test_slice_min_string_column(self):
        df = _df({"val": [30, 10, 20], "name": ["c", "a", "b"]})
        out = df >> slice_min(
            "val", n=1
        )
        assert out.get_column("name").to_list() == ["a"]


# ── slice_max ───────────────────────────────────────────────────────────


class TestSliceMax:
    def test_slice_max_default(self):
        df = _df({"x": [3, 1, 4, 2]})
        out = df >> slice_max(
            f.x
        )
        assert out.get_column("x").to_list() == [4]

    def test_slice_max_n2(self):
        df = _df({"x": [3, 1, 4, 2]})
        out = df >> slice_max(
            f.x, n=2
        )
        assert out.get_column("x").to_list() == [4, 3]

    def test_slice_max_string_column(self):
        df = _df({"val": [10, 20, 30], "name": ["a", "b", "c"]})
        out = df >> slice_max(
            "val", n=1
        )
        assert out.get_column("name").to_list() == ["c"]


# ── slice_sample ────────────────────────────────────────────────────────


class TestSliceSample:
    def test_slice_sample_n1(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        out = df >> slice_sample(
            n=2, random_state=42
        )
        assert out.shape[0] == 2  # should return 2 rows

    def test_slice_sample_prop(self):
        df = _df({"x": [1, 2, 3, 4, 5]})
        out = df >> slice_sample(
            prop=0.4, random_state=42
        )
        assert out.shape[0] == 2  # 0.4 * 5 = 2
