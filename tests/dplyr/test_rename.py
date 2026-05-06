"""Tests for dplyr rename verbs: rename, rename_with.
"""

import polars as pl
from datar import f
from datar.dplyr import rename, rename_with, select
from datar_polars.tibble import as_tibble


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── rename ──────────────────────────────────────────────────────────────


class TestRename:
    def test_rename_single_column(self):
        df = _df({"old_name": [1, 2, 3]})
        out = df >> rename(
            new_name="old_name"
        )
        assert list(out.collect_schema().names()) == ["new_name"]

    def test_rename_multiple_columns(self):
        df = _df({"a": [1], "b": [2], "c": [3]})
        out = df >> rename(
            x="a", y="b"
        )
        assert list(out.collect_schema().names()) == ["x", "y", "c"]

    def test_rename_no_change(self):
        df = _df({"x": [1, 2]})
        out = df >> rename(__backend="polars")
        assert list(out.collect_schema().names()) == ["x"]

    def test_rename_returns_copy(self):
        df = _df({"x": [1, 2]})
        out = df >> rename(new_x="x")
        assert list(df.collect_schema().names()) == ["x"]  # original unchanged
        assert list(out.collect_schema().names()) == ["new_x"]


# ── rename_with ─────────────────────────────────────────────────────────


class TestRenameWith:
    def test_rename_with_uppercase(self):
        df = _df({"hello": [1], "world": [2]})
        out = df >> rename_with(
            str.upper
        )
        assert list(out.collect_schema().names()) == ["HELLO", "WORLD"]

    def test_rename_with_specific_columns(self):
        df = _df({"a_test": [1], "b_test": [2], "c": [3]})
        out = df >> rename_with(
            lambda n: n.replace("_test", ""),
            ["a_test", "b_test"],
            __ast_fallback="normal",
            __backend="polars",
        )
        assert list(out.collect_schema().names()) == ["a", "b", "c"]

    def test_rename_with_prefix(self):
        df = _df({"x": [1], "y": [2]})
        out = df >> rename_with(
            lambda n: f"col_{n}"
        )
        assert list(out.collect_schema().names()) == ["col_x", "col_y"]
