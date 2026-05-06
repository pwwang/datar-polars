"""Tests for dplyr bind verbs: bind_rows, bind_cols.
"""

import pytest
import polars as pl
from datar.dplyr import bind_rows, bind_cols
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── bind_rows ───────────────────────────────────────────────────────────


class TestBindRows:
    def test_bind_rows_two_frames(self):
        df1 = _df({"x": [1, 2], "y": [3, 4]})
        df2 = _df({"x": [5, 6], "y": [7, 8]})
        out = bind_rows(df1, df2)
        assert out.shape == (4, 2)
        assert out.get_column("x").to_list() == [1, 2, 5, 6]
        assert out.get_column("y").to_list() == [3, 4, 7, 8]

    def test_bind_rows_with_id(self):
        df1 = _df({"x": [1]})
        df2 = _df({"x": [2]})
        out = bind_rows(
            df1, df2, _id="source"
        )
        assert out.shape == (2, 2)
        assert "source" in out.collect_schema().names()
        assert out['source'].to_list() == [0, 1]
        assert out.get_column("x").to_list() == [1, 2]

    def test_bind_rows_with_id2(self):
        df1 = _df({"x": [1]})
        df2 = _df({"x": [2]})
        out = bind_rows(
            a=df1, b=df2, _id="source"
        )
        assert out.shape == (2, 2)
        assert "source" in out.collect_schema().names()
        assert out['source'].to_list() == ['a', 'b']
        assert out.get_column("x").to_list() == [1, 2]

    def test_bind_rows_single_frame(self):
        df = _df({"x": [1, 2]})
        out = bind_rows(df)
        assert out.shape == (2, 1)

    def test_bind_rows_empty(self):
        out = bind_rows(__backend="polars")
        assert out.shape == (0, 0)


# ── bind_cols ───────────────────────────────────────────────────────────


class TestBindCols:
    def test_bind_cols_two_frames(self):
        df1 = _df({"a": [1, 2]})
        df2 = _df({"b": [3, 4]})
        out = bind_cols(df1, df2)
        assert out.shape == (2, 2)
        assert list(out.collect_schema().names()) == ["a", "b"]

    def test_bind_cols_duplicate_names(self):
        df1 = _df({"x": [1, 2]})
        df2 = _df({"x": [3, 4]})
        out = bind_cols(
            df1, df2, _name_repair="unique"
        )
        # column names should be uniquified
        assert out.shape == (2, 2)
        assert len(out.collect_schema().names()) == 2
        assert out.collect_schema().names()[0] != out.collect_schema().names()[1]

    def test_bind_cols_with_dict(self):
        df = _df({"a": [1, 2]})
        out = bind_cols(df, {"b": [3, 4]})
        assert out.shape == (2, 2)
        assert out.get_column("b").to_list() == [3, 4]

    def test_bind_cols_empty(self):
        out = bind_cols(__backend="polars")
        assert out.shape == (0, 0)
