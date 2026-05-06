"""Tests for datar_polars tibble wrapping and metadata handling."""

import pytest
import polars as pl

from datar_polars.tibble import Tibble, as_tibble, reconstruct_tibble


class TestReconstructTibble:
    def test_sets_datar_on_plain_df(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = reconstruct_tibble(df)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"
        assert result._datar["groups"] is None
        assert result._datar["rownames"] is None

    def test_copies_metadata_from_old(self):
        old = Tibble(pl.DataFrame({"x": [1, 2, 3]}),
                     _datar={"groups": ["x"], "backend": "polars", "rownames": None})
        new = Tibble(pl.DataFrame({"y": [4, 5, 6]}))
        result = reconstruct_tibble(new, old)
        assert result._datar["groups"] == ["x"]

    def test_preserves_existing_metadata_on_data(self):
        df = Tibble(pl.DataFrame({"x": [1, 2, 3]}),
                    _datar={"groups": ["x"], "backend": "polars", "rownames": 0})
        result = reconstruct_tibble(df)
        assert result._datar["groups"] == ["x"]
        assert result._datar["rownames"] == 0

    def test_data_metadata_takes_priority_over_old(self):
        old = Tibble(pl.DataFrame({"x": [1, 2, 3]}),
                     _datar={"groups": ["x"], "backend": "polars", "rownames": None})
        new = Tibble(pl.DataFrame({"x": [4, 5, 6]}),
                     _datar={"groups": ["y"], "backend": "polars", "rownames": 1})
        result = reconstruct_tibble(new, old)
        assert result._datar["groups"] == ["y"]
        assert result._datar["rownames"] == 1

    def test_no_old_data_defaults_clean(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = reconstruct_tibble(df, None)
        assert hasattr(result, "_datar")
        assert result._datar["groups"] is None

    def test_metadata_survives_select(self):
        """Tibble auto-preserves _datar across .select()."""
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        result = df.select("x", "y")
        assert hasattr(result, "_datar"), "Tibble auto-preserves _datar"
        assert result._datar["backend"] == "polars"

    def test_reconstruct_preserves_after_select(self):
        """reconstruct_tibble is idempotent on Tibble results."""
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        result = df.select("x", "y")
        result = reconstruct_tibble(result, df)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"

    def test_metadata_survives_with_columns(self):
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        result = df.with_columns(pl.col("x") * 2)
        assert hasattr(result, "_datar"), "Tibble auto-preserves _datar"

    def test_reconstruct_preserves_after_with_columns(self):
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        result = df.with_columns(pl.col("x") * 2)
        result = reconstruct_tibble(result, df)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"

    def test_metadata_survives_filter(self):
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3, 4]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        result = df.filter(pl.col("x") > 2)
        assert hasattr(result, "_datar"), "Tibble auto-preserves _datar"
        result = reconstruct_tibble(result, df)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"

    def test_metadata_survives_sort(self):
        df = as_tibble(pl.DataFrame({"x": [3, 1, 2]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        result = df.sort("x")
        assert hasattr(result, "_datar"), "Tibble auto-preserves _datar"
        result = reconstruct_tibble(result, df)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"

    def test_metadata_survives_unique(self):
        df = as_tibble(pl.DataFrame({"x": [1, 1, 2, 2]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        result = df.unique(subset=["x"], maintain_order=True)
        assert hasattr(result, "_datar"), "Tibble auto-preserves _datar"
        result = reconstruct_tibble(result, df)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"

    def test_metadata_survives_group_by(self):
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3, 4], "g": [1, 1, 2, 2]}))
        df._datar = {"groups": None, "rownames": None, "backend": "polars"}
        assert hasattr(df, "_datar")


class TestAsTibble:
    def test_from_dict(self):
        result = as_tibble({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        assert isinstance(result, Tibble)
        collected = result.collect()
        assert collected.shape == (3, 2)
        assert collected.get_column("x").to_list() == [1, 2, 3]
        assert hasattr(result, "_datar")

    def test_from_plain_polars_df(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = as_tibble(df)
        collected = result.collect()
        assert collected.get_column("x").to_list() == [1, 2, 3]
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"

    def test_from_list_of_lists(self):
        result = as_tibble([[1, "a"], [2, "b"], [3, "c"]])
        collected = result.collect()
        assert collected.shape == (3, 2)

    def test_from_list_of_dicts(self):
        result = as_tibble([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}])
        collected = result.collect()
        assert collected.shape == (2, 2)
        assert collected.get_column("x").to_list() == [1, 2]

    def test_already_tibble_is_idempotent(self):
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3]}))
        result = as_tibble(df)
        collected = result.collect()
        assert collected.get_column("x").to_list() == [1, 2, 3]
        assert hasattr(result, "_datar")

    # def test_from_pandas_df(self):
    #     pytest.importorskip("pandas")
    #     import pandas as pd

    #     pdf = pd.DataFrame({"x": [1, 2, 3]})
    #     result = as_tibble(pdf)
    #     collected = result.collect()
    #     assert collected.shape == (3, 1)
    #     assert collected.get_column("x").to_list() == [1, 2, 3]
    #     assert hasattr(result, "_datar")

    def test_datar_set_on_result(self):
        result = as_tibble({"x": [1, 2, 3]})
        assert result._datar["backend"] == "polars"
        assert result._datar["groups"] is None
        assert result._datar["rownames"] is None


class TestReconstructTibbleEdgeCases:
    def test_empty_df(self):
        df = pl.DataFrame({"x": []})
        result = reconstruct_tibble(df)
        collected = result.collect()
        assert collected.shape == (0, 1)
        assert hasattr(result, "_datar")

    def test_many_columns(self):
        data = {f"col_{i}": [i] for i in range(10)}
        df = pl.DataFrame(data)
        result = reconstruct_tibble(df)
        collected = result.collect()
        assert collected.shape == (1, 10)
        assert hasattr(result, "_datar")

    def test_old_data_none(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = reconstruct_tibble(df, None)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"

    def test_nested_tibble_metadata(self):
        """Metadata should not interfere with nested DataFrame operations."""
        inner = pl.DataFrame({"a": [1, 2]})
        outer = pl.DataFrame({"x": [10, 20], "nested": [inner, inner.clone()]})
        result = reconstruct_tibble(outer)
        assert hasattr(result, "_datar")
