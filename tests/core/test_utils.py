"""Tests for datar_polars utility functions."""

import pytest
import polars as pl

from datar import f
from datar_polars.tibble import Tibble, as_tibble, reconstruct_tibble
from datar_polars.utils import vars_select, name_of, is_scalar, NO_DEFAULT, NA_character_
from datar_polars.common import setdiff, union, intersect, unique, is_null, is_integer
from datar_polars.collections import Collection, Negated, Inverted, Intersect


class TestVarsSelect:
    def test_vars_select_by_string(self):
        cols = ["a", "b", "c", "d"]
        result = vars_select(cols, "a", "c")
        assert list(result) == [0, 2]

    def test_vars_select_by_index(self):
        cols = ["a", "b", "c", "d"]
        result = vars_select(cols, 0, 2)
        assert list(result) == [0, 2]

    def test_vars_select_by_reference_attr(self):
        cols = ["a", "b", "c"]
        result = vars_select(cols, f.a, f.c)
        assert list(result) == [0, 2]

    def test_vars_select_mixed(self):
        cols = ["a", "b", "c", "d"]
        result = vars_select(cols, "a", 2, f.d)
        assert list(result) == [0, 2, 3]

    def test_vars_select_raises_on_missing(self):
        cols = ["a", "b"]
        with pytest.raises(KeyError):
            vars_select(cols, "z")

    def test_vars_select_no_raise(self):
        cols = ["a", "b"]
        result = vars_select(cols, "z", raise_nonexists=False)
        assert list(result) == []

    def test_vars_select_raises_on_duplicate_names(self):
        cols = ["a", "b", "a"]
        with pytest.raises(ValueError, match="Names must be unique"):
            vars_select(cols, "a")

    def test_vars_select_empty_args(self):
        cols = ["a", "b", "c"]
        result = vars_select(cols)
        assert list(result) == []

    def test_vars_select_null_skipped(self):
        cols = ["a", "b", "c"]
        result = vars_select(cols, None, "b")
        assert list(result) == [1]


class TestNameOf:
    def test_name_of_string(self):
        assert name_of("hello") == "hello"

    def test_name_of_int(self):
        assert name_of(42) == "42"

    def test_name_of_series(self):
        s = pl.Series("mycol", [1, 2, 3])
        assert name_of(s) == "mycol"

    def test_name_of_unnamed_series(self):
        s = pl.Series([1, 2, 3])
        assert name_of(s) is None


class TestIsScalar:
    def test_is_scalar_int(self):
        assert is_scalar(1) is True

    def test_is_scalar_str(self):
        assert is_scalar("hello") is True

    def test_is_scalar_none(self):
        assert is_scalar(None) is True

    def test_is_scalar_list(self):
        assert is_scalar([1, 2, 3]) is False

    def test_is_scalar_series(self):
        assert is_scalar(pl.Series([1, 2, 3])) is False

    def test_is_scalar_empty_list(self):
        assert is_scalar([]) is True


class TestSetdiff:
    def test_basic(self):
        assert setdiff([1, 2, 3, 4], [2, 3]) == [1, 4]

    def test_empty_a(self):
        assert setdiff([], [1, 2]) == []

    def test_empty_b(self):
        assert setdiff([1, 2, 3], []) == [1, 2, 3]

    def test_no_overlap(self):
        assert setdiff([1, 2], [3, 4]) == [1, 2]


class TestUnion:
    def test_basic(self):
        assert union([1, 2], [2, 3]) == [1, 2, 3]

    def test_order_preserved(self):
        assert union([3, 1], [2, 3]) == [3, 1, 2]

    def test_empty(self):
        assert union([], [1, 2]) == [1, 2]
        assert union([1, 2], []) == [1, 2]


class TestIntersect:
    def test_basic(self):
        assert intersect([1, 2, 3], [2, 3, 4]) == [2, 3]

    def test_no_overlap(self):
        assert intersect([1, 2], [3, 4]) == []

    def test_order_from_first(self):
        assert intersect([3, 1, 2], [2, 3, 4]) == [3, 2]


class TestCollection:
    def test_basic_collection(self):
        c = Collection("a", "b", pool=["a", "b", "c"])
        assert list(c) == [0, 1]

    def test_collection_unmatched(self):
        c = Collection("a", "z", pool=["a", "b", "c"])
        assert c.unmatched == {"z"}

    def test_collection_empty(self):
        c = Collection(pool=["a", "b"])
        assert list(c) == []

    def test_collection_with_none(self):
        c = Collection(None, "a", pool=["a", "b"])
        assert list(c) == [0]

    def test_collection_nested(self):
        inner = Collection("b", "c", pool=["a", "b", "c", "d"])
        c = Collection("a", inner, pool=["a", "b", "c", "d"])
        assert list(c) == [0, 1, 2]

    def test_collection_int_pool(self):
        c = Collection(0, 2, pool=5)
        assert list(c) == [0, 2]

    def test_collection_int_pool_out_of_range(self):
        c = Collection(0, 5, pool=5)
        assert c.unmatched == {5}


class TestNegated:
    def test_negated_int_pool(self):
        n = Negated(0, pool=5)
        assert sorted(list(n)) == [1, 2, 3, 4]

    def test_negated_no_pool(self):
        n = Negated(0, 1, 2)
        assert list(n) == [0, -1, -2]


class TestInverted:
    def test_inverted_int_pool(self):
        inv = Inverted(0, 1, pool=5)
        assert sorted(list(inv)) == [2, 3, 4]

    def test_inverted_no_pool(self):
        inv = Inverted(0, 1)
        assert list(inv) == [0, 1]


class TestIntersectCollection:
    def test_intersect_expand(self):
        ic = Intersect([0, 1, 2, 3], [2, 3, 4], pool=5)
        ic.expand()
        assert list(ic) == [2, 3]


class TestReconstructTibble:
    def test_sets_datar(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = reconstruct_tibble(df)
        assert hasattr(result, "_datar")
        assert result._datar["backend"] == "polars"
        assert result._datar["groups"] is None
        assert result._datar["rownames"] is None

    def test_preserves_existing_datar(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        df._datar = {"groups": ["x"], "backend": "polars", "rownames": None}
        result = reconstruct_tibble(df)
        assert result._datar["groups"] == ["x"]

    def test_copies_from_old_data(self):
        old = pl.DataFrame({"x": [1, 2, 3]})
        old._datar = {"groups": ["x"], "backend": "polars", "rownames": None}
        new = pl.DataFrame({"x": [4, 5, 6]})
        result = reconstruct_tibble(new, old)
        assert result._datar["groups"] == ["x"]


class TestAsTibble:
    def test_from_dict(self):
        result = as_tibble({"x": [1, 2, 3]})
        collected = result.collect()
        assert collected.get_column("x").to_list() == [1, 2, 3]
        assert hasattr(result, "_datar")

    def test_from_polars_df(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        result = as_tibble(df)
        collected = result.collect()
        assert collected.get_column("x").to_list() == [1, 2, 3]
        assert hasattr(result, "_datar")

    def test_from_list_of_dicts(self):
        result = as_tibble([{"x": 1}, {"x": 2}])
        collected = result.collect()
        assert collected.get_column("x").to_list() == [1, 2]
        assert hasattr(result, "_datar")

    def test_preserves_existing_metadata(self):
        df = Tibble(pl.DataFrame({"x": [1, 2, 3]}),
                    _datar={"groups": ["x"], "backend": "polars", "rownames": 0})
        result = as_tibble(df)
        assert result._datar["groups"] == ["x"]
        assert result._datar["rownames"] == 0

    def test_metadata_survives_basic_ops(self):
        df = as_tibble(pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}))
        result = df.with_columns(pl.col("x") * 2)
        # Tibble auto-preserves _datar
        assert hasattr(result, "_datar")


class TestCommonUtils:
    def test_unique_list(self):
        result = unique([1, 2, 2, 3, 1])
        assert set(result) == {1, 2, 3}

    def test_unique_series(self):
        s = pl.Series("x", [1, 2, 2, 3, 1])
        result = unique(s)
        assert set(result) == {1, 2, 3}

    def test_is_null_series(self):
        s = pl.Series("x", [1, None, 3])
        result = is_null(s)
        assert result.to_list() == [False, True, False]

    def test_is_null_scalar(self):
        assert is_null(None) is True
        assert is_null(1) is False

    def test_is_integer(self):
        assert is_integer(5) is True
        assert is_integer(3.14) is False
        assert is_integer("hello") is False
        assert is_integer([1, 2, 3]) is True
        assert is_integer([1.5, 2.0]) is False
