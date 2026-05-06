"""Tests for tibble API implementations with polars backend.

Covers: tibble, tibble_, tribble, tibble_row, as_tibble, enframe, deframe,
add_row, add_column, has_rownames, remove_rownames, rownames_to_column,
rowid_to_column, column_to_rownames.
"""

import pytest
import polars as pl

from datar import f
from datar.base import seq, c, rep
from datar.core.names import NameNonUniqueError
from datar.tibble import (
    tibble,
    tibble_,
    tribble,
    tibble_row,
    as_tibble,
    enframe,
    deframe,
    add_row,
    add_column,
    has_rownames,
    remove_rownames,
    rownames_to_column,
    rowid_to_column,
    column_to_rownames,
)
from datar_polars.tibble import Tibble, as_tibble as _raw_as_tibble

from ..conftest import assert_df_equal, assert_equal, assert_iterable_equal


# Common kwargs to route verb calls through the polars backend
_V = {"__ast_fallback": "normal", "__backend": "polars"}


# ============================================================================
# tibble() — construction
# ============================================================================


class TestTibbleConstruction:
    """Basic tibble() construction."""

    def test_from_kwargs(self):
        df = tibble(x=[1, 2, 3], y=["a", "b", "c"])
        assert isinstance(df, Tibble)
        assert df.shape == (3, 2)
        assert list(df.collect_schema().names()) == ["x", "y"]

    def test_from_args(self):
        df = tibble([1, 2, 3], ["a", "b", "c"], _name_repair="minimal")
        assert df.shape == (3, 2)

    def test_from_mixed(self):
        df = tibble([1, 2, 3], z=[7, 8, 9], _name_repair="minimal")
        assert df.shape == (3, 2)

    def test_empty(self):
        df = tibble()
        assert df.shape == (0, 0)
        assert list(df.collect_schema().names()) == []

    def test_empty_with_rows(self):
        df = tibble(_rows=5)
        assert df.shape == (5, 0)

    def test_scalar_recycling(self):
        df = tibble(x=range(1, 11), y=1)
        assert df.shape == (10, 2)
        assert df.get_column("y").to_list() == [1] * 10

    def test_scalar_recycling_length_mismatch(self):
        with pytest.raises(ValueError):
            tibble(x=range(1, 11), y=[1, 2, 3])

    def test_name_repair_check_unique(self):
        with pytest.raises(ValueError):
            # Default _name_repair="check_unique" should catch duplicates
            # from positional args with same-named Series
            x = pl.Series("x", [1])
            tibble(x, x)

    # @pytest.mark.xfail(reason="Polars does not support duplicate column names")
    def test_name_repair_minimal(self):
        x = pl.Series("x", [1])
        y = pl.Series("x", [2])
        df = tibble(x, y, _name_repair="minimal")
        # assert list(df.collect_schema().names()) == ["x", "x"]
        assert list(df.collect_schema().names()) == ["x"]

    def test_name_repair_unique(self):
        x = pl.Series("x", [1])
        df = tibble(x, x, _name_repair="unique")
        assert list(df.collect_schema().names()) == ["x__0", "x__1"]

    def test_name_repair_literals(self):
        x = pl.Series("x", [1])
        df = tibble(x, x, _name_repair=["x", "y"])
        assert list(df.collect_schema().names()) == ["x", "y"]

    def test_name_repair_literals_rename(self):
        x = pl.Series("x", [1])
        y = pl.Series("y", [2])
        df = tibble(x, y)
        df2 = tibble(df[:, :2], _name_repair=["a", "b"])
        assert list(df2.collect_schema().names()) == ["a", "b"]

    def test_tibble_with_c(self):
        df = tibble(x = c(1, 1, 1, 2, 2, 3), y = c[1:6:1], z = c[6:1:-1])
        assert df.shape == (6, 3)
        assert list(df.get_column("x").to_list()) == [1, 1, 1, 2, 2, 3]
        assert list(df.get_column("y").to_list()) == [1, 2, 3, 4, 5, 6]
        assert list(df.get_column("z").to_list()) == [6, 5, 4, 3, 2, 1]

    def test_dict_arg(self):
        df = tibble({"x": [1, 2], "y": [3, 4]})
        assert df.shape == (2, 2)

    def test_none_value_creates_none_column(self):
        df = tibble(a=None)
        assert list(df.collect_schema().names()) == ["a"]
        assert df.get_column("a").to_list() == [None]

    def test_pl_series_arg(self):
        s = pl.Series("x", [1, 2, 3])
        df = tibble(s)
        assert list(df.collect_schema().names()) == ["x"]
        assert df.get_column("x").to_list() == [1, 2, 3]

    def test_pl_dataframe_arg(self):
        inner = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        df = tibble(inner)
        assert df.shape == (2, 2)
        assert sorted(df.collect_schema().names()) == ["a", "b"]

    def test_tibble_respects_argument_order(self):
        a = range(5)
        df = tibble(a=a, b=f.a*2, c=1)
        assert list(df.collect_schema().names()) == ["a", "b", "c"]


class TestTibbleAccess:

    """Accessing tibble columns."""

    def test_get_column(self):
        df = tibble(x=[1, 2, 3], y=["a", "b", "c"])
        col_x = df.get_column("x")
        assert isinstance(col_x, pl.Series)
        assert col_x.to_list() == [1, 2, 3]

    def test_get_column_dot(self):
        df = tibble(x=[1, 2, 3], y=["a", "b", "c"])
        col_x = df.x
        assert isinstance(col_x, pl.Series)
        assert col_x.to_list() == [1, 2, 3]

    def test_get_nonexistent_column(self):
        df = tibble(x=[1, 2, 3])
        with pytest.raises(pl.exceptions.ColumnNotFoundError):
            df.get_column("y")

    def test_get_column_by_getitem(self):
        df = tibble(x=[1, 2, 3], y=["a", "b", "c"])
        col_y = df["y"]
        assert isinstance(col_y, pl.Series)
        assert col_y.to_list() == ["a", "b", "c"]

    def test_get_column_by_index(self):
        df = tibble(x=[1, 2, 3], y=["a", "b", "c"])
        col_0 = df[:, 0]
        assert isinstance(col_0, pl.Series)
        assert col_0.to_list() == [1, 2, 3]

    def test_subset_df_by_indexes(self):
        df = tibble(x=[1, 2, 3], y=["a", "b", "c"])
        subset = df[:, [0]]
        assert isinstance(subset, Tibble)
        assert subset.shape == (3, 1)
        assert list(subset.collect_schema().names()) == ["x"]


# ============================================================================
# tibble_() — pipeable
# ============================================================================


class TestTibbleUnderscore:
    """Pipeable tibble_() constructor."""

    def test_basic(self):
        df = tibble_(x=[1, 2, 3], y=[4, 5, 6])
        assert df.shape == (3, 2)
        assert isinstance(df, Tibble)


# ============================================================================
# tribble() — row-by-row
# ============================================================================


class TestTribble:
    """Row-by-row tribble() construction."""

    def test_basic(self):
        out = tribble(f.colA, f.colB, "a", 1, "b", 2)
        expect = tibble(colA=["a", "b"], colB=[1, 2])
        assert_df_equal(out.collect(), expect.collect())

    def test_multiple_rows(self):
        out = tribble(f.a, f.b, f.c, 1, 2, 3, 4, 5, 6)
        expect = tibble(a=[1, 4], b=[2, 5], c=[3, 6])
        assert_df_equal(out.collect(), expect.collect())

    def test_empty(self):
        out = tribble(f.x, f.y)
        assert out.shape == (0, 2)
        assert list(out.collect_schema().names()) == ["x", "y"]

    def test_errors_no_f_columns(self):
        with pytest.raises(ValueError):
            tribble(1, 2, 3)

    def test_errors_non_rectangular(self):
        with pytest.raises(ValueError):
            tribble(f.a, f.b, f.c, 1, 2, 3, 4, 5)

    def test_non_atomic_values(self):
        out = tribble(f.a, f.b, None, 1, 2, 3)
        expect = tibble(a=[None, 2], b=[1, 3])
        assert_df_equal(out.collect(), expect.collect())

    # @pytest.mark.xfail(reason="Polars does not support duplicate column names")
    def test_with_name_repair(self):
        out = tribble(f.x, f.x, 1, 2, _name_repair="minimal")
        # assert list(out.collect_schema().names()) == ["x", "x"]
        assert list(out.collect_schema().names()) == ["x"]


# ============================================================================
# tibble_row() — single row
# ============================================================================


class TestTibbleRow:
    """Single-row tibble_row() construction."""

    def test_basic(self):
        df = tibble_row(a=1, b="hello")
        assert df.shape == (1, 2)
        assert df.get_column("a").to_list() == [1]

    def test_empty(self):
        df = tibble_row()
        assert df.shape == (1, 0)

    def test_list_wrapped(self):
        df = tibble_row(a=[1], b=[[2, 3]])
        assert df.shape == (1, 2)

    def test_error_on_multi_row(self):
        with pytest.raises(ValueError):
            tibble_row(a=[1, 2, 3])

    def test_error_on_mixed_length(self):
        with pytest.raises(ValueError):
            tibble_row(a=1, b=[2, 3])


# ============================================================================
# as_tibble() — conversion
# ============================================================================


class TestAsTibble:
    """as_tibble() conversion to Tibble."""

    def test_from_dict(self):
        result = as_tibble({"x": [1, 2, 3], "y": ["a", "b", "c"]}, **_V)
        assert isinstance(result, Tibble)
        assert result.shape == (3, 2)

    def test_from_polars_dataframe(self):
        pdf = pl.DataFrame({"x": [1, 2, 3]})
        result = as_tibble(pdf, **_V)
        assert isinstance(result, Tibble)
        assert result.shape == (3, 1)

    def test_from_polars_lazyframe(self):
        lf = pl.DataFrame({"x": [1, 2, 3]}).lazy()
        result = as_tibble(lf, **_V)
        assert isinstance(result, Tibble)
        assert result.shape == (3, 1)

    def test_from_list_of_dicts(self):
        result = as_tibble([{"x": 1, "y": "a"}, {"x": 2, "y": "b"}], **_V)
        assert result.shape == (2, 2)

    def test_tibble_is_idempotent(self):
        df = tibble(x=[1, 2])
        result = as_tibble(df, **_V)
        assert result.shape == (2, 1)
        assert isinstance(result, Tibble)

    def test_from_pandas_df(self):
        pytest.importorskip("pandas")
        import pandas as pd
        pdf = pd.DataFrame({"x": [1, 2, 3]})
        result = as_tibble(pdf, **_V)
        assert result.shape == (3, 1)


# ============================================================================
# enframe() — list/dict to Tibble
# ============================================================================


class TestEnframe:
    """enframe() conversion to Tibble."""

    def test_from_list(self):
        result = enframe([10, 20, 30], **_V)
        assert result.shape == (3, 2)
        assert list(result.collect_schema().names()) == ["name", "value"]
        assert result.get_column("name").to_list() == [0, 1, 2]
        assert result.get_column("value").to_list() == [10, 20, 30]

    def test_from_lists(self):
        result = enframe(dict(one=1, two=[2,3], three=[4,5,6]))
        assert result.shape == (3, 2)
        assert list(result.collect_schema().names()) == ["name", "value"]
        assert result.get_column("name").to_list() == ["one", "two", "three"]
        assert result.get_column("value").to_list() == [1, [2,3], [4,5,6]]

    def test_from_list_no_name(self):
        result = enframe([10, 20, 30], name=None, **_V)
        assert result.shape == (3, 1)
        assert list(result.collect_schema().names()) == ["value"]
        assert result.get_column("value").to_list() == [10, 20, 30]

    def test_from_dict(self):
        result = enframe({"a": 1, "b": 2, "c": 3}, **_V)
        assert result.shape == (3, 2)
        assert list(result.collect_schema().names()) == ["name", "value"]

    def test_from_dict_no_name(self):
        result = enframe({"a": 1, "b": 2}, name=None, **_V)
        assert result.shape == (2, 1)
        assert sorted(result.get_column("value").to_list()) == [1, 2]

    def test_custom_column_names(self):
        result = enframe([10, 20], name="key", value="val", **_V)
        assert list(result.collect_schema().names()) == ["key", "val"]

    def test_empty_value_raises(self):
        with pytest.raises(ValueError):
            enframe([1, 2], value=None, **_V)

    def test_none_input(self):
        result = enframe(None, **_V)
        assert result.shape == (0, 2)

    def test_from_polars_series(self):
        s = pl.Series("x", [1, 2, 3])
        result = enframe(s, **_V)
        assert result.shape == (3, 2)
        assert result.get_column("value").to_list() == [1, 2, 3]

    def test_error_on_2d_input(self):
        import numpy as np
        with pytest.raises(ValueError):
            enframe(np.array([[1, 2], [3, 4]]), **_V)


# ============================================================================
# deframe() — Tibble to dict/list
# ============================================================================


class TestDeframe:
    """deframe() conversion from Tibble."""

    def test_two_column_to_dict(self):
        df = tibble(key=["a", "b", "c"], val=[1, 2, 3])
        result = deframe(df, **_V)
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_one_column_to_list(self):
        df = tibble(val=[1, 2, 3])
        result = deframe(df, **_V)
        assert result == [1, 2, 3]

    def test_from_tibble_enframe_roundtrip(self):
        data = [10, 20, 30]
        framed = enframe(data, name="idx", value="num", **_V)
        result = deframe(framed, **_V)
        expected = {0: 10, 1: 20, 2: 30}
        assert result == expected

    def test_from_plain_polars_df(self):
        pdf = pl.DataFrame({"key": ["x", "y"], "val": [10, 20]})
        result = deframe(pdf, **_V)
        assert result == {"x": 10, "y": 20}

    def test_three_column_warns(self):
        df = tibble(a=[1], b=[2], c=[3])
        result = deframe(df, **_V)
        # Warning logged but still produces output from first two cols
        assert isinstance(result, dict)


# ============================================================================
# add_row() — adding rows
# ============================================================================


class TestAddRow:
    """add_row() for appending rows to a Tibble."""

    def test_add_single_row(self):
        df = tibble(x=[1, 2], y=["a", "b"])
        result = add_row(df, x=3, y="c", **_V)
        assert result.shape == (3, 2)
        assert result.get_column("x").to_list() == [1, 2, 3]

    def test_add_empty_row(self):
        df = tibble(x=[1, 2], y=["a", "b"])
        result = add_row(df, **_V)
        assert result.shape == (3, 2)
        assert result.get_column("x").to_list() == [1, 2, None]

    def test_add_before(self):
        df = tibble(x=[2, 3])
        result = add_row(df, x=1, _before=0, **_V)
        assert result.get_column("x").to_list() == [1, 2, 3]

    def test_add_after(self):
        df = tibble(x=[1, 3])
        result = add_row(df, x=2, _after=0, **_V)
        assert result.get_column("x").to_list() == [1, 2, 3]

    def test_before_and_after_error(self):
        df = tibble(x=[1, 2])
        with pytest.raises(ValueError):
            add_row(df, x=3, _before=0, _after=0, **_V)

    def test_extra_column_error(self):
        df = tibble(x=[1, 2])
        with pytest.raises(ValueError):
            add_row(df, x=3, z=9, **_V)

    def test_metadata_preserved(self):
        df = tibble(x=[1, 2])
        df._datar = {"groups": None, "rownames": None, "backend": "polars",
                      "custom": "val"}
        result = add_row(df, x=3, **_V)
        assert hasattr(result, "_datar")
        assert result._datar.get("custom") == "val"


# ============================================================================
# add_column() — adding columns
# ============================================================================


class TestAddColumn:
    """add_column() for adding columns to a Tibble."""

    def test_add_single_column(self):
        df = tibble(x=[1, 2, 3])
        result = add_column(df, y=[4, 5, 6], **_V)
        assert result.shape == (3, 2)
        assert list(result.collect_schema().names()) == ["x", "y"]

    def test_add_scalar_column(self):
        df = tibble(x=[1, 2, 3])
        result = add_column(df, y=99, **_V)
        assert result.get_column("y").to_list() == [99, 99, 99]

    def test_add_before(self):
        df = tibble(x=[1, 2], y=[3, 4])
        result = add_column(df, z=[5, 6], _before="y", **_V)
        assert list(result.collect_schema().names()) == ["x", "z", "y"]

    def test_add_after(self):
        df = tibble(x=[1, 2], y=[3, 4])
        result = add_column(df, z=[5, 6], _after="x", **_V)
        assert list(result.collect_schema().names()) == ["x", "z", "y"]

    def test_before_and_after_error(self):
        df = tibble(x=[1, 2])
        with pytest.raises(ValueError):
            add_column(df, y=[3, 4], _before="x", _after="x", **_V)

    def test_before_nonexistent_column_error(self):
        df = tibble(x=[1, 2])
        with pytest.raises(KeyError):
            add_column(df, y=[3, 4], _before="does_not_exist", **_V)

    def test_length_mismatch_error(self):
        df = tibble(x=[1, 2, 3])
        with pytest.raises(ValueError):
            add_column(df, y=[4, 5], **_V)

    def test_name_repair_on_duplicate(self):
        df = tibble(x=[1, 2])
        with pytest.raises(ValueError):
            add_column(df, x=[3, 4], **_V)

    def test_empty_new_columns(self):
        df = tibble(x=[1, 2])
        result = add_column(df, **_V)
        assert list(result.collect_schema().names()) == ["x"]
        assert result.shape == (2, 1)


# ============================================================================
# has_rownames()
# ============================================================================


class TestHasRownames:
    """has_rownames() check."""

    def test_no_rownames(self):
        df = tibble(x=[1, 2])
        assert not has_rownames(df, **_V)

    def test_with_rownames(self):
        df = tibble(x=[1, 2])
        df._datar["rownames"] = ["r1", "r2"]
        assert has_rownames(df, **_V)


# ============================================================================
# remove_rownames()
# ============================================================================


class TestRemoveRownames:
    """remove_rownames() for removing row names."""

    def test_removes_rownames(self):
        df = tibble(x=[1, 2])
        df._datar["rownames"] = ["r1", "r2"]
        assert has_rownames(df, **_V)
        result = remove_rownames(df, **_V)
        assert not has_rownames(result, **_V)
        assert result._datar["rownames"] is None

    def test_noop_without_rownames(self):
        df = tibble(x=[1, 2])
        result = remove_rownames(df, **_V)
        assert result.shape == (2, 1)


# ============================================================================
# rownames_to_column()
# ============================================================================


class TestRownamesToColumn:
    """rownames_to_column() conversion."""

    def test_with_stored_rownames(self):
        df = tibble(x=[1, 2])
        df._datar["rownames"] = ["r1", "r2"]
        result = rownames_to_column(df, **_V)
        assert list(result.collect_schema().names()) == ["rowname", "x"]
        assert result.get_column("rowname").to_list() == ["r1", "r2"]

    def test_without_rownames_uses_integers(self):
        df = tibble(x=[10, 20, 30])
        result = rownames_to_column(df, **_V)
        assert list(result.collect_schema().names()) == ["rowname", "x"]
        assert result.get_column("rowname").to_list() == [0, 1, 2]

    def test_custom_var_name(self):
        df = tibble(x=[1, 2])
        result = rownames_to_column(df, var="id", **_V)
        assert list(result.collect_schema().names()) == ["id", "x"]

    def test_duplicate_var_name_error(self):
        df = tibble(x=[1, 2])
        with pytest.raises(ValueError):
            rownames_to_column(df, var="x", **_V)

    def test_clears_rownames_in_metadata(self):
        df = tibble(x=[1, 2])
        df._datar["rownames"] = ["r1", "r2"]
        result = rownames_to_column(df, **_V)
        assert result._datar["rownames"] is None


# ============================================================================
# rowid_to_column()
# ============================================================================


class TestRowidToColumn:
    """rowid_to_column() for adding row IDs."""

    def test_basic(self):
        df = tibble(x=[100, 200, 300])
        result = rowid_to_column(df, **_V)
        assert list(result.collect_schema().names()) == ["rowid", "x"]
        assert result.get_column("rowid").to_list() == [0, 1, 2]

    def test_custom_var_name(self):
        df = tibble(x=[1, 2])
        result = rowid_to_column(df, var="idx", **_V)
        assert list(result.collect_schema().names()) == ["idx", "x"]

    def test_duplicate_var_error(self):
        df = tibble(x=[1, 2])
        with pytest.raises(ValueError):
            rowid_to_column(df, var="x", **_V)


# ============================================================================
# column_to_rownames()
# ============================================================================


class TestColumnToRownames:
    """column_to_rownames() for converting a column to row names."""

    def test_basic(self):
        df = tibble(label=["r1", "r2"], x=[10, 20])
        result = column_to_rownames(df, var="label", **_V)
        assert list(result.collect_schema().names()) == ["x"]
        assert result._datar["rownames"] == ["r1", "r2"]

    def test_default_var(self):
        df = tibble(rowname=["a", "b"], y=[1, 2])
        result = column_to_rownames(df, **_V)
        assert list(result.collect_schema().names()) == ["y"]
        assert result._datar["rownames"] == ["a", "b"]

    def test_missing_column_error(self):
        df = tibble(x=[1, 2])
        with pytest.raises(KeyError):
            column_to_rownames(df, var="does_not_exist", **_V)

    def test_existing_rownames_error(self):
        df = tibble(label=["r1", "r2"], x=[10, 20])
        df._datar["rownames"] = ["existing"]
        with pytest.raises(ValueError):
            column_to_rownames(df, var="label", **_V)

    def test_converts_to_strings(self):
        df = tibble(label=[1, 2], x=[10, 20])
        result = column_to_rownames(df, var="label", **_V)
        assert result._datar["rownames"] == ["1", "2"]


# ============================================================================
# Round-trip / integration tests
# ============================================================================


class TestRoundTrips:
    """End-to-end round-trip scenarios."""

    def test_enframe_deframe_roundtrip(self):
        data = ["apple", "banana", "cherry"]
        framed = enframe(data, name="idx", value="fruit", **_V)
        deframed = deframe(framed, **_V)
        assert deframed == {0: "apple", 1: "banana", 2: "cherry"}

    def test_rownames_roundtrip(self):
        df = tibble(x=[10, 20])
        # Add row names via column_to_rownames
        df_with_names = tibble(label=["a", "b"], x=[10, 20])
        result = column_to_rownames(df_with_names, var="label", **_V)
        assert result._datar["rownames"] == ["a", "b"]
        # Convert back
        result2 = rownames_to_column(result, **_V)
        assert list(result2.collect_schema().names()) == ["rowname", "x"]
        assert result2.get_column("rowname").to_list() == ["a", "b"]

    def test_add_row_and_add_column(self):
        df = tibble(x=[1, 2])
        with_col = add_column(df, y=["a", "b"], **_V)
        assert with_col.shape == (2, 2)
        with_row = add_row(with_col, x=3, y="c", **_V)
        assert with_row.shape == (3, 2)
        assert with_row.get_column("x").to_list() == [1, 2, 3]

    def test_add_column_before(self):
        df = tibble(x=[1, 2], z=[5, 6])
        result = add_column(df, y=seq(3,4), _before=f.z, **_V)
        assert list(result.collect_schema().names()) == ["x", "y", "z"]

    def test_add_column_nonunique(self):
        df = tibble(x=[1, 2])
        with pytest.raises(NameNonUniqueError):
            add_column(df, x=[3, 4], **_V)

    def test_tibble_of_tribble(self):
        """tribble and tibble should produce equivalent results."""
        a = tibble(x=["a", "b"], y=[1, 2])
        b = tribble(f.x, f.y, "a", 1, "b", 2)
        assert_df_equal(a.collect(), b.collect())
