"""Tests for across, pick, c_across, if_any, if_all."""

import pytest
import polars as pl

from datar import f
from datar.dplyr import (
    across,
    pick,
    c_across,
    if_any,
    if_all,
    everything,
    mutate,
    summarise,
    group_by,
    filter_,
)
from datar.tibble import tibble
from datar_polars.tibble import as_tibble

from ..conftest import assert_iterable_equal, assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


# ── across ──────────────────────────────────────────────────────────────────

def test_across_no_args_returns_all_non_group_cols():
    """across() with no arguments selects all non-group columns."""
    df = _df({"x": [1, 2], "y": [3, 4], "z": [5, 6]})
    result = df >> mutate(
        test=across(),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert "x" in result.collect_schema().names()
    assert "y" in result.collect_schema().names()
    assert "z" in result.collect_schema().names()


def test_across_single_function():
    """across with a single function applies it to selected columns."""
    df = _df({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df >> summarise(
        across(f[f.a : f.b], lambda x: x.sum()),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert result.get_column("a").to_list() == [6]
    assert result.get_column("b").to_list() == [15]


def test_across_with_complex_expr():
    """across with a complex expression."""
    df = _df({"a": [1, 1, 2, 2], "b": [3, 3, 4, 4]})
    result = df >> group_by(f.a, __ast_fallback="piping") >> summarise(
        across(everything(), lambda x: [x.sum()]),
        __ast_fallback="piping",
        __backend="polars",
    )
    assert result.get_column("a").to_list() == [1, 2]
    assert result.get_column("b").to_list() == [[6], [8]]


def test_across_named_functions():
    """across with a dict of named functions."""
    df = _df({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df >> summarise(
        across(f[f.a : f.b], {"sum": lambda x: x.sum(), "mean": lambda x: x.mean()}),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert "a_sum" in result.collect_schema().names()
    assert "b_mean" in result.collect_schema().names()


def test_across_list_functions():
    """across with a list of functions."""
    df = _df({"a": [1, 2, 3]})
    n_unique = lambda x: x.n_unique()
    result = df >> summarise(
        across(f.a, [lambda x: x.sum(), lambda x: x.mean(), n_unique]),
        __ast_fallback="normal",
        __backend="polars",
    )
    # Three columns: a_0 (sum), a_1 (mean), a_2 (n_unique)
    assert result.shape[1] >= 3


def test_across_single_col_single_fn_returns_expr():
    """across on a single column with single fn returns a single expression."""
    df = _df({"a": [1, 2, 3]})
    result = df >> mutate(
        doubled=across(f.a, lambda x: x * 2),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert result.get_column("a").to_list() == [2, 4, 6]


def test_across_grouped():
    """across works with grouped data."""
    df = _df({"g": ["a", "a", "b"], "x": [1, 2, 3]})
    gf = df >> group_by(f.g)
    result = gf >> summarise(
        across(f.x, lambda x: x.sum()),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert result.get_column("x").to_list() == [3, 3]  # a:1+2=3, b:3


def test_across_select_specific_columns():
    """across selects only specified columns."""
    df = _df({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df >> mutate(
        across(f[f.a : f.b], lambda x: x * 10),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert result.get_column("a").to_list() == [10, 20]
    assert result.get_column("b").to_list() == [30, 40]
    assert result.get_column("c").to_list() == [5, 6]  # unchanged


# ── pick ────────────────────────────────────────────────────────────────────

def test_pick_selects_columns():
    """pick selects specific columns."""
    df = _df({"x": [1, 2], "y": [3, 4]})
    result = df >> pick(f.x)
    assert list(result.collect_schema().names()) == ["x"]


def test_pick_requires_at_least_one_column():
    """pick raises error if no columns specified."""
    df = _df({"x": [1, 2]})
    with pytest.raises(ValueError):
        df >> pick(__backend="polars")


# ── if_any / if_all ─────────────────────────────────────────────────────────


def test_if_any_true_for_any_match():
    """if_any returns True if any selected column matches the predicate."""
    df = _df({"a": [1, 10, 0], "b": [0, 0, 0]})
    result = df >> filter_(
        if_any(f[f.a : f.b], lambda x: x > 5),
        __ast_fallback="normal",
        __backend="polars",
    )
    # Row 0: a=1 >5? No, b=0 >5? No → False
    # Row 1: a=10 >5? Yes → True
    # Row 2: a=0 >5? No, b=0 >5? No → False
    assert result.shape[0] == 1
    assert result.get_column("a").to_list() == [10]


def test_if_all_true_when_all_match():
    """if_all returns True only if all selected columns match."""
    df = _df({"a": [3, 10, 6], "b": [4, 20, 5]})
    result = df >> filter_(
        if_all(f[f.a : f.b], lambda x: x > 5),
        __ast_fallback="normal",
        __backend="polars",
    )
    # Row 0: a=3 >5? No → False
    # Row 1: a=10 >5? Yes, b=20 >5? Yes → True
    # Row 2: a=6 >5? Yes, b=5 >5? No → False
    assert result.shape[0] == 1
    assert result.get_column("a").to_list() == [10]


def test_if_any_no_predicate_uses_col_as_bool():
    """if_any with no predicate uses column values as booleans."""
    df = _df({"a": [0, 1, 0], "b": [0, 0, 1]})
    result = df >> filter_(
        if_any(f[f.a : f.b]),
        __ast_fallback="normal",
        __backend="polars",
    )
    # Row 0: 0,0 → False; Row 1: 1,0 → True; Row 2: 0,1 → True
    assert result.shape[0] == 2


def test_if_all_no_predicate_uses_col_as_bool():
    """if_all with no predicate uses column values as booleans."""
    df = _df({"a": [1, 1, 0], "b": [1, 0, 0]})
    result = df >> filter_(
        if_all(f[f.a : f.b]),
        __ast_fallback="normal",
        __backend="polars",
    )
    assert result.shape[0] == 1  # only row 0 has both non-zero


# ── c_across ────────────────────────────────────────────────────────────────

def test_c_across_returns_expressions():
    """c_across selects columns for rowwise operations."""
    df = _df({"a": [1, 2], "b": [3, 4]})
    result = df >> mutate(
        x=c_across(_cols=f[f.a : f.b]),
        __ast_fallback="normal",
        __backend="polars",
    )
    # c_across selects columns a and b inside mutate
    assert "a" in result.collect_schema().names()
    assert "b" in result.collect_schema().names()
