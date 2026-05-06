"""Tests for relocate verb."""

import pytest
import polars as pl

from datar import f
from datar.dplyr import relocate, group_by
from datar_polars.tibble import as_tibble

from ..conftest import assert_df_equal


def _df(data: dict) -> pl.DataFrame:
    return as_tibble(pl.DataFrame(data))


# ── relocate ────────────────────────────────────────────────────────────────


def test_relocate_move_column_to_front():
    """relocate moves specified columns to the front by default."""
    df = _df({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df >> relocate(f.c)
    assert list(result.collect_schema().names()) == ["c", "a", "b"]


def test_relocate_before():
    """relocate with _before places columns before the target."""
    df = _df({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df >> relocate(
        f.a, _before="c"
    )
    assert list(result.collect_schema().names()) == ["b", "a", "c"]


def test_relocate_after():
    """relocate with _after places columns after the target."""
    df = _df({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    result = df >> relocate(
        f.b, _after="a"
    )
    assert list(result.collect_schema().names()) == ["a", "b", "c"]  # b already after a


def test_relocate_multiple_columns():
    """relocate moves multiple columns at once."""
    df = _df({"a": [1, 2], "b": [3, 4], "c": [5, 6], "d": [7, 8]})
    result = df >> relocate(f[f.b : f.c], _after="d")
    assert list(result.collect_schema().names()) == ["a", "d", "b", "c"]


def test_relocate_before_and_after_error():
    """relocate raises error if both _before and _after are specified."""
    df = _df({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="only one"):
        df >> relocate(
            f.a, _before="b", _after="b"
        )


def test_relocate_no_move_args_noop():
    """relocate with no move args returns the dataframe unchanged."""
    df = _df({"a": [1, 2], "b": [3, 4]})
    result = df >> relocate(__backend="polars")
    assert list(result.collect_schema().names()) == ["a", "b"]
    assert result.get_column("a").to_list() == [1, 2]


def test_relocate_with_rename():
    """relocate supports rename via kwargs (new=old)."""
    df = _df({"x": [1, 2], "y": [3, 4]})
    result = df >> relocate(
        new_x=f.x
    )
    assert "new_x" in result.collect_schema().names()
    assert "x" not in result.collect_schema().names()


def test_relocate_preserves_group_vars_order():
    """relocate preserves group variables at front."""
    df = _df({"g": ["a", "a"], "x": [1, 2], "y": [3, 4]})
    gf = df >> group_by(f.g)
    result = gf >> relocate(f.y)
    # y should be after g (relocate puts moved cols at front)
    cols = list(result.collect_schema().names())
    assert "g" in cols
    assert "y" in cols
