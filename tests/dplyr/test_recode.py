"""Tests for recode and recode_factor."""

import pytest
import polars as pl

from datar.base import NA, factor, c, levels
from datar.dplyr import recode, recode_factor

from ..conftest import assert_iterable_equal


# ── recode ──────────────────────────────────────────────────────────────────


def test_recode_simple_pair():
    """recode replaces values using positional pairs."""
    result = recode([1, 2, 3], 1, "a")
    assert result.to_list() == ["a", "2", "3"]


def test_recode_dict():
    """recode with a dict argument."""
    result = recode([1, 2, 3], {1: "one", 2: "two"})
    assert result.to_list() == ["one", "two", "3"]


def test_recode_default():
    """recode with _default replaces unmatched values."""
    result = recode(
        [1, 2, 3], "a", _default="other"
    )
    assert result.to_list() == ["other", "other", "other"]


def test_recode_missing():
    """recode with _missing replaces None/NA values."""
    result = recode(
        [1, None, 3], 1, "a", _missing="NA_val"
    )
    assert result.to_list() == ["a", "NA_val", "3"]


def test_recode_kwargs():
    """recode supports keyword argument replacements."""
    result = recode(["x", "y", "z"], x="X", y="Y")
    assert result.to_list() == ["X", "Y", "z"]


def test_recode_no_replacements_error():
    """recode raises error if no replacements provided and no defaults."""
    with pytest.raises(ValueError):
        recode([1, 2, 3])


def test_recode_int_to_str():
    """recode converts integer values to strings."""
    result = recode([1, 2, 3], None, "one", "two", "three")
    assert result.to_list() == ["one", "two", "three"]


def test_recode_factor_treated_as_vector():
    factor_vec = factor(c("a", "b", "c"))
    result = recode(factor_vec, a="Apple")
    assert result.to_list() == ["Apple", "b", "c"]


# ── recode_factor ───────────────────────────────────────────────────────────


def test_recode_factor():
    """recode_factor replaces factor values and preserves levels."""
    factor_vec = factor(c("a", "b", "c"))
    result = recode_factor(factor_vec, a="Apple", b="Banana")
    assert result.to_list() == ["Apple", "Banana", "c"]
    assert levels(result) == ["Apple", "Banana", "c"]


def test_recode_factor_numeric():
    """recode_factor replaces factor values and preserves levels."""
    vec = [0, 1, 2, 3, NA]
    result = recode_factor(vec, {0: "z", 1: "y", 2: "x"}, _default="D", _missing="M")
    assert result.to_list() == ["z", "y", "x", "D", "M"]
    assert levels(result) == ["z", "y", "x", "D", "M"]


def test_recode_factor_basic():
    """recode_factor replaces factor values."""
    result = recode_factor(
        ["a", "b", "c"], {"a": "A", "b": "B"}
    )
    assert result.to_list() == ["A", "B", "c"]
    assert levels(result) == ["A", "B", "c"]


def test_recode_factor_no_replacements_error():
    """recode_factor requires at least one replacement."""
    with pytest.raises(ValueError):
        recode_factor([1, 2])

