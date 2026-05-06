import pytest
import polars as pl

from datar import f
from datar.base import NA, c
from datar.dplyr import (
    group_by,
    if_else,
    case_when,
    mutate,
    pull,
    case_match,
    group_vars,
)
from datar.tibble import tibble
from ..conftest import assert_iterable_equal, assert_equal


def test_scalar_true_false_are_vectorized():
    x = c(True, True, False, False)
    out = if_else(x, 1, 2)
    assert list(out) == [1, 1, 2, 2]

    x = pl.Series(c(True, True, False, False))
    out = if_else(x, 1, 2)
    assert list(out) == [1, 1, 2, 2]


def test_vector_true_false_ok():
    x = [-1, 0, 1]
    out = if_else([v < 0 for v in x], x, 0)
    assert list(out) == [-1, 0, 0]

    out = if_else([v > 0 for v in x], x, 0)
    assert list(out) == [0, 0, 1]


def test_missing_values_are_missing():
    out = if_else(c(True, NA, False), -1, 1)
    assert_iterable_equal(out, [-1, 1, 1])

    out = if_else(c(True, NA, False), -1, 1, 0)
    assert_iterable_equal(out, [-1, 0, 1])


def test_if_else_errors():
    out = if_else(range(1, 11), 1, 2)
    assert list(out) == [1.0] * 10

    data = [1, 2, 3]
    with pytest.raises(ValueError, match="size"):
        if_else([v < 2 for v in data], [1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="size"):
        if_else([v < 2 for v in data], [1, 2, 3], [1, 2])


# case_when ------------------
def test_matches_values_in_order():
    x = [1, 2, 3]
    out = case_when(
        [v <= 1 for v in x], 1,
        [v <= 2 for v in x], 2,
        [v <= 3 for v in x], 3,
    )
    assert list(out) == [1, 2, 3]


def test_unmatched_gets_missing_value():
    x = [1, 2, 3]
    out = case_when([v <= 1 for v in x], 1, [v <= 2 for v in x], 2)
    # Unmatched gets None (polars default)
    assert list(out) == [1, 2, None]


def test_missing_values_can_be_replaced():
    x = [1, 2, 3, None]
    out = case_when(
        [v is not None and v <= 1 for v in x], 1,
        [v is not None and v <= 2 for v in x], 2,
        [v is None for v in x], 0,
    )
    assert list(out) == [1, 2, None, 0]


def test_na_conditions():
    out = case_when([True, False, None], [1, 2, 3], True, 4)
    assert list(out) == [1, 4, 4]


def test_atomic_conditions():
    out = case_when(True, [1, 2, 3], False, [4, 5, 6])
    assert list(out) == [1, 2, 3]

    out = case_when(None, [1, 2, 3], True, [4, 5, 6])
    assert list(out) == [4, 5, 6]


def test_0len_conditions_and_values():
    # Polars when/then with empty lists returns empty
    out = case_when(True, [], False, [])
    assert list(out) == []


def test_inside_mutate():
    from datar.data import mtcars

    out = (
        mtcars.head(4)
        >> mutate(out=case_when(f.cyl == 4, 1, f["am"] == 1, 2, True, 0))
        >> pull(to="list")
    )
    assert out == [2, 2, 1, 0]


def test_errors():
    # 5 flat args: unbalanced pairs
    with pytest.raises(ValueError, match="paired"):
        case_when([1, 2, 3], [1, 2], [3, 4], [5, 6], [7, 8])
    with pytest.raises(TypeError):
        case_when()
    with pytest.raises(TypeError):
        case_when("a")


# case_match ------------------
def test_case_match_lhs_can_match_multiple_values():
    assert_iterable_equal(case_match(1, [1, 2], "x"), ["x"])


def test_case_match_lhs_can_match_na():
    assert_iterable_equal(case_match(None, None, "x"), ["x"])


def test_case_match_rhs_recycling():
    x = [1, 2, 3]
    assert_iterable_equal(case_match(x, [1, 3], [v * 2 for v in x]), [2, None, 6])


def test_case_match_requires_at_least_one_condition():
    with pytest.raises(ValueError):
        case_match(1)


def test_case_match_default_works():
    assert_iterable_equal(case_match(1, 3, 1, _default=2), [2])
    assert_iterable_equal(case_match([1, 2, 3, 4, 5], 6, 1, _default=2), [2] * 5)
    assert_iterable_equal(
        case_match([1, 2, 3, 4, 5], 6, [1, 2, 3, 4, 5], _default=[2, 3, 4, 5, 6]),
        [2, 3, 4, 5, 6],
    )


def test_case_match_dtypes():
    assert_iterable_equal(case_match(1, 1, 1.1, _dtypes=int), [1])


def test_case_match_on_grouped():
    gdf = tibble(g=[1, 2], x=[1, 2]) >> group_by(f.g)
    out = gdf >> mutate(y=case_match(f.x, 1, 2))
    assert_iterable_equal(group_vars(out), ["g"])
    # Collect lazy frame before indexing
    collected = out.collect() if hasattr(out, "collect") else out
    vals = list(collected["y"])
    assert vals == [2, None]
