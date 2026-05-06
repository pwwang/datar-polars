"""Tests for base set operations: intersect, union, setdiff, setequal."""

import polars as pl
from datar import f
from datar.base import intersect, union, setdiff, setequal
from datar.dplyr import mutate
from datar_polars.tibble import as_tibble


def _df(data: dict):
    """Create a polars Tibble from a dict."""
    return as_tibble(pl.DataFrame(data))


# ── intersect ───────────────────────────────────────────────────────────


class TestIntersect:
    def test_intersect_scalar_lists(self):
        assert intersect([1, 2, 3], [3, 4]) == [3]

    def test_intersect_scalar_no_overlap(self):
        assert intersect([1, 2, 3], [4, 5]) == []

    def test_intersect_series(self):
        s1 = pl.Series([1, 2, 3, 2])
        s2 = pl.Series([3, 4, 5])
        result = intersect(s1, s2)
        assert result == [3]

    def test_intersect_mutate(self):
        df = _df({"a": [[1, 2, 3], [4, 5, 6]], "b": [[3, 4, 5], [4, 6, 7]]})
        out = mutate(
            df, c=intersect(f.a, f.b)
        )
        assert out.collect().get_column("c").to_list() == [[3], [4, 6]]


# ── union ───────────────────────────────────────────────────────────────


class TestUnion:
    def test_union_scalar_lists(self):
        assert union([1, 2, 3], [3, 4]) == [1, 2, 3, 4]

    def test_union_scalar_duplicates(self):
        assert union([1, 1, 2], [2, 3]) == [1, 2, 3]

    def test_union_series(self):
        s1 = pl.Series([1, 2, 3])
        s2 = pl.Series([3, 4, 5])
        result = union(s1, s2)
        assert result == [1, 2, 3, 4, 5]

    def test_union_mutate(self):
        df = _df({"a": [[1, 2, 3], [4, 5, 6]], "b": [[3, 4, 5], [4, 6, 7]]})
        out = mutate(
            df, c=union(f.a, f.b)
        )
        assert out.collect().get_column("c").to_list() == [
            [1, 2, 3, 4, 5],
            [4, 5, 6, 7],
        ]


# ── setdiff ─────────────────────────────────────────────────────────────


class TestSetdiff:
    def test_setdiff_scalar_lists(self):
        assert setdiff([1, 2, 3], [3, 4]) == [1, 2]

    def test_setdiff_scalar_no_overlap(self):
        assert setdiff([1, 2], [3, 4]) == [1, 2]

    def test_setdiff_series(self):
        s1 = pl.Series([1, 2, 3])
        s2 = pl.Series([3, 4])
        result = setdiff(s1, s2)
        assert result == [1, 2]

    def test_setdiff_mutate(self):
        df = _df({"a": [[1, 2, 3], [4, 5, 6]], "b": [[3, 4, 5], [4, 6, 7]]})
        out = mutate(
            df, c=setdiff(f.a, f.b)
        )
        assert out.collect().get_column("c").to_list() == [[1, 2], [5]]


# ── setequal ────────────────────────────────────────────────────────────


class TestSetequal:
    def test_setequal_scalar_true(self):
        assert setequal([1, 2, 3], [3, 2, 1]) is True

    def test_setequal_scalar_false(self):
        assert setequal([1, 2], [1, 2, 3]) is False

    def test_setequal_series_true(self):
        s1 = pl.Series([1, 2, 3])
        s2 = pl.Series([3, 1, 2])
        assert setequal(s1, s2) is True

    def test_setequal_series_false(self):
        s1 = pl.Series([1, 2, 3])
        s2 = pl.Series([3, 4, 5])
        assert setequal(s1, s2) is False

    def test_setequal_mutate(self):
        df = _df({"a": [[1, 2, 3], [4, 5, 6]], "b": [[3, 2, 1], [6, 4, 5]]})
        out = mutate(
            df, c=setequal(f.a, f.b)
        )
        assert out.collect().get_column("c").to_list() == [True, True]
