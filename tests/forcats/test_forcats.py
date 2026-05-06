"""Comprehensive tests for the forcats API in datar-polars.

Tests cover:
- Basic info: fct_count, fct_unique, fct_match
- Level reordering: fct_relevel, fct_reorder, fct_rev
- Level combining: fct_collapse, fct_lump_min, fct_lump_prop
- Value modification: fct_recode, fct_relabel
- Level management: fct_drop, fct_expand, fct_explicit_na
- Construction: fct_inorder, fct_infreq, fct_c, fct_cross

Known bugs marked with @pytest.mark.xfail:
- fct_relevel / fct_reorder / fct_reorder2: idx calculation is inverted,
  computing new_levels.index(l) for l in old_levels instead of
  old_levels.index(l) for l in new_levels.
- fct_match: fails on empty list argument due to is_scalar([]) == True.
"""

import pytest
import polars as pl
from datar import f, options
options(backends=["polars"], allow_conflict_names=True)
from datar.data import ChickWeight
from datar.base import factor, c, factor, rep, table, as_integer, LETTERS
from datar.forcats import (
    fct_count,
    fct_unique,
    fct_match,
    fct_relevel,
    fct_reorder,
    fct_rev,
    fct_collapse,
    fct_lump_lowfreq,
    fct_lump_min,
    fct_lump_prop,
    fct_recode,
    fct_relabel,
    fct_drop,
    fct_expand,
    fct_explicit_na,
    fct_inorder,
    fct_infreq,
    fct_c,
    fct_cross,
    fct_anon,
    fct_inseq,
    fct_shift,
    fct_shuffle,
    fct_other,
    fct_lump,
    fct_reorder2,
    fct_unify,
)
from datar.dplyr import mutate, filter_
from datar_polars.tibble import as_tibble


# ── Helpers ───────────────────────────────────────────────────────────────


def _s(name, values):
    """Create a polars Series with given name and values."""
    return pl.Series(name, values)


def _levels(s):
    """Get the levels (categories) of an Enum/Categorical Series."""
    if s.dtype == pl.Enum:
        cats = s.dtype.categories
        return list(cats.keys()) if hasattr(cats, "keys") else list(cats)
    if s.dtype == pl.Categorical:
        return s.cat.get_categories().to_list()
    return sorted(set(v for v in s.to_list() if v is not None))


# ── fct_count ──────────────────────────────────────────────────────────────


class TestFctCount:
    def test_count_basic(self):
        s = _s("x", ["a", "b", "a", "c", "a"])
        result = fct_count(s)
        assert result.shape == (3, 2)
        assert result["f"].to_list() == ["a", "b", "c"]
        assert result["n"].to_list() == [3, 1, 1]

    def test_count_sorted(self):
        s = _s("x", ["a", "b", "a", "c", "a"])
        result = fct_count(s, sort=True)
        assert result["n"].to_list() == [3, 1, 1]
        assert result["f"].to_list() == ["a", "b", "c"]

    def test_count_prop(self):
        s = _s("x", ["a", "b", "a", "c", "a"])
        result = fct_count(s, prop=True)
        assert result.shape == (3, 3)
        assert "p" in result.collect_schema().names()
        assert result["p"][0] == pytest.approx(0.6)
        assert result["p"][1] == pytest.approx(0.2)
        assert result["p"][2] == pytest.approx(0.2)

    def test_count_with_nulls(self):
        s = _s("x", ["a", None, "b", None, "a"])
        result = fct_count(s)
        assert result.shape == (3, 2)
        n_na = result.filter(pl.col("f").is_null())["n"][0]
        assert n_na == 2


# ── fct_unique ──────────────────────────────────────────────────────────────


class TestFctUnique:
    def test_unique_basic(self):
        s = _s("x", ["b", "a", "b", "c", "a"])
        result = fct_unique(s)
        assert result.to_list() == ["b", "a", "c"]

    def test_unique_single(self):
        s = _s("x", ["a", "a", "a"])
        result = fct_unique(s)
        assert result.to_list() == ["a"]

    def test_unique_with_nulls(self):
        s = _s("x", ["b", None, "a", None, "c"])
        result = fct_unique(s)
        vals = result.to_list()
        assert vals == ["b", "a", "c"]
        assert None not in vals


# ── fct_match ───────────────────────────────────────────────────────────────


class TestFctMatch:
    def test_match_basic(self):
        s = _s("x", ["a", "b", "c", "d"])
        result = fct_match(s, ["a", "c"])
        assert result.tolist() == [True, False, True, False]

    def test_match_scalar(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_match(s, "b")
        assert result.tolist() == [False, True, False]

    def test_match_empty(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_match(s, [])
        assert result.tolist() == [False, False, False]

    def test_match_unknown_levels(self):
        s = _s("x", ["a", "b"])
        with pytest.raises(ValueError):
            fct_match(s, ["z"])


# ── fct_relevel ─────────────────────────────────────────────────────────────


class TestFctRelevel:

    def test_relevel_basic(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_relevel(s, "c", "a")
        assert _levels(result) == ["c", "a", "b"]

    def test_relevel_with_after(self):
        s = _s("x", ["a", "b", "c", "d"])
        result = fct_relevel(s, "d", after=1)
        assert _levels(result) == ["a", "b", "d", "c"]

    def test_relevel_after_end(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_relevel(s, "a", after=2)
        assert _levels(result) == ["b", "c", "a"]

    def test_relevel_unknown_warns(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_relevel(s, "z", "a")
        # "z" is unknown, should be skipped; remaining: a, b, c
        assert set(_levels(result)) == {"a", "b", "c"}

    def test_relevel_callable(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_relevel(s, lambda levs: ["c"])
        assert _levels(result) == ["c", "a", "b"]


# ── fct_reorder ─────────────────────────────────────────────────────────────


class TestFctReorder:
    def test_reorder_basic(self):
        s = _s("x", ["a", "a", "b", "b", "c"])
        vals = _s("y", [10, 20, 5, 5, 30])
        result = fct_reorder(s, vals)
        # Sorted by median: b=5, a=15, c=30
        assert _levels(result) == ["b", "a", "c"]

    def test_reorder_descending(self):
        s = _s("x", ["a", "a", "b", "b", "c"])
        vals = _s("y", [10, 20, 5, 5, 30])
        result = fct_reorder(s, vals, _desc=True)
        assert _levels(result) == ["c", "a", "b"]

    def test_reorder_length_mismatch(self):
        s = _s("x", ["a", "b", "c"])
        vals = _s("y", [1, 2])
        with pytest.raises(ValueError):
            fct_reorder(s, vals)

    def test_reorder_works_with_mutate(self):
        df = pl.DataFrame({"x": ["a", "a", "b", "b", "c"], "y": [10, 20, 5, 5, 30]})
        out = df >> mutate(
            z=fct_reorder(f.x, f.y),
            __backend="polars",
        )
        result = out.get_column("z").dtype
        assert isinstance(result, pl.Enum)
        assert _levels(out.get_column("z")) == ["b", "a", "c"]


# ── fct_rev ─────────────────────────────────────────────────────────────────


class TestFctRev:
    def test_rev_basic(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_rev(s)
        assert _levels(result) == ["c", "b", "a"]

    def test_rev_values_unchanged(self):
        s = _s("x", ["a", "b", "c", "a"])
        result = fct_rev(s)
        assert result.to_list() == ["a", "b", "c", "a"]


# ── fct_collapse ────────────────────────────────────────────────────────────


class TestFctCollapse:
    def test_collapse_basic(self):
        s = _s("x", ["a", "b", "c", "d"])
        result = fct_collapse(s, ab=["a", "b"])
        assert _levels(result) == ["ab", "c", "d"]
        assert result.to_list() == ["ab", "ab", "c", "d"]

    def test_collapse_with_other(self):
        s = _s("x", ["a", "b", "c", "d", "e"])
        result = fct_collapse(s, ab=["a", "b"], other_level="Other")
        assert result.to_list() == ["ab", "ab", "Other", "Other", "Other"]
        # "Other" should be at end
        assert _levels(result)[-1] == "Other"

    def test_collapse_no_mapping(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_collapse(s)
        # No kwargs means no mapping; returns same
        assert set(_levels(result)) == set(_levels(s))
        assert result.to_list() == ["a", "b", "c"]


# ── fct_lump_min ────────────────────────────────────────────────────────────


class TestFctLumpMin:
    def test_lump_min_basic(self):
        s = _s("x", ["a", "a", "a", "b", "c"])
        result = fct_lump_min(s, min_=2)
        # b and c appear only once (< 2)
        assert "Other" in _levels(result)
        assert result.to_list() == ["a", "a", "a", "Other", "Other"]

    def test_lump_min_none_to_lump(self):
        s = _s("x", ["a", "a", "b", "b", "c", "c"])
        result = fct_lump_min(s, min_=1)
        # All appear >= 1, nothing lumped
        assert "Other" not in _levels(result)


# ── fct_lump_prop ───────────────────────────────────────────────────────────


class TestFctLumpProp:
    def test_lump_prop_basic(self):
        s = _s("x", ["a", "a", "a", "a", "a", "b", "c"])
        result = fct_lump_prop(s, prop=0.2)
        # b=1/7≈0.143 <= 0.2, c=1/7≈0.143 <= 0.2 → both lumped
        assert "Other" in _levels(result)
        assert result.to_list() == ["a", "a", "a", "a", "a", "Other", "Other"]

    def test_lump_prop_single_level_unchanged(self):
        # Per R forcats: if only one level would be lumped, don't lump
        s = _s("x", ["a", "a", "a", "a", "b"])
        result = fct_lump_prop(s, prop=0.3)
        # b=1/5=0.2 <= 0.3 but only 1 level → unchanged
        assert "Other" not in _levels(result)

    def test_lump_prop_nothing_to_lump(self):
        s = _s("x", ["a", "a", "b", "b"])
        result = fct_lump_prop(s, prop=0.1)
        # Both appear 50%, > 0.1
        assert "Other" not in _levels(result)


# ── fct_recode ──────────────────────────────────────────────────────────────


class TestFctRecode:
    def test_recode_basic(self):
        s = _s("x", ["a", "b", "c", "a"])
        result = fct_recode(s, {"x": "a", "y": "b"})
        assert result.to_list() == ["x", "y", "c", "x"]
        assert _levels(result) == ["x", "y", "c"]

    def test_recode_missing_unchanged(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_recode(s, x="a")
        assert result.to_list() == ["x", "b", "c"]

    def test_recode_unknown(self):
        s = _s("x", ["a", "b"])
        result = fct_recode(s, z="x")  # "x" not in levels
        assert result.to_list() == ["a", "b"]  # unchanged


# ── fct_relabel ─────────────────────────────────────────────────────────────


class TestFctRelabel:
    def test_relabel_basic(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_relabel(s, lambda levs: [x.upper() for x in levs])
        assert _levels(result) == ["A", "B", "C"]
        assert result.to_list() == ["A", "B", "C"]

    def test_relabel_not_callable(self):
        s = _s("x", ["a", "b"])
        with pytest.raises(TypeError):
            fct_relabel(s, "not_a_function")

    def test_relabel_with_fun(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_relabel(s, lambda levs: [f"level_{i}" for i in range(len(levs))])
        assert _levels(result) == ["level_0", "level_1", "level_2"]
        assert result.to_list() == ["level_0", "level_1", "level_2"]


# ── fct_drop ────────────────────────────────────────────────────────────────


class TestFctDrop:
    def test_drop_unused(self):
        es = pl.Series("x", ["a", "b", "a"], dtype=pl.Enum(["a", "b", "c"]))
        result = fct_drop(es)
        assert _levels(result) == ["a", "b"]

    def test_drop_only_specific(self):
        es = pl.Series("x", ["a", "b", "a"], dtype=pl.Enum(["a", "b", "c", "d"]))
        result = fct_drop(es, only=["d"])
        assert _levels(result) == ["a", "b", "c"]

    def test_drop_nothing_to_drop(self):
        es = pl.Series("x", ["a", "b", "c"], dtype=pl.Enum(["a", "b", "c"]))
        result = fct_drop(es)
        assert _levels(result) == ["a", "b", "c"]


# ── fct_expand ──────────────────────────────────────────────────────────────


class TestFctExpand:
    def test_expand_basic(self):
        s = _s("x", ["a", "b"])
        result = fct_expand(s, "c", "d")
        assert _levels(result) == ["a", "b", "c", "d"]

    def test_expand_existing(self):
        s = _s("x", ["a", "b"])
        result = fct_expand(s, "a")  # already exists
        assert _levels(result) == ["a", "b"]


# ── fct_explicit_na ─────────────────────────────────────────────────────────


class TestFctExplicitNa:
    def test_explicit_na_basic(self):
        s = _s("x", ["a", None, "b", None])
        result = fct_explicit_na(s, na_level="(Missing)")
        assert result.to_list() == ["a", "(Missing)", "b", "(Missing)"]
        assert "(Missing)" in _levels(result)

    def test_explicit_na_no_missing(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_explicit_na(s)
        assert result.to_list() == ["a", "b", "c"]
        assert "(Missing)" not in _levels(result)


# ── fct_inorder ─────────────────────────────────────────────────────────────


class TestFctInorder:
    def test_inorder_basic(self):
        s = _s("x", ["b", "a", "c", "b", "a"])
        result = fct_inorder(s)
        assert _levels(result) == ["b", "a", "c"]

    def test_inorder_preserves_values(self):
        s = _s("x", ["c", "a", "b"])
        result = fct_inorder(s)
        assert result.to_list() == ["c", "a", "b"]


# ── fct_infreq ──────────────────────────────────────────────────────────────


class TestFctInfreq:
    def test_infreq_basic(self):
        s = _s("x", ["a", "b", "b", "c", "a", "a"])
        result = fct_infreq(s)
        # a:3, b:2, c:1 → a, b, c
        assert _levels(result) == ["a", "b", "c"]


# ── fct_c ───────────────────────────────────────────────────────────────────


class TestFctC:
    def test_c_basic(self):
        a = _s("a", ["x", "y"])
        b = _s("b", ["z"])
        result = fct_c(a, b)
        assert result.to_list() == ["x", "y", "z"]
        assert set(_levels(result)) == {"x", "y", "z"}

    def test_c_empty(self):
        result = fct_c()
        assert result.to_list() == []
        assert isinstance(result, pl.Series)

    def test_c_overlapping_levels(self):
        a = _s("a", ["x", "y"])
        b = _s("b", ["y", "z"])
        result = fct_c(a, b)
        assert result.to_list() == ["x", "y", "y", "z"]
        assert set(_levels(result)) == {"x", "y", "z"}

    def test_c_with_factors(self):
        fa = factor("a")
        fb = factor("b")
        fab = factor(c("a", "b"))
        result = fct_c(fa, fb, fab)
        assert result.to_list() == ["a", "b", "a", "b"]
        assert set(_levels(result)) == {"a", "b"}
        assert fa.to_list() == ["a"]
        assert fb.to_list() == ["b"]
        assert fab.to_list() == ["a", "b"]
        assert _levels(fa) == ["a"]
        assert _levels(fb) == ["b"]
        assert _levels(fab) == ["a", "b"]


# ── fct_cross ───────────────────────────────────────────────────────────────


class TestFctCross:
    def test_cross_basic(self):
        a = _s("a", ["x", "y", "z"])
        b = _s("b", ["1", "2", "3"])
        result = fct_cross(a, b)
        # Element-wise: x:1, y:2, z:3. Default keep_empty=False
        # filters to only present combinations.
        assert result.to_list() == ["x:1", "y:2", "z:3"]
        assert set(_levels(result)) == {"x:1", "y:2", "z:3"}

    def test_cross_keep_empty(self):
        a = _s("a", ["x", "y"])
        b = _s("b", ["1", "2"])
        result = fct_cross(a, b, keep_empty=True)
        assert set(_levels(result)) == {"x:1", "x:2", "y:1", "y:2"}

    def test_cross_custom_sep(self):
        a = _s("a", ["x", "y"])
        b = _s("b", ["1", "2"])
        result = fct_cross(a, b, sep="-")
        assert result.to_list() == ["x-1", "y-2"]

    def test_cross_length_mismatch(self):
        a = _s("a", ["x", "y"])
        b = _s("b", ["1", "2", "3"])
        with pytest.raises(ValueError):
            fct_cross(a, b)

    def test_cross_with_nulls(self):
        a = _s("a", ["x", None])
        b = _s("b", ["1", "2"])
        result = fct_cross(a, b)
        assert result.to_list() == ["x:1", None]


# ── fct_inseq ───────────────────────────────────────────────────────────────


class TestFctInseq:
    def test_inseq_numeric(self):
        s = _s("x", ["10", "2", "1", "20"])
        result = fct_inseq(s)
        assert _levels(result) == ["1", "2", "10", "20"]

    def test_inseq_mixed(self):
        s = _s("x", ["10", "abc", "2", "xyz"])
        result = fct_inseq(s)
        # numeric sorted first, then non-numeric
        levs = _levels(result)
        assert levs[0] == "2"
        assert levs[1] == "10"
        assert set(levs[2:]) == {"abc", "xyz"}


# ── fct_anon ────────────────────────────────────────────────────────────────


class TestFctAnon:
    def test_anon_basic(self):
        s = _s("x", ["a", "b", "c", "a"])
        result = fct_anon(s, prefix="L")
        levs = _levels(result)
        assert len(levs) == 3
        assert all(l.startswith("L") for l in levs)

    def test_anon_no_prefix(self):
        s = _s("x", ["x", "y", "z"])
        result = fct_anon(s)
        levs = _levels(result)
        assert len(levs) == 3
        assert all(l.isdigit() for l in levs)


# ── fct_shift ───────────────────────────────────────────────────────────────


class TestFctShift:
    def test_shift_positive(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_shift(s, n=1)
        # shift 1: levels rotated left → b, c, a
        assert _levels(result) == ["b", "c", "a"]

    def test_shift_negative(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_shift(s, n=-1)
        # shift -1 → n=2 → c, a, b
        assert _levels(result) == ["c", "a", "b"]

    def test_shift_noop(self):
        s = _s("x", ["a", "b", "c"])
        result = fct_shift(s, n=3)
        assert _levels(result) == ["a", "b", "c"]


# ── fct_shuffle ─────────────────────────────────────────────────────────────


class TestFctShuffle:
    def test_shuffle_changes_order(self):
        # With many levels, shuffle is very likely to change order
        s = _s("x", [str(i) for i in range(20)])
        original = _levels(s)
        changed = False
        for _ in range(5):
            result = fct_shuffle(s)
            if _levels(result) != original:
                changed = True
                break
        assert changed, "Shuffle did not change level order after 5 attempts"

    def test_shuffle_preserves_values(self):
        s = _s("x", ["a", "b", "c", "a"])
        result = fct_shuffle(s)
        assert sorted(result.to_list()) == ["a", "a", "b", "c"]

    def test_shuffle_with_mutate(self):
        df = pl.DataFrame({"x": ["a", "b", "c", "d"]})
        out = df >> mutate(
            z=fct_shuffle(f.x),
            __backend="polars",
        )
        result = out.get_column("z").dtype
        assert isinstance(result, pl.Enum)
        assert set(_levels(out.get_column("z"))) == {"a", "b", "c", "d"}

    def test_shuffle_chickweight(self):
        chks = (
            ChickWeight
            >> filter_(as_integer(f.Chick) < 10)
            >> mutate(Chick=fct_shuffle(f.Chick))
        )
        assert set(_levels(chks.get_column("Chick"))) == set(str(i) for i in range(1, 10))


# ── fct_other ───────────────────────────────────────────────────────────────


class TestFctOther:
    def test_other_keep(self):
        s = _s("x", ["a", "b", "c", "d"])
        result = fct_other(s, keep=["a", "b"])
        assert result.to_list() == ["a", "b", "Other", "Other"]

    def test_other_drop(self):
        s = _s("x", ["a", "b", "c", "d"])
        result = fct_other(s, drop=["c", "d"])
        assert result.to_list() == ["a", "b", "Other", "Other"]

    def test_other_neither_keep_nor_drop(self):
        s = _s("x", ["a", "b"])
        with pytest.raises(ValueError):
            fct_other(s)

    def test_other_both_keep_and_drop(self):
        s = _s("x", ["a", "b"])
        with pytest.raises(ValueError):
            fct_other(s, keep=["a"], drop=["b"])


# ── fct_lump (generic) ──────────────────────────────────────────────────────


class TestFctLump:
    def test_lump_no_args(self):
        s = _s("x", ["a", "a", "a", "b", "c"])
        result = fct_lump(s)
        # Default: lump_lowfreq
        assert "Other" in _levels(result)

    def test_lump_with_n(self):
        s = _s("x", ["a", "a", "a", "b", "b", "c", "d"])
        result = fct_lump(s, n=2)
        # Keep top 2: a, b; lump c and d
        assert _levels(result) == ["a", "b", "Other"]

    def test_lump_both_n_and_prop(self):
        s = _s("x", ["a", "b"])
        with pytest.raises(ValueError):
            fct_lump(s, n=1, prop=0.5)


# ── fct_reorder2 ────────────────────────────────────────────────────────────


class TestFctReorder2:
    def test_reorder2_basic(self):
        s = _s("x", ["a", "a", "b", "b"])
        x_vals = _s("xval", [1, 2, 3, 4])
        y_vals = _s("yval", [10, 20, 30, 40])
        result = fct_reorder2(s, x_vals, y_vals)
        assert isinstance(result, pl.Series)
        assert len(_levels(result)) == 2

    def test_reorder2_length_mismatch(self):
        s = _s("x", ["a", "b"])
        x_vals = _s("xval", [1, 2, 3])
        y_vals = _s("yval", [10, 20])
        with pytest.raises(ValueError):
            fct_reorder2(s, x_vals, y_vals)


# --- fct_unify -──────────────────────────────────────────────────────────────


class TestFctUnify:
    def test_unify_basic(self):
        fs = [factor("a"), factor("b"), factor(c("a", "b"))]
        result = fct_unify(fs)
        assert result[0].to_list() == ["a"]
        assert result[1].to_list() == ["b"]
        assert result[2].to_list() == ["a", "b"]
        assert set(_levels(result[0])) == {"a", "b"}
        assert set(_levels(result[1])) == {"a", "b"}
        assert set(_levels(result[2])) == {"a", "b"}
        # Original factors should be unchanged
        assert fs[0].to_list() == ["a"]
        assert fs[1].to_list() == ["b"]
        assert fs[2].to_list() == ["a", "b"]
        assert _levels(fs[0]) == ["a"]
        assert _levels(fs[1]) == ["b"]
        assert _levels(fs[2]) == ["a", "b"]

    def test_unify_empty(self):
        result = fct_unify([])
        assert result == []


# --- fct_lump_lowfreq -────────────────────────────────────────────────────────────


class TestFctLumpLowfreq:
    def test_lump_lowfreq_basic(self):
        x = factor(rep(LETTERS[:9], times = c(40, 10, 5, 27, 1, 1, 1, 1, 1)))
        result = fct_lump_lowfreq(x)
        tb = table(result)
        assert set(tb[:, 0].to_list()) == {"A", "D", "Other"}
        # Other = B(10) + C(5) + E(1) + F(1) + G(1) + H(1) + I(1) = 20
        assert set(tb[:, 1].to_list()) == {40, 27, 20}
