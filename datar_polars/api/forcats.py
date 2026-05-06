"""Provides forcats verbs for the polars backend.

All forcats functions operate on factor-like data (pl.Enum or string Series)
and return Series with adjusted categories/levels.
"""

from __future__ import annotations

import contextvars
import itertools
import logging
import random
from typing import Any, Callable, Iterable, Optional, Sequence, Union, cast

import numpy as np
import polars as pl

from datar.core.utils import logger
from datar.apis.forcats import (
    fct_anon,
    fct_c,
    fct_collapse,
    fct_count,
    fct_cross,
    fct_drop,
    fct_expand,
    fct_explicit_na,
    fct_infreq,
    fct_inorder,
    fct_inseq,
    fct_lump,
    fct_lump_lowfreq,
    fct_lump_min,
    fct_lump_n,
    fct_lump_prop,
    fct_match,
    fct_other,
    fct_recode,
    fct_relabel,
    fct_relevel,
    fct_reorder,
    fct_reorder2,
    fct_rev,
    fct_shift,
    fct_shuffle,
    fct_unique,
    fct_unify,
    first2,
    last2,
    lvls_expand,
    lvls_reorder,
    lvls_revalue,
    lvls_union,
)

from ..contexts import Context
from ..common import is_scalar, setdiff, union, intersect, is_null

# Context variable to pass the evaluating LazyFrame to Expr-based forcats
# implementations. Set by verb implementations before calling evaluate_expr().
_EVAL_DATA: contextvars.ContextVar = contextvars.ContextVar(
    '_EVAL_DATA', default=None
)

# ---- Types ---------------------------------------------------------------

# ForcatsRegType: types accepted by forcats verbs (materialized)
ForcatsRegType = (pl.Series, list, tuple, np.ndarray)

# ---- Utility functions ---------------------------------------------------


def _to_series(x: Any, name: str = "") -> pl.Series:
    """Convert input to a polars Series (materialized)."""
    if isinstance(x, pl.Series):
        return x
    if isinstance(x, pl.Expr):
        raise TypeError("pl.Expr not supported — materialize first")
    if is_scalar(x):
        x = [x]
    return pl.Series(name, list(x))


def _check_factor(_f: Any) -> pl.Series:
    """Ensure input is a pl.Series treated as a factor.

    polars doesn't have a native R-style factor, but pl.Enum and
    pl.Categorical are close equivalents.  For datar compatibility,
    we convert plain lists/arrays to pl.Enum and track levels
    separately.
    """
    s = _to_series(_f, name=getattr(_f, "name", ""))
    # If it's already an Enum, return as-is
    if s.dtype == pl.Enum or getattr(s.dtype, "base_type", None)() == pl.Enum:
        return s
    # If it's Categorical, convert to Enum with current categories
    if s.dtype == pl.Categorical:
        cats = s.cat.get_categories().to_list()
        return s.cast(pl.Enum(cats))
    # If it's a string, convert to Enum with observed values as categories
    if s.dtype in (pl.Utf8, pl.String):
        # Get unique non-null values in order of first appearance
        vals = s.drop_nulls()
        seen = set()
        cats = []
        for v in vals.to_list():
            if v not in seen:
                seen.add(v)
                cats.append(v)
        if cats:
            return s.cast(pl.Enum(cats))
        return s
    # For other types (numeric, etc.), convert to string and build Enum
    str_vals = [str(v) if v is not None else None for v in s.to_list()]
    seen = set()
    cats = []
    for v in str_vals:
        if v is not None and v not in seen:
            seen.add(v)
            cats.append(v)
    if cats:
        return pl.Series(s.name, str_vals, dtype=pl.Enum(cats))
    return pl.Series(s.name, str_vals)


# ---- Expr helpers (for fct_* inside mutate/summarise) --------------------


def _check_eval_data(fn_name: str):
    """Get the current evaluating LazyFrame from the context variable."""
    data = _EVAL_DATA.get()
    if data is None:
        raise ValueError(
            f"{fn_name} with pl.Expr requires an evaluating data context. "
            "Use inside mutate(), summarise(), or another dplyr verb."
        )
    return data


def _resolve_factor_expr(_f: pl.Expr, data) -> pl.Series:
    """Resolve a pl.Expr factor reference to a materialized Enum Series."""
    f_name = _f.meta.output_name()
    if f_name is None:
        raise ValueError(
            "Forcats functions require a simple column reference (e.g., f.x), "
            "not a computed expression."
        )
    subset = data.select([f_name]).collect()
    return _check_factor(subset.get_column(f_name))


def _finish_expr(_f: pl.Expr, result: pl.Series) -> pl.Expr:
    """Cast the original expression to match the result's Enum levels."""
    new_levels = _get_levels(result)
    return _f.cast(pl.Utf8).cast(pl.Enum(new_levels))


def _get_levels(_f: pl.Series) -> list:
    """Get the levels (categories) of a factor-like Series."""
    if _f.dtype == pl.Enum:
        cats = _f.dtype.categories
        return list(cats.keys()) if hasattr(cats, "keys") else list(cats)
    if _f.dtype == pl.Categorical:
        return _f.cat.get_categories().to_list()
    # For string/non-enum, return unique values in appearance order
    vals = _f.drop_nulls()
    seen = set()
    cats = []
    for v in vals.to_list():
        if v not in seen:
            seen.add(v)
            cats.append(v)
    return cats


def _get_nlevels(_f: pl.Series) -> int:
    """Get the number of levels."""
    return len(_get_levels(_f))


def _make_enum(values: pl.Series, categories: list, name: str = "") -> pl.Series:
    """Create or cast to pl.Enum with given categories."""
    return pl.Series(
        name or values.name,
        values.to_list(),
        dtype=pl.Enum(categories),
    )


def _recode_values(
    values: pl.Series, mapping: dict, name: str = ""
) -> pl.Series:
    """Recode values in a Series using a mapping dict (old -> new)."""
    data = values.to_list()
    recoded = [mapping.get(v, v) for v in data]
    return pl.Series(name or values.name, recoded)


# ---- lvls_reorder --------------------------------------------------------


@lvls_reorder.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _lvls_reorder(_f, idx, ordered: Optional[bool] = None) -> pl.Series:
    """Reorder factor levels by index."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    nlevs = len(levs)

    if not isinstance(idx, (list, tuple, np.ndarray)):
        idx = [idx]

    idx_list = list(idx)
    if not all(isinstance(i, (int, np.integer)) for i in idx_list):
        raise ValueError("`idx` must be integers")

    if len(idx_list) != nlevs:
        raise ValueError("`idx` must contain one integer for each level of `f`")

    if set(idx_list) != set(range(nlevs)):
        raise ValueError("`idx` must contain one integer for each level of `f`")

    new_levels = [levs[i] for i in idx_list]
    return _make_enum(_f, new_levels)


# ---- lvls_revalue --------------------------------------------------------


@lvls_revalue.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _lvls_revalue(_f, new_levels) -> pl.Series:
    """Change the values of existing levels."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    new_levs = list(new_levels)

    if len(new_levs) != len(levs):
        raise ValueError(
            "`new_levels` must be the same length as `levels(f)`: "
            f"expected {len(levs)} new levels, got {len(new_levs)}."
        )

    mapping = dict(zip(levs, new_levs))
    recoded = _recode_values(_f, mapping)
    # Use unique new levels as categories (maintaining order)
    seen = set()
    cats = []
    for nl in new_levs:
        if nl is not None and nl not in seen:
            seen.add(nl)
            cats.append(nl)
    return _make_enum(recoded, cats)


# ---- lvls_expand ---------------------------------------------------------


@lvls_expand.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _lvls_expand(_f, new_levels) -> pl.Series:
    """Expand the set of levels."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    new_levs = list(new_levels)

    missing = setdiff(levs, new_levs)
    if len(missing) > 0:
        raise ValueError(
            f"Must include all existing levels. Missing: {missing}"
        )

    return _make_enum(_f, new_levs)


# ---- lvls_union ----------------------------------------------------------


@lvls_union.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _lvls_union(fs) -> list:
    """Find all levels in a list of factors."""
    out = []
    for fct in fs:
        fct = _check_factor(fct)
        levs = _get_levels(fct)
        out = union(out, levs)
    return out


# ---- Helper: lvls_seq ----------------------------------------------------


def _lvls_seq(_f: pl.Series) -> list:
    """Get index sequence 0..nlevels-1."""
    return list(range(_get_nlevels(_f)))


# ---- fct_relevel ---------------------------------------------------------


@fct_relevel.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_relevel(_f, *lvls: Any, after: Optional[int] = None) -> pl.Series:
    """Reorder factor levels by hand."""
    _f = _check_factor(_f)
    old_levels = _get_levels(_f)

    if len(lvls) == 1 and callable(lvls[0]):
        first_levels = lvls[0](old_levels)
        if not isinstance(first_levels, (list, tuple, np.ndarray, pl.Series)):
            first_levels = list(first_levels)
        first_levels = list(first_levels)
    else:
        first_levels = list(lvls)

    unknown = setdiff(first_levels, old_levels)
    if unknown:
        logger.warning("[fct_relevel] Unknown levels in `_f`: %s", unknown)
        first_levels = intersect(first_levels, old_levels)

    remaining = setdiff(old_levels, first_levels)

    if after is not None:
        # Insert first_levels after the given position in old_levels
        first_set = set(first_levels)
        n_before = sum(
            1 for l in old_levels[: after + 1] if l not in first_set
        )
        new_levels = remaining[:n_before] + first_levels + remaining[n_before:]
    else:
        new_levels = first_levels + remaining

    idx = [old_levels.index(l) for l in new_levels]
    return _lvls_reorder(_f, idx)


@fct_relevel.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_relevel_expr(_f, *lvls, after=None):
    data = _check_eval_data("fct_relevel")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_relevel(f_series, *lvls, after=after)
    return _finish_expr(_f, result)


# ---- fct_inorder ---------------------------------------------------------


@fct_inorder.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_inorder(_f, ordered: Optional[bool] = None) -> pl.Series:
    """Reorder factor levels by first appearance."""
    _f = _check_factor(_f)
    vals = _f.drop_nulls().to_list()
    seen = set()
    new_levels = []
    for v in vals:
        if v not in seen:
            seen.add(v)
            new_levels.append(v)
    return _make_enum(_f, new_levels)


@fct_inorder.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_inorder_expr(_f, ordered=None):
    data = _check_eval_data("fct_inorder")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_inorder(f_series, ordered=ordered)
    return _finish_expr(_f, result)


# ---- fct_infreq ----------------------------------------------------------


@fct_infreq.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_infreq(_f, ordered: Optional[bool] = None) -> pl.Series:
    """Reorder factor levels by frequency (most frequent first)."""
    _f = _check_factor(_f)
    vals = [v for v in _f.to_list() if v is not None]
    counts = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    # Sort by count descending, then by first appearance for ties
    seen = set()
    appearance_order = {}
    for v in vals:
        if v not in seen:
            seen.add(v)
            appearance_order[v] = len(appearance_order)
    new_levels = sorted(
        counts.keys(),
        key=lambda k: (-counts[k], appearance_order[k]),
    )
    return _make_enum(_f, new_levels)


@fct_infreq.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_infreq_expr(_f, ordered=None):
    data = _check_eval_data("fct_infreq")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_infreq(f_series, ordered=ordered)
    return _finish_expr(_f, result)


# ---- fct_inseq -----------------------------------------------------------


@fct_inseq.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_inseq(_f, ordered: Optional[bool] = None) -> pl.Series:
    """Reorder factor levels by numeric order."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    num_levels = []
    for lev in levs:
        if lev is None:
            num_levels.append(float("inf"))
        else:
            try:
                num_levels.append(float(lev))
            except (ValueError, TypeError):
                num_levels.append(float("inf"))

    if all(np.isinf(nl) for nl in num_levels):
        raise ValueError(
            "At least one existing level must be coercible to numeric."
        )

    # Sort: numeric values first (ascending), non-numeric at end
    paired = list(zip(levs, num_levels))
    paired.sort(
        key=lambda p: (np.isinf(p[1]), p[1], p[0])
    )
    new_levels = [p[0] for p in paired]
    return _make_enum(_f, new_levels)


@fct_inseq.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_inseq_expr(_f, ordered=None):
    data = _check_eval_data("fct_inseq")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_inseq(f_series, ordered=ordered)
    return _finish_expr(_f, result)


# ---- fct_rev -------------------------------------------------------------


@fct_rev.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_rev(_f) -> pl.Series:
    """Reverse order of factor levels."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    return _lvls_reorder(_f, list(reversed(range(len(levs)))))


@fct_rev.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_rev_expr(_f):
    data = _check_eval_data("fct_rev")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_rev(f_series)
    return _finish_expr(_f, result)


# ---- fct_shuffle ---------------------------------------------------------


@fct_shuffle.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_shuffle(_f) -> pl.Series:
    """Randomly permute factor levels."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    n = len(levs)
    indices = list(range(n))
    random.shuffle(indices)
    return _lvls_reorder(_f, indices)


@fct_shuffle.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_shuffle_expr(_f: pl.Expr) -> pl.Expr:
    """fct_shuffle for pl.Expr inputs (used inside mutate/summarise)."""
    data = _EVAL_DATA.get()
    if data is None:
        raise ValueError(
            "fct_shuffle with pl.Expr requires an evaluating data context. "
            "Use inside mutate(), summarise(), or another dplyr verb."
        )
    f_name = _f.meta.output_name()
    if f_name is None:
        raise ValueError(
            "fct_shuffle requires a simple column reference (e.g., f.x), "
            "not a computed expression."
        )
    subset = data.select([f_name]).collect()
    f_series = _check_factor(subset.get_column(f_name))
    levs = _get_levels(f_series)
    indices = list(range(len(levs)))
    random.shuffle(indices)
    new_levels = [levs[i] for i in indices]
    return _f.cast(pl.Utf8).cast(pl.Enum(new_levels))


# ---- fct_shift -----------------------------------------------------------


@fct_shift.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_shift(_f, n: int = 1) -> pl.Series:
    """Shift factor levels to left or right, wrapping around at end."""
    _f = _check_factor(_f)
    nlvls = _get_nlevels(_f)
    if nlvls == 0:
        return _f
    shift = n % nlvls
    new_order = list(range(shift, nlvls)) + list(range(shift))
    return _lvls_reorder(_f, new_order)


@fct_shift.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_shift_expr(_f, n=1):
    data = _check_eval_data("fct_shift")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_shift(f_series, n=n)
    return _finish_expr(_f, result)


# ---- first2, last2 -------------------------------------------------------


@first2.register(object, backend="polars")
def _first2(_x, _y) -> Any:
    """Find the first element of `_y` ordered by `_x`."""
    x = list(_x)
    y = list(_y)
    idx = sorted(range(len(x)), key=lambda i: (x[i] is None, x[i]))
    return y[idx[0]]


@last2.register(object, backend="polars")
def _last2(_x, _y) -> Any:
    """Find the last element of `_y` ordered by `_x`."""
    x = list(_x)
    y = list(_y)
    idx = sorted(range(len(x)), key=lambda i: (x[i] is None, x[i]))
    return y[idx[-1]]


# ---- fct_reorder ---------------------------------------------------------


@fct_reorder.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_reorder(
    _f,
    _x,
    *args: Any,
    _fun=None,
    _desc: bool = False,
    **kwargs: Any,
) -> pl.Series:
    """Reorder factor levels by sorting along another variable."""
    _f = _check_factor(_f)
    if is_scalar(_x):
        _x = [_x]
    _x = list(_x)

    if len(_f) != len(_x):
        raise ValueError("Unmatched length between `_x` and `_f`.")

    if _fun is None:
        _fun = _default_median

    # Group _x by _f levels and compute summary
    levs = _get_levels(_f)
    vals = _f.to_list()
    summary = {}
    for lev in levs:
        group_vals = [x for v, x in zip(vals, _x) if v == lev and x is not None]
        if group_vals:
            summary[lev] = _fun(group_vals)
        else:
            summary[lev] = float("inf") if _desc else float("-inf")

    sorted_levs = sorted(summary, key=summary.get, reverse=_desc)
    idx = [levs.index(l) for l in sorted_levs]
    return _lvls_reorder(_f, idx)


def _default_median(x):
    """Default median helper."""
    s = sorted(x)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


@fct_reorder.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_reorder_expr(
    _f: pl.Expr,
    _x: pl.Expr,
    *args: Any,
    _fun=None,
    _desc: bool = False,
    **kwargs: Any,
) -> pl.Expr:
    """fct_reorder for pl.Expr inputs (used inside mutate/summarise)."""
    data = _EVAL_DATA.get()
    if data is None:
        raise ValueError(
            "fct_reorder with pl.Expr requires an evaluating data context. "
            "Use inside mutate(), summarise(), or another dplyr verb."
        )

    f_name = _f.meta.output_name()
    x_name = _x.meta.output_name()

    if f_name is None or x_name is None:
        raise ValueError(
            "fct_reorder requires simple column references (e.g., f.x, f.y), "
            "not computed expressions."
        )

    subset = data.select([f_name, x_name]).collect()
    f_series = _check_factor(subset.get_column(f_name))
    x_series = subset.get_column(x_name)

    result = _fct_reorder(
        f_series, x_series, *args, _fun=_fun, _desc=_desc, **kwargs
    )
    new_levels = _get_levels(result)
    return _f.cast(pl.Utf8).cast(pl.Enum(new_levels))


# ---- fct_reorder2 --------------------------------------------------------


@fct_reorder2.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_reorder2(
    _f,
    _x,
    *args: Any,
    _fun=None,
    _desc: bool = True,
    **kwargs: Any,
) -> pl.Series:
    """Reorder factor levels by sorting along two variables (default: last2)."""
    if _fun is None:
        _fun = last2

    _f = _check_factor(_f)
    if is_scalar(_x):
        _x = [_x]

    # args[0] is _y for fct_reorder2
    _y = args[0] if args else []
    args = args[1:] if len(args) > 0 else ()

    if is_scalar(_y):
        _y = [_y]
    _x = list(_x)
    _y = list(_y)

    if len(_f) != len(_x) or len(_f) != len(_y):
        raise ValueError("Unmatched length between `_x` and `_f`.")

    levs = _get_levels(_f)
    vals = _f.to_list()
    summary = {}
    for lev in levs:
        group_x = [x for v, x in zip(vals, _x) if v == lev]
        group_y = [y for v, y in zip(vals, _y) if v == lev]
        if group_x and group_y:
            summary[lev] = _fun(group_x, group_y)
        else:
            summary[lev] = float("inf") if _desc else float("-inf")

    sorted_levs = sorted(summary, key=summary.get, reverse=_desc)
    idx = [levs.index(l) for l in sorted_levs]
    return _lvls_reorder(_f, idx)


# ---- fct_anon ------------------------------------------------------------


@fct_anon.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_anon(_f, prefix: str = "") -> pl.Series:
    """Anonymise factor levels."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    nlvls = len(levs)
    ndigits = max(1, len(str(nlvls)))
    new_lvls = [f"{prefix}{str(i).rjust(ndigits, '0')}" for i in range(nlvls)]
    # Shuffle the labels
    shuffled = new_lvls.copy()
    random.shuffle(shuffled)
    mapping = dict(zip(levs, shuffled))
    recoded = _recode_values(_f, mapping)
    # Reorder levels to match the numeric order
    return _make_enum(recoded, new_lvls)


# ---- fct_recode ----------------------------------------------------------


@fct_recode.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_recode(_f, *args, **kwargs) -> pl.Series:
    """Change factor levels by hand."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)

    # Build recodings: new -> old
    recodings = {}
    for arg in args:
        if not isinstance(arg, dict):
            raise ValueError("`*args` have to be all mappings.")
        recodings.update(arg)
    recodings.update(kwargs)

    # Build mapping: old -> new
    mapping = {}
    unknown = set()
    for new_val, old_vals in recodings.items():
        if isinstance(old_vals, (np.ndarray, set, list)):
            for ov in old_vals:
                if ov not in levs:
                    unknown.add(ov)
                else:
                    mapping[ov] = new_val
        else:
            if old_vals not in levs:
                unknown.add(old_vals)
            else:
                mapping[old_vals] = new_val

    if unknown:
        logger.warning("[fct_recode] Unknown levels in `_f`: %s", unknown)

    recoded = _recode_values(_f, mapping)
    # Calculate new levels
    new_levs_dedup = set()
    new_levels = []
    for lev in levs:
        new_lev = mapping.get(lev, lev)
        if new_lev not in new_levs_dedup and new_lev is not None:
            new_levs_dedup.add(new_lev)
            new_levels.append(new_lev)
    return _make_enum(recoded, new_levels)


# ---- fct_collapse --------------------------------------------------------


@fct_collapse.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_collapse(_f, other_level: Any = None, **kwargs) -> pl.Series:
    """Collapse factor levels into manually defined groups."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)

    all_collapsed = set()
    for sublevs in kwargs.values():
        all_collapsed.update(sublevs)

    if other_level is not None:
        remaining = set(levs) - all_collapsed
        kwargs[other_level] = list(remaining)

    # Build recode mapping: old -> new
    mapping = {}
    for new_val, old_vals in kwargs.items():
        for ov in old_vals:
            mapping[ov] = new_val

    if not mapping:
        return _f

    recoded = _recode_values(_f, mapping)
    # Build new levels: original order but with collapsed groups
    new_levs_dedup = set()
    new_levels = []
    for lev in levs:
        new_lev = mapping.get(lev, lev)
        if new_lev not in new_levs_dedup and new_lev is not None:
            new_levs_dedup.add(new_lev)
            new_levels.append(new_lev)

    if other_level is not None and other_level in new_levels:
        # Move "other" to end
        new_levels.remove(other_level)
        new_levels.append(other_level)

    return _make_enum(recoded, new_levels)


# ---- Helpers for lump functions ------------------------------------------


def _check_calc_levels(_f, w=None):
    """Check and compute level counts."""
    _f = _check_factor(_f)
    vals = _f.to_list()

    if w is not None:
        w = list(w)
        if len(w) != len(vals):
            raise ValueError(
                f"`w` must be the same length as `f` ({len(vals)}), "
                f"not length {len(w)}."
            )
        for weight in w:
            if weight < 0 or weight is None:
                raise ValueError(
                    f"All `w` must be non-negative and non-missing, got {weight}."
                )

    levs = _get_levels(_f)
    if w is None:
        counts = {lev: 0 for lev in levs}
        for v in vals:
            if v is not None:
                counts[v] += 1
        total = sum(counts.values())
    else:
        counts = {lev: 0.0 for lev in levs}
        for v, weight in zip(vals, w):
            if v is not None and weight is not None:
                counts[v] += weight
        total = sum(counts.values())

    counts_arr = np.array([counts.get(lev, 0) for lev in levs], dtype=float)
    return {"_f": _f, "count": counts_arr, "total": total, "levs": levs}


def _lump_cutoff(x) -> int:
    """Lump smallest groups ensuring 'other' is still the smallest."""
    left = sum(x)
    for i, elem in enumerate(x):
        left -= elem
        if elem > left:
            return i + 1
    return len(x)


def _in_smallest(x) -> np.ndarray:
    """Check which elements are in the smallest group."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(-x, kind="stable")
    idx = _lump_cutoff(x[order])
    to_lump = np.zeros(len(x), dtype=bool)
    to_lump[order[idx:]] = True
    return to_lump


# ---- fct_lump ------------------------------------------------------------


@fct_lump.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_lump(
    _f,
    n=None,
    prop=None,
    w=None,
    other_level: Any = "Other",
    ties_method: str = "min",
) -> pl.Series:
    """Lump together factor levels into 'other'."""
    _check_calc_levels(_f, w)

    if n is None and prop is None:
        return _fct_lump_lowfreq(_f, other_level=other_level)
    if prop is None:
        return _fct_lump_n(
            _f, n=n, w=w, other_level=other_level, ties_method=ties_method
        )
    if n is None:
        return _fct_lump_prop(_f, prop=prop, w=w, other_level=other_level)
    raise ValueError("Must supply only one of `n` and `prop`")


# ---- fct_lump_min --------------------------------------------------------


@fct_lump_min.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_lump_min(_f, min_: int, w=None, other_level: Any = "Other") -> pl.Series:
    """Lumps levels that appear fewer than `min_` times."""
    calcs = _check_calc_levels(_f, w)
    _f_data = calcs["_f"]
    levs = calcs["levs"]
    counts = calcs["count"]

    if min_ < 0:
        raise ValueError("`min_` must be a positive number.")

    new_levels_list = [
        other_level if counts[i] < min_ else levs[i]
        for i in range(len(levs))
    ]

    if other_level not in new_levels_list:
        return _f_data

    mapping = dict(zip(levs, new_levels_list))
    recoded = _recode_values(_f_data, mapping)

    new_levs_dedup = set()
    new_levels = []
    for nl in new_levels_list:
        if nl not in new_levs_dedup and nl is not None:
            new_levs_dedup.add(nl)
            new_levels.append(nl)

    # Move "other" to end
    if other_level in new_levels:
        new_levels.remove(other_level)
        new_levels.append(other_level)

    return _make_enum(recoded, new_levels)


# ---- fct_lump_prop -------------------------------------------------------


@fct_lump_prop.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_lump_prop(_f, prop, w=None, other_level: Any = "Other") -> pl.Series:
    """Lumps levels that appear in fewer `prop * n` times."""
    calcs = _check_calc_levels(_f, w)
    _f_data = calcs["_f"]
    levs = calcs["levs"]
    counts = calcs["count"]
    total = calcs["total"]

    prop_n = counts / total if total > 0 else counts

    if prop >= 0:
        new_levels_list = [
            other_level if prop_n[i] <= prop else levs[i]
            for i in range(len(levs))
        ]
        if other_level in new_levels_list and sum(prop_n <= prop) <= 1:
            return _f_data
    else:
        neg_prop = -prop
        new_levels_list = [
            other_level if prop_n[i] <= neg_prop else levs[i]
            for i in range(len(levs))
        ]

    if other_level not in new_levels_list:
        return _f_data

    mapping = dict(zip(levs, new_levels_list))
    recoded = _recode_values(_f_data, mapping)

    new_levs_dedup = set()
    new_levels = []
    for nl in new_levels_list:
        if nl not in new_levs_dedup and nl is not None:
            new_levs_dedup.add(nl)
            new_levels.append(nl)

    if other_level in new_levels:
        new_levels.remove(other_level)
        new_levels.append(other_level)

    return _make_enum(recoded, new_levels)


# ---- fct_lump_n ----------------------------------------------------------


@fct_lump_n.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_lump_n(
    _f, n: int, w=None, other_level: Any = "Other", ties_method: str = "min"
) -> pl.Series:
    """Lumps all levels except for the `n` most frequent."""
    calcs = _check_calc_levels(_f, w)
    _f_data = calcs["_f"]
    levs = calcs["levs"]
    counts = calcs["count"]

    # Rank (1-based, descending: highest count = rank 1)
    if ties_method == "min":
        # dense rank
        unique_sorted = sorted(set(counts), reverse=True)
        rank_map = {v: r for r, v in enumerate(unique_sorted, 1)}
        ranks = np.array([rank_map[c] for c in counts])
    elif ties_method == "max":
        # max rank for ties
        order = np.argsort(-counts)
        ranks = np.zeros(len(counts), dtype=int)
        j = 1
        for i in range(len(order)):
            if i > 0 and counts[order[i]] != counts[order[i - 1]]:
                j = i + 1
            ranks[order[i]] = j
    elif ties_method == "first":
        order = np.argsort(-counts, kind="stable")
        ranks = np.zeros(len(counts), dtype=int)
        for i, o in enumerate(order, 1):
            ranks[o] = i
    else:
        # "dense" or "average" — use dense
        unique_sorted = sorted(set(counts), reverse=True)
        rank_map = {v: r for r, v in enumerate(unique_sorted, 1)}
        ranks = np.array([rank_map[c] for c in counts])

    if n > 0:
        new_levels_list = [
            other_level if ranks[i] > n else levs[i]
            for i in range(len(levs))
        ]
    else:
        # Negative n: keep least common
        neg_n = -n
        new_levels_list = [
            other_level if ranks[::-1][i] > neg_n else levs[i]
            for i in range(len(levs))
        ]
        # Simpler approach: reverse rank
        order = np.argsort(counts)
        ranks_asc = np.zeros(len(counts), dtype=int)
        for r, o in enumerate(order, 1):
            ranks_asc[o] = r
        new_levels_list = [
            other_level if ranks_asc[i] > neg_n else levs[i]
            for i in range(len(levs))
        ]

    if other_level not in new_levels_list:
        return _f_data

    if sum(np.array(new_levels_list) == other_level) <= 1:
        return _f_data

    mapping = dict(zip(levs, new_levels_list))
    recoded = _recode_values(_f_data, mapping)

    new_levs_dedup = set()
    new_levels = []
    for nl in new_levels_list:
        if nl not in new_levs_dedup and nl is not None:
            new_levs_dedup.add(nl)
            new_levels.append(nl)

    if other_level in new_levels:
        new_levels.remove(other_level)
        new_levels.append(other_level)

    return _make_enum(recoded, new_levels)


# ---- fct_lump_lowfreq ----------------------------------------------------


@fct_lump_lowfreq.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_lump_lowfreq(_f, other_level: Any = "Other") -> pl.Series:
    """Lumps together the least frequent levels."""
    calcs = _check_calc_levels(_f)
    _f_data = calcs["_f"]
    levs = calcs["levs"]
    counts = calcs["count"]

    smallest = _in_smallest(counts)
    new_levels_list = [
        other_level if smallest[i] else levs[i]
        for i in range(len(levs))
    ]

    if other_level not in new_levels_list:
        return _f_data

    mapping = dict(zip(levs, new_levels_list))
    recoded = _recode_values(_f_data, mapping)

    new_levs_dedup = set()
    new_levels = []
    for nl in new_levels_list:
        if nl not in new_levs_dedup and nl is not None:
            new_levs_dedup.add(nl)
            new_levels.append(nl)

    if other_level in new_levels:
        new_levels.remove(other_level)
        new_levels.append(other_level)

    return _make_enum(recoded, new_levels)


# ---- fct_other -----------------------------------------------------------


@fct_other.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_other(
    _f,
    keep=None,
    drop=None,
    other_level: Any = "Other",
) -> pl.Series:
    """Replace levels with 'other'."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)

    if (keep is None and drop is None) or (keep is not None and drop is not None):
        raise ValueError("Must supply exactly one of `keep` and `drop`")

    if keep is not None:
        keep_set = set(keep) if not is_scalar(keep) else {keep}
        new_levs = [other_level if l not in keep_set else l for l in levs]
    else:
        drop_set = set(drop) if not is_scalar(drop) else {drop}
        new_levs = [other_level if l in drop_set else l for l in levs]

    if other_level not in new_levs:
        return _f

    mapping = dict(zip(levs, new_levs))
    recoded = _recode_values(_f, mapping)

    new_levs_dedup = set()
    new_levels = []
    for nl in new_levs:
        if nl not in new_levs_dedup and nl is not None:
            new_levs_dedup.add(nl)
            new_levels.append(nl)

    # Move other_level to end
    if other_level in new_levels:
        new_levels.remove(other_level)
        new_levels.append(other_level)

    return _make_enum(recoded, new_levels)


# ---- fct_relabel ---------------------------------------------------------


@fct_relabel.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_relabel(_f, _fun, *args, **kwargs) -> pl.Series:
    """Automatically relabel factor levels."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)

    if not callable(_fun):
        raise TypeError("`_fun` must be callable")

    new_levs = _fun(levs, *args, **kwargs)

    if isinstance(new_levs, np.ndarray):
        new_levs = new_levs.tolist()
    elif not isinstance(new_levs, (list, tuple, pl.Series)):
        raise TypeError("`_fun` must return a list-like of new level labels")

    if len(new_levs) != len(levs):
        raise ValueError(
            "`new_levels` must be the same length as `levels(f)`: "
            f"expected {len(levs)} new levels, got {len(new_levs)}."
        )

    return _lvls_revalue(_f, new_levs)


# ---- fct_expand ----------------------------------------------------------


@fct_expand.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_expand(_f, *additional_levels: Any) -> pl.Series:
    """Add additional levels to a factor."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)

    addlevs = []
    for alev in additional_levels:
        if is_scalar(alev):
            addlevs.append(alev)
        else:
            addlevs.extend(alev)

    new_levels = union(levs, addlevs)
    return _lvls_expand(_f, new_levels)


# ---- fct_explicit_na -----------------------------------------------------


@fct_explicit_na.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_explicit_na(_f, na_level: Any = "(Missing)") -> pl.Series:
    """Make missing values explicit."""
    _f = _check_factor(_f)
    vals = _f.to_list()
    levs = _get_levels(_f)

    has_missing = any(v is None for v in vals)
    if not has_missing:
        return _f

    new_levs = union(levs, [na_level])
    new_vals = [na_level if v is None else v for v in vals]
    return _make_enum(
        pl.Series(_f.name, new_vals),
        new_levs,
        name=_f.name,
    )


# ---- fct_drop ------------------------------------------------------------


@fct_drop.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_drop(_f, only: Any = None) -> pl.Series:
    """Drop unused levels."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    vals = set(v for v in _f.to_list() if v is not None)

    to_drop = [l for l in levs if l not in vals]
    if only is not None:
        if is_scalar(only):
            only = [only]
        to_drop = intersect(to_drop, only)

    new_levels = setdiff(levs, to_drop)
    return _make_enum(_f, new_levels)


# ---- fct_unify -----------------------------------------------------------


@fct_unify.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_unify(fs, levels=None) -> list:
    """Unify the levels in a list of factors."""
    if levels is None:
        levels = _lvls_union(fs)

    out = []
    for fct in fs:
        fct = _check_factor(fct)
        out.append(_lvls_expand(fct, new_levels=levels))
    return out


# ---- fct_c ---------------------------------------------------------------


@fct_c.register(object, backend="polars")
def _fct_c(*fs) -> pl.Series:
    """Concatenate factors, combining levels."""
    if not fs:
        return pl.Series("", [], dtype=pl.Enum([]))

    checked = [_check_factor(f) for f in fs]
    all_levels = []
    for f in checked:
        all_levels = union(all_levels, _get_levels(f))

    all_vals = []
    for f in checked:
        all_vals.extend(f.to_list())

    return _make_enum(pl.Series("", all_vals), all_levels)


# ---- fct_cross -----------------------------------------------------------


@fct_cross.register(object, backend="polars")
def _fct_cross(
    *fs,
    sep: str = ":",
    keep_empty: bool = False,
) -> pl.Series:
    """Combine levels from two or more factors to create a new factor."""
    if not fs or (len(fs) == 1 and len(_check_factor(fs[0])) == 0):
        return pl.Series("", [], dtype=pl.Enum([]))

    checked = [_check_factor(f) for f in fs]
    n = len(checked[0])
    for f in checked[1:]:
        if len(f) != n:
            raise ValueError("All factors must have the same length")

    # Cross values
    all_values = []
    for i in range(n):
        parts = [str(f[i]) if f[i] is not None else None for f in checked]
        if any(p is None for p in parts):
            all_values.append(None)
        else:
            all_values.append(sep.join(parts))

    # Cross levels
    level_sets = [_get_levels(f) for f in checked]
    new_levels = []
    for combo in itertools.product(*level_sets):
        new_levels.append(sep.join(str(c) for c in combo))

    if not keep_empty:
        present = set(v for v in all_values if v is not None)
        new_levels = [l for l in new_levels if l in present]

    return _make_enum(pl.Series("", all_values), new_levels)


# ---- fct_reorder2 Expr ---------------------------------------------------


@fct_reorder2.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_reorder2_expr(
    _f: pl.Expr,
    _x: pl.Expr,
    *args: Any,
    _fun=None,
    _desc: bool = True,
    **kwargs: Any,
) -> pl.Expr:
    """fct_reorder2 for pl.Expr inputs (used inside mutate/summarise)."""
    data = _check_eval_data("fct_reorder2")
    f_name = _f.meta.output_name()
    x_name = _x.meta.output_name()
    if f_name is None or x_name is None:
        raise ValueError(
            "fct_reorder2 requires simple column references (e.g., f.x, f.y), "
            "not computed expressions."
        )
    cols = [f_name, x_name]
    if args and isinstance(args[0], pl.Expr):
        y_name = args[0].meta.output_name()
        if y_name is None:
            raise ValueError(
                "fct_reorder2 requires simple column references."
            )
        if y_name not in cols:
            cols.append(y_name)
        args = args[1:]
    y_name = cols[2] if len(cols) > 2 else None

    subset = data.select(cols).collect()
    f_series = _check_factor(subset.get_column(f_name))
    x_series = subset.get_column(x_name)
    if y_name is not None:
        y_series = subset.get_column(y_name)
    else:
        # _x and _y reference the same column
        y_series = x_series
    result = _fct_reorder2(
        f_series, x_series, y_series,
        _fun=_fun, _desc=_desc, **kwargs
    )
    return _finish_expr(_f, result)


# ---- Simple fct_* Expr wrappers ------------------------------------------


@fct_lump.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_lump_expr(_f, n=None, prop=None, w=None, other_level="Other",
                   ties_method="min"):
    data = _check_eval_data("fct_lump")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_lump(f_series, n=n, prop=prop, w=w, other_level=other_level,
                       ties_method=ties_method)
    return _finish_expr(_f, result)


@fct_lump_min.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_lump_min_expr(_f, min_, w=None, other_level="Other"):
    data = _check_eval_data("fct_lump_min")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_lump_min(f_series, min_=min_, w=w, other_level=other_level)
    return _finish_expr(_f, result)


@fct_lump_prop.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_lump_prop_expr(_f, prop, w=None, other_level="Other"):
    data = _check_eval_data("fct_lump_prop")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_lump_prop(f_series, prop=prop, w=w, other_level=other_level)
    return _finish_expr(_f, result)


@fct_lump_n.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_lump_n_expr(_f, n, w=None, other_level="Other", ties_method="min"):
    data = _check_eval_data("fct_lump_n")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_lump_n(f_series, n=n, w=w, other_level=other_level,
                         ties_method=ties_method)
    return _finish_expr(_f, result)


@fct_lump_lowfreq.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_lump_lowfreq_expr(_f, other_level="Other"):
    data = _check_eval_data("fct_lump_lowfreq")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_lump_lowfreq(f_series, other_level=other_level)
    return _finish_expr(_f, result)


@fct_expand.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_expand_expr(_f, *additional_levels):
    data = _check_eval_data("fct_expand")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_expand(f_series, *additional_levels)
    return _finish_expr(_f, result)


@fct_explicit_na.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_explicit_na_expr(_f, na_level="(Missing)"):
    data = _check_eval_data("fct_explicit_na")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_explicit_na(f_series, na_level=na_level)
    return _finish_expr(_f, result)


@fct_drop.register(pl.Expr, context=Context.EVAL, backend="polars")
def _fct_drop_expr(_f, only=None):
    data = _check_eval_data("fct_drop")
    f_series = _resolve_factor_expr(_f, data)
    result = _fct_drop(f_series, only=only)
    return _finish_expr(_f, result)


# ---- fct_count -----------------------------------------------------------


@fct_count.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_count(_f, sort: bool = False, prop: bool = False) -> pl.DataFrame:
    """Count entries in a factor."""
    _f = _check_factor(_f)
    vals = _f.to_list()
    levs = _get_levels(_f)

    # Count per level
    counts = {}
    for lev in levs:
        counts[lev] = 0
    n_na = 0
    for v in vals:
        if v is None:
            n_na += 1
        elif v in counts:
            counts[v] += 1

    f_col = list(levs)
    n_col = [counts[l] for l in levs]

    if n_na > 0:
        f_col.append(None)
        n_col.append(n_na)

    df = pl.DataFrame({"f": f_col, "n": n_col})

    if sort:
        df = df.sort("n", descending=True)

    if prop:
        total = sum(n_col)
        p_col = [n / total if total > 0 else 0.0 for n in df["n"]]
        df = df.with_columns(pl.Series("p", p_col))

    return df


# ---- fct_match -----------------------------------------------------------


@fct_match.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_match(_f, lvls: Any) -> np.ndarray:
    """Test for presence of levels in a factor."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)

    if is_scalar(lvls):
        lvls = [lvls]
    lvls = list(lvls)

    bad_lvls = setdiff(lvls, levs)
    bad_lvls = [l for l in bad_lvls if l is not None]
    if bad_lvls:
        raise ValueError(f"Levels not present in factor: {bad_lvls}.")

    vals = _f.to_list()
    lvl_set = set(lvls)
    return np.array([v in lvl_set for v in vals], dtype=bool)


# ---- fct_unique ----------------------------------------------------------


@fct_unique.register(ForcatsRegType, context=Context.EVAL, backend="polars")
def _fct_unique(_f) -> pl.Series:
    """Unique values of a factor, as a factor."""
    _f = _check_factor(_f)
    levs = _get_levels(_f)
    vals = _f.to_list()
    seen = set()
    uniq = []
    for v in vals:
        if v is not None and v not in seen:
            seen.add(v)
            uniq.append(v)
    return _make_enum(pl.Series(_f.name, uniq), levs)
