"""Factor handling functions for the polars backend.

Implements: factor, ordered, levels, set_levels, nlevels, droplevels,
is_factor, is_ordered.

Polars uses pl.Categorical for factor-like behavior.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import (
    as_factor,
    as_ordered,
    cut,
    factor,
    ordered,
    levels,
    set_levels,
    nlevels,
    droplevels,
    is_factor,
    is_ordered,
)

from ...contexts import Context
from ...utils import replace_na_with_none


# ── helpers ──────────────────────────────────────────────────────────────


def _prepare_levels(levels: Any, exclude: Any) -> list:
    """Convert `levels` to a string list for pl.Enum, optionally removing
    `exclude` values."""
    str_levels = [str(l) for l in levels]
    if exclude is not None:
        ex_set = {str(e) for e in (
            exclude if isinstance(exclude, (list, tuple, set)) else [exclude]
        )}
        str_levels = [l for l in str_levels if l not in ex_set]
    return str_levels


def _derive_levels(values: Any) -> list:
    """Derive unique non-null levels from a values iterable, preserving
    order of first appearance. Returns string levels for pl.Enum."""
    seen = set()
    result = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and (v != v):  # NaN
            continue
        s = str(v)
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _apply_labels_series(
    series: pl.Series, labels: Any, str_levels: list
) -> pl.Series:
    """Replace enum categories with `labels` on a Series."""
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    label_map = dict(zip(str_levels, [str(l) for l in labels]))
    return series.cast(pl.Utf8).replace_strict(label_map).cast(pl.Enum([str(l) for l in labels]))


def _is_cat_dtype(dtype) -> bool:
    """Check if a polars dtype is Categorical or Enum (factor-like)."""
    return isinstance(dtype, (pl.Categorical, pl.Enum))


def _apply_labels_expr(
    expr: pl.Expr, labels: Any, str_levels: list
) -> pl.Expr:
    """Replace enum categories with `labels` on an Expr."""
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    label_map = dict(zip(str_levels, [str(l) for l in labels]))
    return expr.cast(pl.Utf8).replace_strict(label_map).cast(pl.Enum([str(l) for l in labels]))


# ── factor ──────────────────────────────────────────────────────────────


@factor.register(pl.Expr, context=Context.EVAL, backend="polars")
def _factor_expr(
    x: pl.Expr | None = None,
    *,
    levels: Any = None,
    labels: Any = None,
    exclude: Any = None,
    ordered: bool = False,
    nmax: Any = None,
) -> pl.Expr:
    """Create a factor (categorical) vector from data.

    In polars, factors are represented as pl.Categorical columns.
    When `levels` is provided, pl.Enum is used to preserve the
    explicit level set, including levels not present in the data.
    """
    if x is None:
        return pl.lit(None).cast(pl.Categorical)

    if levels is not None:
        str_levels = _prepare_levels(levels, exclude)
        result = x.cast(pl.Utf8).cast(pl.Enum(str_levels))
        if labels is not None:
            result = _apply_labels_expr(result, labels, str_levels)
    else:
        result = x.cast(pl.Categorical)
        if exclude is not None:
            ex = [exclude] if not isinstance(exclude, (list, tuple, set)) else exclude
            result = result.filter(~x.cast(pl.Utf8).is_in(ex))

    return result


@factor.register(object, context=Context.EVAL, backend="polars")
def _factor_obj(
    x: Any = None,
    *,
    levels: Any = None,
    labels: Any = None,
    exclude: Any = None,
    ordered: bool = False,
    nmax: Any = None,
) -> Any:
    """Create a factor from scalar/Series data.

    Always uses pl.Enum to avoid polars global string cache issues
    with pl.Categorical where categories leak between series.
    """
    if x is None:
        return pl.Series([], dtype=pl.Enum([]))

    if isinstance(x, pl.Series):
        if levels is not None:
            str_levels = _prepare_levels(levels, exclude)
            result = x.cast(pl.Utf8).cast(pl.Enum(str_levels))
            if labels is not None:
                result = _apply_labels_series(result, labels, str_levels)
        else:
            str_levels = _derive_levels(x.to_list())
            result = x.cast(pl.Utf8).cast(pl.Enum(str_levels))
            if exclude is not None:
                ex_set = {exclude} if not isinstance(
                    exclude, (list, tuple, set)
                ) else set(exclude)
                mask = ~result.cast(pl.Utf8).is_in(list(ex_set))
                result = result.filter(mask)
                # Re-derive levels after filtering
                str_levels = _derive_levels(result.to_list())
                result = result.cast(pl.Utf8).cast(pl.Enum(str_levels))
        return result

    # Scalar / iterable
    if isinstance(x, str) or not isinstance(x, (list, tuple, pl.Series)):
        values = [x]
    else:
        values = x
    values = replace_na_with_none(values)
    if levels is not None:
        str_levels = _prepare_levels(levels, exclude)
        result = pl.Series(values, dtype=pl.Enum(str_levels), strict=False)
        if labels is not None:
            result = _apply_labels_series(result, labels, str_levels)
    else:
        str_levels = _derive_levels(values)
        result = pl.Series(values, dtype=pl.Enum(str_levels), strict=False)
        if exclude is not None:
            ex_set = {exclude} if not isinstance(
                exclude, (list, tuple, set)
            ) else set(exclude)
            mask = ~result.cast(pl.Utf8).is_in(list(ex_set))
            result = result.filter(mask)
            str_levels = _derive_levels(result.to_list())
            result = result.cast(pl.Utf8).cast(pl.Enum(str_levels))
    return result


# ── ordered ─────────────────────────────────────────────────────────────


@ordered.register(pl.Expr, context=Context.EVAL, backend="polars")
def _ordered_expr(
    x: pl.Expr,
    levels: Any = None,
    labels: Any = None,
    exclude: Any = None,
    nmax: Any = None,
) -> pl.Expr:
    """Create an ordered factor from data."""
    return _factor_expr(
        x, levels=levels, labels=labels, exclude=exclude, ordered=True, nmax=nmax
    )


@ordered.register(object, context=Context.EVAL, backend="polars")
def _ordered_obj(
    x: Any,
    levels: Any = None,
    labels: Any = None,
    exclude: Any = None,
    nmax: Any = None,
) -> Any:
    """Create an ordered factor from scalar/Series data."""
    return _factor_obj(
        x, levels=levels, labels=labels, exclude=exclude, ordered=True, nmax=nmax
    )


# ── levels ──────────────────────────────────────────────────────────────


@levels.register(pl.Expr, context=Context.EVAL, backend="polars")
def _levels_expr(x: pl.Expr) -> pl.Expr:
    """Get the levels of a categorical expression (lazy — placeholder)."""
    return pl.lit([])


@levels.register(object, context=Context.EVAL, backend="polars")
def _levels_obj(x: Any) -> Any:
    """Get the levels of a factor."""
    if isinstance(x, pl.Series):
        if isinstance(x.dtype, pl.Enum):
            cats = x.cat.get_categories()
            return cats.to_list()
        if isinstance(x.dtype, pl.Categorical):
            return x.drop_nulls().unique(maintain_order=True).cast(pl.Utf8).to_list()
        return None
    return None


# ── set_levels ──────────────────────────────────────────────────────────


@set_levels.register(pl.Expr, context=Context.EVAL, backend="polars")
def _set_levels_expr(x: pl.Expr, levels: Any) -> pl.Expr:
    """Set levels on a categorical expression — best effort."""
    # In lazy mode, we can't directly set levels; return as-is
    return x


@set_levels.register(object, context=Context.EVAL, backend="polars")
def _set_levels_obj(x: Any, levels: Any) -> Any:
    """Set levels on a factor Series."""
    if isinstance(x, pl.Series) and _is_cat_dtype(x.dtype):
        str_levels = [str(l) for l in levels]
        return x.cast(pl.Utf8).cast(pl.Enum(str_levels))
    return x


# ── nlevels ─────────────────────────────────────────────────────────────


@nlevels.register(pl.Expr, context=Context.EVAL, backend="polars")
def _nlevels_expr(x: pl.Expr) -> pl.Expr:
    """Get the number of levels of a categorical expression."""
    return pl.lit(0)


@nlevels.register(object, context=Context.EVAL, backend="polars")
def _nlevels_obj(x: Any) -> int:
    """Get the number of levels of a factor."""
    if isinstance(x, pl.Series):
        if isinstance(x.dtype, pl.Enum):
            return len(x.cat.get_categories())
        if isinstance(x.dtype, pl.Categorical):
            return x.drop_nulls().n_unique()
    return 0


# ── droplevels ──────────────────────────────────────────────────────────


@droplevels.register(pl.Expr, context=Context.EVAL, backend="polars")
def _droplevels_expr(x: pl.Expr) -> pl.Expr:
    """Drop unused levels from a categorical expression."""
    return x


@droplevels.register(object, context=Context.EVAL, backend="polars")
def _droplevels_obj(x: Any) -> Any:
    """Drop unused levels from a factor."""
    if isinstance(x, pl.Series) and _is_cat_dtype(x.dtype):
        used_vals = set(x.drop_nulls().cast(pl.Utf8).unique().to_list())
        if isinstance(x.dtype, pl.Enum):
            current_cats = x.cat.get_categories().to_list()
            new_cats = [c for c in current_cats if c in used_vals]
            return x.cast(pl.Utf8).cast(pl.Enum(new_cats))
        return x.drop_nulls().cast(pl.Categorical)
    return x


# ── is_factor ───────────────────────────────────────────────────────────


@is_factor.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_factor_expr(x: pl.Expr) -> pl.Expr:
    """Check if an expression is a factor (lazy — placeholder)."""
    return pl.lit(False)


@is_factor.register(object, context=Context.EVAL, backend="polars")
def _is_factor_obj(x: Any) -> bool:
    """Check if x is a factor."""
    if isinstance(x, pl.Series):
        return _is_cat_dtype(x.dtype)
    return False


# ── is_ordered ──────────────────────────────────────────────────────────


@is_ordered.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_ordered_expr(x: pl.Expr) -> pl.Expr:
    """Check if an expression is an ordered factor (lazy — placeholder)."""
    return pl.lit(False)


@is_ordered.register(object, context=Context.EVAL, backend="polars")
def _is_ordered_obj(x: Any) -> bool:
    """Check if x is an ordered factor.

    In polars >= 1.32, all Categorical types are lexical (ordered).
    """
    if isinstance(x, pl.Series) and _is_cat_dtype(x.dtype):
        return True
    return False


# ── as_factor ─────────────────────────────────────────────────────────────


@as_factor.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_factor_expr(x: pl.Expr) -> pl.Expr:
    """Convert an expression to a factor (categorical)."""
    return x.cast(pl.Utf8).cast(pl.Categorical)


@as_factor.register(object, context=Context.EVAL, backend="polars")
def _as_factor_obj(x: Any) -> Any:
    """Convert an object to a factor (categorical).

    Uses pl.Enum to avoid polars global string cache issues with
    pl.Categorical where categories leak between series.
    """
    if isinstance(x, pl.Series):
        if _is_cat_dtype(x.dtype):
            return x
        str_levels = _derive_levels(x.to_list())
        return x.cast(pl.Utf8).cast(pl.Enum(str_levels))
    if isinstance(x, (list, tuple)):
        str_levels = _derive_levels(x)
        return pl.Series(
            [str(v) if v is not None else None for v in x],
            dtype=pl.Enum(str_levels),
        )
    return pl.Series([str(x)], dtype=pl.Enum([str(x)]))


# ── as_ordered ────────────────────────────────────────────────────────────


@as_ordered.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_ordered_expr(x: pl.Expr) -> pl.Expr:
    """Convert an expression to an ordered factor."""
    return x.cast(pl.Utf8).cast(pl.Categorical)


@as_ordered.register(object, context=Context.EVAL, backend="polars")
def _as_ordered_obj(x: Any) -> Any:
    """Convert an object to an ordered factor.

    Uses pl.Enum to avoid polars global string cache issues with
    pl.Categorical where categories leak between series.
    """
    if isinstance(x, pl.Series):
        if _is_cat_dtype(x.dtype):
            return x
        str_levels = _derive_levels(x.to_list())
        return x.cast(pl.Utf8).cast(pl.Enum(str_levels))
    if isinstance(x, (list, tuple)):
        str_levels = _derive_levels(x)
        return pl.Series(
            [str(v) if v is not None else None for v in x],
            dtype=pl.Enum(str_levels),
        )
    return pl.Series([str(x)], dtype=pl.Enum([str(x)]))


# ── cut ──────────────────────────────────────────────────────────────────


@cut.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cut_expr(
    x: pl.Expr,
    breaks: Any,
    labels: Any = None,
    include_lowest: bool = False,
    right: bool = True,
    dig_lab: int = 3,
    ordered_result: bool = False,
) -> pl.Expr:
    """Cut a numeric expression into bins."""
    if isinstance(breaks, int):
        return x.qcut(breaks, labels=labels is not None)
    if isinstance(breaks, (list, tuple)):
        return x.cut(breaks, labels=labels)
    raise ValueError(f"Unsupported breaks type: {type(breaks)}")


@cut.register(object, context=Context.EVAL, backend="polars")
def _cut_obj(
    x: Any,
    breaks: Any,
    labels: Any = None,
    include_lowest: bool = False,
    right: bool = True,
    dig_lab: int = 3,
    ordered_result: bool = False,
) -> Any:
    """Cut a numeric vector into bins."""
    if isinstance(x, pl.Series):
        if isinstance(breaks, int):
            if labels is not None:
                return x.qcut(breaks, labels=labels, include_breaks=True)
            return x.qcut(breaks, labels=labels)
        if isinstance(breaks, (list, tuple)):
            return x.cut(breaks, labels=labels)
        raise ValueError(f"Unsupported breaks type: {type(breaks)}")
    if isinstance(x, pl.Expr):
        return _cut_expr(x, breaks, labels, include_lowest, right, dig_lab, ordered_result)
    import numpy as np
    x_arr = np.asarray(x, dtype=float)
    if isinstance(breaks, int):
        bins = np.histogram_bin_edges(x_arr, bins=breaks)
    else:
        bins = np.asarray(breaks, dtype=float)
    indices = np.digitize(x_arr, bins, right=right)
    if include_lowest:
        indices[x_arr == bins[0]] = 1
    if labels is not None:
        if len(labels) != len(bins) - 1:
            raise ValueError("`labels` must be the same length as the number of bins.")
        label_arr = np.array(labels)
        result_arr = np.where((indices > 0) & (indices < len(bins)),
                              label_arr[indices - 1], np.nan)
        return result_arr.tolist()
    return indices
