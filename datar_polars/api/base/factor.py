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
from ...tibble import to_lazy
from ...utils import cat_levels, replace_na_with_none
from ..forcats import _EVAL_DATA


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


def _derive_levels_from_data(x: pl.Expr) -> list | None:
    """Derive the Enum levels of `x` by evaluating it against the data it is
    being evaluated on.

    `mutate` publishes that data in `_EVAL_DATA` for the duration of each
    expression's evaluation. Outside of it (any other verb, or a bare Expr)
    the data is not available, so None is returned and the caller falls back
    to pl.Categorical."""
    data = _EVAL_DATA.get()
    if data is None:
        return None
    try:
        values = (
            to_lazy(data).select(x).collect().get_column(x.meta.output_name())
        )
    except Exception:
        # Levels are best-effort: fall back to Categorical rather than fail here
        return None
    return _derive_levels(values.cast(pl.Utf8).to_list())


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
    if isinstance(x, pl.Series) and _is_cat_dtype(x.dtype):
        return cat_levels(x)
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
            return len(x.dtype.categories)
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
            current_cats = x.dtype.categories.to_list()
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
    """Convert an expression to an ordered factor.

    Ordered factors are represented as pl.Enum, whose categories carry the
    level order (levels are derived from the data being evaluated, in order
    of first appearance). Falls back to pl.Categorical when the levels are
    not available (e.g. a bare Expr outside a verb).
    """
    str_levels = _derive_levels_from_data(x)
    if str_levels is None:
        return x.cast(pl.Utf8).cast(pl.Categorical)
    return x.cast(pl.Utf8).cast(pl.Enum(str_levels))


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


def _cut_breaks(lo: float, hi: float, n: int) -> list[float]:
    """The n + 1 breaks R's cut() derives for an integer `breaks`.

    Equal-width intervals spanning [lo, hi], with the outer breaks nudged
    outwards so the extreme values fall inside the first and last interval
    instead of on an excluded edge.
    """
    dx = hi - lo
    if dx == 0:
        dx = abs(lo) if lo else 1.0
        return [lo + dx / 1000 * (2 * i / n - 1) for i in range(n + 1)]
    breaks = [lo + dx * i / n for i in range(n + 1)]
    breaks[0] -= dx / 1000
    breaks[-1] += dx / 1000
    return breaks


def _cut_labels(breaks: list, right: bool, dig_lab: int) -> list[str]:
    """R's interval labels for `breaks`, e.g. '(0.995,2.67]'."""
    ends = [f"{b:.{dig_lab}g}" for b in breaks]
    return [
        f"({a},{b}]" if right else f"[{a},{b})"
        for a, b in zip(ends, ends[1:])
    ]


def _cut_codes(
    x: pl.Expr, breaks: list, right: bool, include_lowest: bool
) -> pl.Expr:
    """The 1-based bin of each value of `x` within `breaks`, null outside them.

    Bin i holds the values in (breaks[i-1], breaks[i]] when `right`, or
    [breaks[i-1], breaks[i]) otherwise. `breaks` may hold floats or, when the
    values are only known at runtime, expressions.
    """
    if right:
        idx = pl.sum_horizontal([x > b for b in breaks])
    else:
        idx = pl.sum_horizontal([x >= b for b in breaks])
    nbins = len(breaks) - 1
    if include_lowest:
        # R closes the first interval's left edge -- the last one's right edge
        # when right=False -- so the extreme value is binned rather than dropped
        idx = (
            pl.when(x == breaks[0])
            .then(pl.lit(1 if right else nbins))
            .otherwise(idx)
        )
    return pl.when(x.is_not_null() & (idx >= 1) & (idx <= nbins)).then(idx)


def _cut_expr_breaks(x: pl.Expr, n: int) -> list:
    """`_cut_breaks` with the range left as expressions, for the case where the
    data cut() is being applied to cannot be reached."""
    lo, hi = x.min(), x.max()
    dx = hi - lo
    return (
        [lo - dx / 1000]
        + [lo + dx * i / n for i in range(1, n)]
        + [hi + dx / 1000]
    )


def _cut_factor(
    x: pl.Expr,
    breaks: Any,
    labels: Any,
    include_lowest: bool,
    right: bool,
    dig_lab: int,
) -> pl.Expr:
    """The cut() expression for `x`: a factor whose levels are the bin labels."""
    breaks = list(breaks)
    if len(breaks) < 2:
        raise ValueError("`breaks` must hold at least two values.")
    if any(b <= a for a, b in zip(breaks, breaks[1:])):
        raise ValueError("`breaks` must be strictly increasing.")

    nbins = len(breaks) - 1
    if labels is None:
        labels = _cut_labels(breaks, right, dig_lab)
    elif len(labels) != nbins:
        raise ValueError(f"`labels` must hold {nbins} values, got {len(labels)}.")
    labels = [str(label) for label in labels]

    codes = _cut_codes(x, breaks, right, include_lowest)
    return codes.replace_strict(
        dict(enumerate(labels, 1)), default=None, return_dtype=pl.Enum(labels)
    )


def _cut_range(x: pl.Expr) -> tuple | None:
    """The (min, max) of `x`, or None when its data cannot be reached.

    `mutate` publishes the data under evaluation in `_EVAL_DATA`; outside of
    it (any other verb, or a bare Expr) the range is not known up front.
    """
    data = _EVAL_DATA.get()
    if data is None:
        return None
    try:
        lo, hi = (
            to_lazy(data)
            .select(x.min().alias("_lo"), x.max().alias("_hi"))
            .collect()
            .row(0)
        )
    except Exception:
        return None
    if lo is None:
        raise ValueError("`cut` cannot derive breaks from an all-null vector.")
    return lo, hi


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
        rng = _cut_range(x)
        if rng is None:
            # The data is out of reach, so the intervals cannot be spaced or
            # named after it; all that can be returned is the bin number.
            return _cut_codes(
                x, _cut_expr_breaks(x, breaks), right, include_lowest
            )
        breaks = _cut_breaks(rng[0], rng[1], breaks)
    return _cut_factor(x, breaks, labels, include_lowest, right, dig_lab)


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
    if isinstance(x, pl.Expr):
        return _cut_expr(
            x, breaks, labels, include_lowest, right, dig_lab, ordered_result
        )
    if not isinstance(x, pl.Series):
        x = pl.Series("x", list(x), strict=False)
    frame = x.to_frame()
    name = frame.columns[0]
    if isinstance(breaks, int):
        values = frame.get_column(name)
        if values.min() is None:
            raise ValueError("`cut` cannot derive breaks from an all-null vector.")
        breaks = _cut_breaks(values.min(), values.max(), breaks)
    factor = _cut_factor(pl.col(name), breaks, labels, include_lowest, right, dig_lab)
    return frame.select(factor).to_series()
