"""Expression-level helper functions from dplyr.

These are used within mutate/filter/summarise expressions.
Functions already in api/base/ (if_else, case_when, etc.) are NOT duplicated here.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from datar.apis.dplyr import (
    n,
    between,
    cumall,
    cumany,
    cummean,
    near,
    coalesce,
    na_if,
    nth,
    first,
    last,
)

from ...common import is_iterable, to_series
from ...polars import Series
from ...contexts import Context
from ...tibble import Tibble, LazyTibble


# ── n() ─────────────────────────────────────────────────────────────────────

@n.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _n(_data: Tibble) -> pl.Expr:
    """Row count in current group — returns pl.len()."""
    return pl.len()


@n.register(Series, backend="polars")
def _n_series(_data: Series) -> int:
    return len(_data)


@n.register(object, backend="polars")
def _n_object(_data: Any = None) -> int:
    raise ValueError(
        "`n()` must only be used inside dplyr verbs (mutate, summarise, filter)."
    )


# ── between ─────────────────────────────────────────────────────────────────

@between.register(pl.Expr, context=Context.EVAL, backend="polars")
def _between_expr(
    x: pl.Expr,
    left: Any,
    right: Any,
    inclusive: str = "both",
) -> pl.Expr:
    return x.is_between(left, right, closed=inclusive)


@between.register(Series, backend="polars")
def _between_series(
    x: Series,
    left: Any,
    right: Any,
    inclusive: str = "both",
) -> Series:
    return x.is_between(left, right, closed=inclusive)


@between.register(object, backend="polars")
def _between_obj(
    x: Any,
    left: Any,
    right: Any,
    inclusive: str = "both",
) -> Any:
    if isinstance(x, (list, tuple, range)):
        x = pl.Series(x)
    if isinstance(x, pl.Series):
        return x.is_between(left, right, closed=inclusive).to_list()
    if inclusive == "both":
        return left <= x <= right
    elif inclusive == "left":
        return left <= x < right
    elif inclusive == "right":
        return left < x <= right
    elif inclusive == "neither":
        return left < x < right
    raise ValueError(
        f"`inclusive` must be one of 'both', 'neither', 'left', 'right', "
        f"got '{inclusive}'."
    )


# ── cumall ─────────────────────────────────────────────────────────────────


@cumall.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cumall_expr(x: pl.Expr) -> pl.Expr:
    """Cumulative all — TRUE until the first FALSE, then FALSE forever."""
    return x.fill_null(False).cast(pl.Boolean).cum_prod().cast(pl.Boolean)


@cumall.register(object, context=Context.EVAL, backend="polars")
def _cumall_obj(x: Any) -> Any:
    if isinstance(x, Series):
        return x.fill_null(False).cast(pl.Boolean).cum_prod().cast(pl.Boolean)
    import numpy as np

    arr = np.asarray(list(x), dtype=bool)
    arr[arr != arr] = False  # replace NaN
    return np.cumprod(arr).astype(bool)


# ── cumany ─────────────────────────────────────────────────────────────────


@cumany.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cumany_expr(x: pl.Expr) -> pl.Expr:
    """Cumulative any — FALSE until the first TRUE, then TRUE forever."""
    return x.fill_null(False).cast(pl.Boolean).cum_sum().cast(pl.Boolean)


@cumany.register(object, context=Context.EVAL, backend="polars")
def _cumany_obj(x: Any) -> Any:
    if isinstance(x, Series):
        return x.fill_null(False).cast(pl.Boolean).cum_sum().cast(pl.Boolean)
    import numpy as np

    arr = np.asarray(list(x), dtype=bool)
    arr[arr != arr] = False  # replace NaN
    return np.cumsum(arr).astype(bool)


# ── cummean ────────────────────────────────────────────────────────────────


@cummean.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cummean_expr(x: pl.Expr) -> pl.Expr:
    """Cumulative mean — cumsum(x) / 1..n (1-based index)."""
    return x.cum_sum().cast(pl.Float64) / pl.int_range(
        1, pl.len() + 1
    ).cast(pl.Float64)


@cummean.register(object, context=Context.EVAL, backend="polars")
def _cummean_obj(x: Any) -> Any:
    if isinstance(x, Series):
        n = pl.int_range(1, x.len() + 1, eager=True, dtype=pl.Float64)
        return x.cum_sum().cast(pl.Float64) / n
    import numpy as np

    arr = np.asarray(list(x), dtype=float)
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)


# ── near ───────────────────────────────────────────────────────────────────


@near.register(pl.Expr, context=Context.EVAL, backend="polars")
def _near_expr(x: pl.Expr, y: Any, tol: float = 1e-8) -> pl.Expr:
    """Are x and y nearly equal within tolerance? abs(x - y) < tol."""
    if not isinstance(y, pl.Expr):
        y = pl.lit(y)
    return (x - y).abs() < tol


@near.register(object, context=Context.EVAL, backend="polars")
def _near_obj(x: Any, y: Any, tol: float = 1e-8) -> Any:
    if isinstance(x, Series):
        return (x - y).abs() < tol
    import numpy as np

    return np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float)) < tol


# ── coalesce ──────────────────────────────────────────────────────────────


@coalesce.register(pl.Expr, context=Context.EVAL, backend="polars")
def _coalesce_expr(x: pl.Expr, *replace: Any) -> pl.Expr:
    result = x
    for r in replace:
        if isinstance(r, pl.Expr):
            result = result.fill_null(r).fill_nan(r)
        elif isinstance(r, pl.Series):
            result = result.fill_null(pl.lit(r)).fill_nan(pl.lit(r))
        else:
            result = result.fill_null(r).fill_nan(r)
    return result


@coalesce.register(object, context=Context.EVAL, backend="polars")
def _coalesce_obj(x: Any, *replace: Any) -> Any:
    if isinstance(x, pl.Series) or (is_iterable(x) and len(replace) > 0):
        x_s = to_series(x)
        result = x_s
        for r in replace:
            r_s = to_series(r, len(result))
            result = result.fill_null(r_s)
        return result
    if x is not None:
        import math

        if isinstance(x, float) and math.isnan(x):
            pass
        else:
            return x
    for r in replace:
        if r is not None:
            import math

            if isinstance(r, float) and math.isnan(r):
                continue
            return r
    return None


# ── na_if ─────────────────────────────────────────────────────────────────


@na_if.register(pl.Expr, context=Context.EVAL, backend="polars")
def _na_if_expr(x: pl.Expr, value: Any) -> pl.Expr:
    if isinstance(value, pl.Expr):
        return pl.when(x == value).then(None).otherwise(x)
    return pl.when(x == value).then(None).otherwise(x)


@na_if.register(object, context=Context.EVAL, backend="polars")
def _na_if_obj(x: Any, value: Any) -> Any:
    if isinstance(x, pl.Series):
        mask = x == value
        result = x.clone()
        result[mask] = None
        return result
    if isinstance(x, (list, tuple, range)):
        x_series = pl.Series(x)
        mask = x_series == value
        result = x_series.clone()
        result[mask] = None
        return result.to_list()
    if x == value:
        return None
    return x


# ── nth ───────────────────────────────────────────────────────────────────


@nth.register(pl.Expr, context=Context.EVAL, backend="polars")
def _nth_expr(
    x: pl.Expr,
    n: int,
    order_by: Any = None,
    default: Any = None,
) -> pl.Expr:
    result = x.get(n)
    if default is not None:
        result = result.fill_null(default)
    return result


@nth.register(object, context=Context.EVAL, backend="polars")
def _nth_obj(
    x: Any,
    n: int,
    order_by: Any = None,
    default: Any = None,
) -> Any:
    if isinstance(x, pl.Series):
        try:
            return x[n]
        except IndexError:
            return default
    try:
        return list(x)[n]
    except (IndexError, TypeError):
        return default


# ── first ─────────────────────────────────────────────────────────────────


@first.register(pl.Expr, context=Context.EVAL, backend="polars")
def _first_expr(
    x: pl.Expr,
    order_by: Any = None,
    default: Any = None,
) -> pl.Expr:
    return x.first()


@first.register(object, context=Context.EVAL, backend="polars")
def _first_obj(
    x: Any,
    order_by: Any = None,
    default: Any = None,
) -> Any:
    if isinstance(x, pl.Series):
        if order_by is not None:
            order_s = (
                order_by
                if isinstance(order_by, pl.Series)
                else pl.Series(list(order_by))
            )
            sorted_x = x[order_s.arg_sort()]
            if len(sorted_x) > 0:
                return sorted_x[0]
            return default
        if len(x) > 0:
            return x[0]
        return default
    if order_by is not None:
        x_list = list(x)
        order_list = list(order_by)
        if not x_list:
            return default
        pairs = sorted(zip(order_list, x_list))
        return pairs[0][1]
    try:
        lst = list(x)
        return lst[0] if lst else default
    except (TypeError, IndexError):
        return default


# ── last ──────────────────────────────────────────────────────────────────


@last.register(pl.Expr, context=Context.EVAL, backend="polars")
def _last_expr(
    x: pl.Expr,
    order_by: Any = None,
    default: Any = None,
) -> pl.Expr:
    return x.last()


@last.register(object, context=Context.EVAL, backend="polars")
def _last_obj(
    x: Any,
    order_by: Any = None,
    default: Any = None,
) -> Any:
    if isinstance(x, pl.Series):
        if order_by is not None:
            order_s = (
                order_by
                if isinstance(order_by, pl.Series)
                else pl.Series(list(order_by))
            )
            sorted_x = x[order_s.arg_sort()]
            if len(sorted_x) > 0:
                return sorted_x[-1]
            return default
        if len(x) > 0:
            return x[-1]
        return default
    if order_by is not None:
        x_list = list(x)
        order_list = list(order_by)
        if not x_list:
            return default
        pairs = sorted(zip(order_list, x_list))
        return pairs[-1][1]
    try:
        lst = list(x)
        return lst[-1] if lst else default
    except (TypeError, IndexError):
        return default
