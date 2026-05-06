"""Set operations for polars

Implements: intersect, union, setdiff, setequal,
all_, any_, any_na, append, diff.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import (
    all_,
    any_,
    any_na,
    append,
    diff,
    intersect,
    setdiff,
    setequal,
    union,
)

from ...contexts import Context
from ..dplyr.context import _MultiValueExpr


# ---- intersect ----------------------------------------------------------


@intersect.register(pl.Expr, context=Context.EVAL, backend="polars")
def _intersect_expr(x: pl.Expr, y: Any) -> Any:
    if isinstance(y, pl.Expr):
        return x.list.set_intersection(y)
    return _MultiValueExpr(x.implode().list.set_intersection(y))


@intersect.register(object, backend="polars")
def _intersect_obj(x: Any, y: Any) -> Any:
    if isinstance(x, pl.Series):
        x = x.to_list()
    if isinstance(y, pl.Series):
        y = y.to_list()
    xl = list(x) if hasattr(x, "__iter__") else [x]
    yl = list(y) if hasattr(y, "__iter__") else [y]
    return sorted(
        set(xl) & set(yl),
        key=lambda v: xl.index(v) if v in xl else 999,
    )


# ---- union --------------------------------------------------------------


@union.register(pl.Expr, context=Context.EVAL, backend="polars")
def _union_expr(x: pl.Expr, y: Any) -> Any:
    if isinstance(y, pl.Expr):
        return x.list.set_union(y)
    return _MultiValueExpr(x.implode().list.set_union(y))


@union.register(object, backend="polars")
def _union_obj(x: Any, y: Any) -> Any:
    if isinstance(x, pl.Series):
        x = x.to_list()
    if isinstance(y, pl.Series):
        y = y.to_list()
    seen: set = set()
    result: list = []
    for v in (list(x) if hasattr(x, "__iter__") else [x]) + (
        list(y) if hasattr(y, "__iter__") else [y]
    ):
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


# ---- setdiff ------------------------------------------------------------


@setdiff.register(pl.Expr, context=Context.EVAL, backend="polars")
def _setdiff_expr(x: pl.Expr, y: Any) -> Any:
    if isinstance(y, pl.Expr):
        return x.list.set_difference(y)
    return _MultiValueExpr(x.implode().list.set_difference(y))


@setdiff.register(object, backend="polars")
def _setdiff_obj(x: Any, y: Any) -> Any:
    if isinstance(x, pl.Series):
        x = x.to_list()
    if isinstance(y, pl.Series):
        y = y.to_list()
    xl = list(x) if hasattr(x, "__iter__") else [x]
    ys = set(y) if hasattr(y, "__iter__") else {y}
    return [v for v in xl if v not in ys]


# ---- setequal -----------------------------------------------------------


@setequal.register(pl.Expr, context=Context.EVAL, backend="polars")
def _setequal_expr(x: pl.Expr, y: Any) -> Any:
    if isinstance(y, pl.Expr):
        return (x.list.set_difference(y).list.len() == 0) & (
            y.list.set_difference(x).list.len() == 0
        )
    return (x.implode().list.set_difference(y).list.len() == 0) & (
        pl.lit(y).list.set_difference(x.implode()).list.len() == 0
    )


@setequal.register(object, backend="polars")
def _setequal_obj(x: Any, y: Any) -> Any:
    if isinstance(x, pl.Series):
        x = x.to_list()
    if isinstance(y, pl.Series):
        y = y.to_list()
    xl = list(x) if hasattr(x, "__iter__") else [x]
    yl = list(y) if hasattr(y, "__iter__") else [y]
    return set(xl) == set(yl)


# ---- all_ ---------------------------------------------------------------


@all_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _all_expr(x: pl.Expr) -> pl.Expr:
    return x.all()


@all_.register(object, context=Context.EVAL, backend="polars")
def _all_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.all()
    if isinstance(x, pl.Expr):
        return x.all()
    return all(x)


# ---- any_ ---------------------------------------------------------------


@any_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _any_expr(x: pl.Expr) -> pl.Expr:
    return x.any()


@any_.register(object, context=Context.EVAL, backend="polars")
def _any_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.any()
    if isinstance(x, pl.Expr):
        return x.any()
    return any(x)


# ---- any_na -------------------------------------------------------------


@any_na.register(pl.Expr, context=Context.EVAL, backend="polars")
def _any_na_expr(x: pl.Expr) -> pl.Expr:
    return x.is_null().any()


@any_na.register(object, context=Context.EVAL, backend="polars")
def _any_na_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.is_null().any()
    if isinstance(x, pl.Expr):
        return x.is_null().any()
    return (
        any(v is None for v in x)
        if hasattr(x, "__iter__")
        else x is None
    )


# ---- append -------------------------------------------------------------


@append.register(pl.Expr, context=Context.EVAL, backend="polars")
def _append_expr(
    x: pl.Expr, values: Any, after: Any = None
) -> pl.Expr:
    raise TypeError("append() is not supported on polars Expr directly.")


@append.register(object, context=Context.EVAL, backend="polars")
def _append_obj(x: Any, values: Any, after: Any = None) -> Any:
    if isinstance(x, pl.Series):
        if isinstance(values, pl.Series):
            return x.extend_constant(values, len(values))
        return pl.Series(
            "",
            list(x) + list(values)
            if hasattr(values, "__iter__")
            else list(x) + [values],
        )
    if isinstance(x, pl.Expr):
        raise TypeError(
            "append() is not supported on polars Expr directly."
        )
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        result = list(x)
        if hasattr(values, "__iter__") and not isinstance(
            values, (str, bytes)
        ):
            result.extend(values)
        else:
            result.append(values)
        return result
    return x


# ---- diff ---------------------------------------------------------------


@diff.register(pl.Expr, context=Context.EVAL, backend="polars")
def _diff_expr(
    x: pl.Expr,
    lag: int = 1,
    differences: int = 1,
) -> pl.Expr:
    """Lagged differences of an expression."""
    result = x
    for _ in range(differences):
        result = result.diff(n=lag)
    return result


@diff.register(object, context=Context.EVAL, backend="polars")
def _diff_obj(
    x: Any,
    lag: int = 1,
    differences: int = 1,
) -> Any:
    """Lagged differences of a vector/iterable."""
    if isinstance(x, pl.Series):
        result = x
        for _ in range(differences):
            result = result.diff(n=lag)
        return result
    if isinstance(x, pl.Expr):
        return _diff_expr(x, lag=lag, differences=differences)
    # scalar -> None (R behavior)
    if not hasattr(x, "__iter__") or isinstance(x, (str, bytes)):
        return None
    # convert to Series to get same-length R-style diff (with nulls)
    s = pl.Series("x", list(x), strict=False)
    result = s
    for _ in range(differences):
        result = result.diff(n=lag)
    return result
