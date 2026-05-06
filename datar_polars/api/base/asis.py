"""Type-check functions for the polars backend.

Implements: is_na, is_finite, is_infinite, is_null, is_numeric, is_integer,
is_character.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import (
    is_na,
    is_finite,
    is_infinite,
    is_null,
    is_numeric,
    is_integer,
    is_character,
)
from ...utils import is_null as utils_is_null
from ...contexts import Context


def _to_series(x: Any) -> pl.Series:
    if isinstance(x, pl.Series):
        return x
    if isinstance(x, pl.Expr):
        raise TypeError("Cannot convert pl.Expr to Series")
    return pl.Series(
        "", [x] if not hasattr(x, "__len__") or isinstance(x, (str, bytes)) else x
    )


# ---- is_na --------------------------------------------------------------


@is_na.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_na_expr(x: pl.Expr) -> pl.Expr:
    return x.is_null()


@is_na.register(object, context=Context.EVAL, backend="polars")
def _is_na_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.is_null()
    if isinstance(x, pl.Expr):
        return x.is_null()
    return x is None


# ---- is_finite ----------------------------------------------------------


@is_finite.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_finite_expr(x: pl.Expr) -> pl.Expr:
    return x.is_finite()


@is_finite.register(object, context=Context.EVAL, backend="polars")
def _is_finite_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.is_finite()
    if isinstance(x, pl.Expr):
        return x.is_finite()
    import math
    return math.isfinite(x)


# ---- is_infinite --------------------------------------------------------


@is_infinite.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_infinite_expr(x: pl.Expr) -> pl.Expr:
    return x.is_infinite()


@is_infinite.register(object, context=Context.EVAL, backend="polars")
def _is_infinite_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.is_infinite()
    if isinstance(x, pl.Expr):
        return x.is_infinite()
    import math
    return math.isinf(x)


# ---- is_null ------------------------------------------------------------


@is_null.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_null_expr(x: pl.Expr) -> pl.Expr:
    return utils_is_null(x)


@is_null.register(object, context=Context.EVAL, backend="polars")
def _is_null_obj(x: Any) -> Any:
    return utils_is_null(x)


# ---- is_numeric ---------------------------------------------------------


@is_numeric.register(object, context=Context.EVAL, backend="polars")
def _is_numeric(x: Any) -> Any:
    """Check if x is numeric.

    For pl.Expr, we return True (cannot check dtype at expression level).
    For pl.Series, we check if the dtype is numeric.
    For scalars, we check if it's a number type.
    """
    if isinstance(x, pl.Expr):
        # At expression level, we can't know the dtype.
        # Return True and let it fail at collect time if needed.
        return True
    if isinstance(x, pl.Series):
        return x.dtype.is_numeric()
    if isinstance(x, (list, tuple)):
        return all(_is_numeric(e) for e in x)
    import numbers

    return isinstance(x, (int, float, numbers.Number))


# ---- is_integer ---------------------------------------------------------


@is_integer.register(object, context=Context.EVAL, backend="polars")
def _is_integer(x: Any) -> Any:
    """Check if x is an integer type."""
    if isinstance(x, pl.Expr):
        return True
    if isinstance(x, pl.Series):
        return x.dtype.is_integer()
    if isinstance(x, (list, tuple)):
        return all(_is_integer(e) for e in x)
    import numbers

    return isinstance(x, (int, numbers.Integral)) and not isinstance(x, bool)


# ---- is_character -------------------------------------------------------


@is_character.register(object, context=Context.EVAL, backend="polars")
def _is_character(x: Any) -> Any:
    """Check if x is a character/string type."""
    if isinstance(x, pl.Expr):
        return True
    if isinstance(x, pl.Series):
        return x.dtype == pl.Utf8 or x.dtype == pl.String
    if isinstance(x, (list, tuple)):
        return all(_is_character(e) for e in x)
    return isinstance(x, str)
