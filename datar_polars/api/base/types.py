"""Type conversion and checking functions for the polars backend.

Implements: as_character, as_double, as_integer, as_logical, as_numeric,
is_atomic, is_character, is_double, is_element, is_false, is_integer,
is_logical, is_true.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import (
    as_character,
    as_date,
    as_double,
    as_integer,
    as_logical,
    as_null,
    as_numeric,
    is_atomic,
    is_character,
    is_double,
    is_element,
    is_false,
    is_integer,
    is_logical,
    is_true,
)

from ...contexts import Context


def _is_iterable(x: Any) -> bool:
    """Check if x is a non-string iterable (list, tuple, etc.)."""
    if isinstance(x, (str, bytes, pl.Expr, pl.Series)):
        return False
    return hasattr(x, "__iter__")


def _to_series(x: Any) -> pl.Series | None:
    """Convert a list/iterable to pl.Series if not already one."""
    if isinstance(x, pl.Series):
        return x
    if _is_iterable(x):
        return pl.Series(list(x), strict=False)
    return None

# ── as_character ────────────────────────────────────────────────────────


@as_character.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_character_expr(x: pl.Expr) -> pl.Expr:
    return x.cast(pl.Utf8)


@as_character.register(object, context=Context.EVAL, backend="polars")
def _as_character_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cast(pl.Utf8)
    s = _to_series(x)
    if s is not None:
        return s.cast(pl.Utf8)
    return str(x)


# ── as_double ───────────────────────────────────────────────────────────


@as_double.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_double_expr(x: pl.Expr) -> pl.Expr:
    return x.cast(pl.Float64)


@as_double.register(object, context=Context.EVAL, backend="polars")
def _as_double_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cast(pl.Float64)
    s = _to_series(x)
    if s is not None:
        return s.cast(pl.Float64)
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ── as_integer ──────────────────────────────────────────────────────────


@as_integer.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_integer_expr(x: pl.Expr) -> pl.Expr:
    return x.cast(pl.Int64)


@as_integer.register(object, context=Context.EVAL, backend="polars")
def _as_integer_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cast(pl.Int64)
    s = _to_series(x)
    if s is not None:
        return s.cast(pl.Int64)
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# ── as_logical ──────────────────────────────────────────────────────────


@as_logical.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_logical_expr(x: pl.Expr) -> pl.Expr:
    return x.cast(pl.Boolean)


@as_logical.register(object, context=Context.EVAL, backend="polars")
def _as_logical_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cast(pl.Boolean)
    s = _to_series(x)
    if s is not None:
        return s.cast(pl.Boolean)
    return bool(x)


# ── as_numeric ──────────────────────────────────────────────────────────


@as_numeric.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_numeric_expr(x: pl.Expr) -> pl.Expr:
    return x.cast(pl.Float64)


@as_numeric.register(object, context=Context.EVAL, backend="polars")
def _as_numeric_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cast(pl.Float64)
    s = _to_series(x)
    if s is not None:
        return s.cast(pl.Float64)
    try:
        return float(x)
    except (TypeError, ValueError):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None


# ── is_atomic ───────────────────────────────────────────────────────────


@is_atomic.register(object, context=Context.EVAL, backend="polars")
def _is_atomic(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return True
    if isinstance(x, pl.Expr):
        return True
    return isinstance(x, (int, float, str, bool, complex, bytes))


# ── is_character ────────────────────────────────────────────────────────


@is_character.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_character_expr(x: pl.Expr) -> pl.Expr:
    return pl.lit(False)


@is_character.register(object, context=Context.EVAL, backend="polars")
def _is_character_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.dtype in (pl.Utf8, pl.Categorical)
    if isinstance(x, pl.Expr):
        return False
    if isinstance(x, (list, tuple)):
        return all(is_character(e) for e in x)
    return isinstance(x, str)


# ── is_double ───────────────────────────────────────────────────────────


@is_double.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_double_expr(x: pl.Expr) -> pl.Expr:
    return pl.lit(False)


@is_double.register(object, context=Context.EVAL, backend="polars")
def _is_double_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.dtype in (pl.Float32, pl.Float64)
    if isinstance(x, pl.Expr):
        return False
    return isinstance(x, float)


# ── is_element ──────────────────────────────────────────────────────────


@is_element.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_element_expr(x: pl.Expr, y: Any) -> pl.Expr:
    if isinstance(y, pl.Expr):
        return x.is_in(y)
    if isinstance(y, pl.Series):
        return x.is_in(pl.lit(y))
    return x.is_in(y)


@is_element.register(object, context=Context.EVAL, backend="polars")
def _is_element_obj(x: Any, y: Any) -> Any:
    if isinstance(x, pl.Series):
        if isinstance(y, pl.Series):
            return x.is_in(y)
        return x.is_in(y)
    if isinstance(x, pl.Expr):
        return _is_element_expr(x, y)
    if isinstance(y, pl.Series):
        return x in y.to_list()
    if isinstance(y, pl.Expr):
        return False
    try:
        return x in y
    except TypeError:
        return False


# ── is_false ────────────────────────────────────────────────────────────


@is_false.register(object, context=Context.EVAL, backend="polars")
def _is_false(x: Any) -> Any:
    return x is False


# ── is_integer ──────────────────────────────────────────────────────────


@is_integer.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_integer_expr(x: pl.Expr) -> pl.Expr:
    return pl.lit(False)


@is_integer.register(object, context=Context.EVAL, backend="polars")
def _is_integer_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
        )
    if isinstance(x, pl.Expr):
        return False
    if isinstance(x, (list, tuple)):
        return all(is_integer(e) for e in x)
    return isinstance(x, int) and not isinstance(x, bool)


# ── is_logical ──────────────────────────────────────────────────────────


@is_logical.register(pl.Expr, context=Context.EVAL, backend="polars")
def _is_logical_expr(x: pl.Expr) -> pl.Expr:
    return pl.lit(False)


@is_logical.register(object, context=Context.EVAL, backend="polars")
def _is_logical_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.dtype == pl.Boolean
    if isinstance(x, pl.Expr):
        return False
    return isinstance(x, bool)


# ── is_true ─────────────────────────────────────────────────────────────


@is_true.register(object, context=Context.EVAL, backend="polars")
def _is_true(x: Any) -> Any:
    return x is True


# ── as_null ──────────────────────────────────────────────────────────────


@as_null.register(object, context=Context.EVAL, backend="polars")
def _as_null(x: Any) -> Any:
    return pl.Null


# ── as_date ───────────────────────────────────────────────────────────────


@as_date.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_date_expr(
    x: pl.Expr,
    format: Any = None,
    try_formats: Any = None,
    optional: bool = False,
    tz: Any = 0,
    origin: Any = "1970-01-01",
) -> pl.Expr:
    """Convert an expression to Date type."""
    if x.dtype.is_temporal():
        return x.cast(pl.Date)
    return x.str.to_date(format=format, strict=not optional)


@as_date.register(object, context=Context.EVAL, backend="polars")
def _as_date_obj(
    x: Any,
    format: Any = None,
    try_formats: Any = None,
    optional: bool = False,
    tz: Any = 0,
    origin: Any = "1970-01-01",
) -> Any:
    """Convert to Date type."""
    if isinstance(x, pl.Series):
        if x.dtype.is_temporal():
            return x.cast(pl.Date)
        try:
            if format is not None:
                return x.str.to_date(format=format, strict=not optional)
            return x.str.to_date(strict=not optional)
        except Exception:
            return x.cast(pl.Date)
    if isinstance(x, pl.Expr):
        return _as_date_expr(x, format, try_formats, optional, tz, origin)
    import datetime

    if isinstance(x, (int, float)):
        return datetime.date.fromtimestamp(x)
    if isinstance(x, str):
        from datetime import datetime as dt

        if format:
            return dt.strptime(x, format).date()
        return dt.fromisoformat(x).date()
    return x
