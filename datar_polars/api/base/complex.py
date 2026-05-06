"""Complex number functions for the polars backend.

Implements: arg, conj, re_, im.
Note: mod (modulus) is handled by arithm.py.
"""

from __future__ import annotations

import cmath
from typing import Any

import polars as pl

from datar.apis.base import arg, as_complex, conj, im, is_complex, re_

from ...contexts import Context


def _is_iterable(x: Any) -> bool:
    """Check if x is a non-string iterable (list, tuple, Series, etc.)."""
    if isinstance(x, (str, bytes, pl.Expr)):
        return False
    return hasattr(x, "__iter__")


def _maybe_series(x: Any) -> pl.Series:
    """Convert a list/iterable to pl.Series if not already one."""
    if isinstance(x, pl.Series):
        return x
    if _is_iterable(x):
        return pl.Series(list(x))
    return None


def _handle_obj(x: Any, scalar_fn, series_fn=None):
    """Route to scalar or Series handler based on input type."""
    if isinstance(x, pl.Series):
        if series_fn:
            return series_fn(x)
        return pl.Series([scalar_fn(v) for v in x.to_list()])
    if _is_iterable(x):
        if series_fn:
            return series_fn(pl.Series(list(x)))
        return pl.Series([scalar_fn(v) for v in x])
    return scalar_fn(x)


# ---- arg ---------------------------------------------------------------


@arg.register(pl.Expr, context=Context.EVAL, backend="polars")
def _arg_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(cmath.phase, return_dtype=pl.Float64)


@arg.register(object, context=Context.EVAL, backend="polars")
def _arg_obj(x: Any) -> Any:
    return _handle_obj(x, cmath.phase)


# ---- conj --------------------------------------------------------------


@conj.register(pl.Expr, context=Context.EVAL, backend="polars")
def _conj_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(lambda v: v.conjugate(), return_dtype=pl.Object)


@conj.register(object, context=Context.EVAL, backend="polars")
def _conj_obj(x: Any) -> Any:
    return _handle_obj(
        x,
        lambda v: v.conjugate(),
        lambda s: s.map_elements(lambda v: v.conjugate(), return_dtype=pl.Object),
    )


# ---- re_ ---------------------------------------------------------------


@re_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _re_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(lambda v: v.real, return_dtype=pl.Float64)


@re_.register(object, context=Context.EVAL, backend="polars")
def _re_obj(x: Any) -> Any:
    return _handle_obj(x, lambda v: v.real)


# ---- im ----------------------------------------------------------------


@im.register(pl.Expr, context=Context.EVAL, backend="polars")
def _im_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(lambda v: v.imag, return_dtype=pl.Float64)


@im.register(object, context=Context.EVAL, backend="polars")
def _im_obj(x: Any) -> Any:
    return _handle_obj(x, lambda v: v.imag)


# ---- as_complex ----------------------------------------------------------


@as_complex.register(pl.Expr, context=Context.EVAL, backend="polars")
def _as_complex_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(
        lambda v: complex(float(v)) if v is not None else None,
        return_dtype=pl.Object,
    )


@as_complex.register(object, context=Context.EVAL, backend="polars")
def _as_complex_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.map_elements(
            lambda v: complex(float(v)) if v is not None else None,
            return_dtype=pl.Object,
        )
    try:
        return complex(x)
    except (TypeError, ValueError):
        return None


# ---- is_complex ----------------------------------------------------------


@is_complex.register(object, context=Context.EVAL, backend="polars")
def _is_complex(x: Any) -> Any:
    if isinstance(x, pl.Expr):
        return False
    if isinstance(x, pl.Series):
        return x.dtype == pl.Object
    return isinstance(x, complex)
