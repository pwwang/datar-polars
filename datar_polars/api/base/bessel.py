"""Bessel functions for the polars backend.

Implements: bessel_i, bessel_j, bessel_k, bessel_y.
Requires scipy.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import bessel_i, bessel_j, bessel_k, bessel_y

from ...contexts import Context


def _get_special_func(name: str):
    try:
        from scipy import special
    except ImportError as imperr:
        raise ImportError(
            "`bessel` family requires `scipy` package.\n"
            "Try: pip install -U scipy"
        ) from imperr
    return getattr(special, name)


def _bessel_i_impl(x, nu: float, expon_scaled: bool = False):
    if nu not in (0, 1):
        fn_name = "ive" if expon_scaled else "iv"
        return _get_special_func(fn_name)(nu, x)
    if expon_scaled:
        fn_name = "i0e" if nu == 0 else "i1e"
    else:
        fn_name = "i0" if nu == 0 else "i1"
    return _get_special_func(fn_name)(x)


def _bessel_j_impl(x, nu: float):
    if nu not in (0, 1):
        return _get_special_func("jv")(nu, x)
    fn_name = "j0" if nu == 0 else "j1"
    return _get_special_func(fn_name)(x)


def _bessel_k_impl(x, nu: float, expon_scaled: bool = False):
    if nu not in (0, 1):
        fn_name = "kve" if expon_scaled else "kv"
        return _get_special_func(fn_name)(nu, x)
    if expon_scaled:
        fn_name = "k0e" if nu == 0 else "k1e"
    else:
        fn_name = "k0" if nu == 0 else "k1"
    return _get_special_func(fn_name)(x)


def _bessel_y_impl(x, nu: float):
    if nu not in (0, 1):
        return _get_special_func("yv")(nu, x)
    fn_name = "y0" if nu == 0 else "y1"
    return _get_special_func(fn_name)(x)


# ---- bessel_i ----------------------------------------------------------


@bessel_i.register(pl.Expr, context=Context.EVAL, backend="polars")
def _bessel_i_expr(
    x: pl.Expr, nu: float, expon_scaled: bool = False
) -> pl.Expr:
    return x.map_elements(
        lambda v: _bessel_i_impl(v, nu, expon_scaled),
        return_dtype=pl.Float64,
    )


@bessel_i.register(object, context=Context.EVAL, backend="polars")
def _bessel_i_obj(x: Any, nu: float, expon_scaled: bool = False) -> Any:
    if isinstance(x, pl.Series):
        return x.map_elements(
            lambda v: _bessel_i_impl(v, nu, expon_scaled),
            return_dtype=pl.Float64,
        )
    return _bessel_i_impl(x, nu, expon_scaled)


# ---- bessel_j ----------------------------------------------------------


@bessel_j.register(pl.Expr, context=Context.EVAL, backend="polars")
def _bessel_j_expr(x: pl.Expr, nu: float) -> pl.Expr:
    return x.map_elements(
        lambda v: _bessel_j_impl(v, nu), return_dtype=pl.Float64
    )


@bessel_j.register(object, context=Context.EVAL, backend="polars")
def _bessel_j_obj(x: Any, nu: float) -> Any:
    if isinstance(x, pl.Series):
        return x.map_elements(
            lambda v: _bessel_j_impl(v, nu), return_dtype=pl.Float64
        )
    return _bessel_j_impl(x, nu)


# ---- bessel_k ----------------------------------------------------------


@bessel_k.register(pl.Expr, context=Context.EVAL, backend="polars")
def _bessel_k_expr(
    x: pl.Expr, nu: float, expon_scaled: bool = False
) -> pl.Expr:
    return x.map_elements(
        lambda v: _bessel_k_impl(v, nu, expon_scaled),
        return_dtype=pl.Float64,
    )


@bessel_k.register(object, context=Context.EVAL, backend="polars")
def _bessel_k_obj(x: Any, nu: float, expon_scaled: bool = False) -> Any:
    if isinstance(x, pl.Series):
        return x.map_elements(
            lambda v: _bessel_k_impl(v, nu, expon_scaled),
            return_dtype=pl.Float64,
        )
    return _bessel_k_impl(x, nu, expon_scaled)


# ---- bessel_y ----------------------------------------------------------


@bessel_y.register(pl.Expr, context=Context.EVAL, backend="polars")
def _bessel_y_expr(x: pl.Expr, nu: float) -> pl.Expr:
    return x.map_elements(
        lambda v: _bessel_y_impl(v, nu), return_dtype=pl.Float64
    )


@bessel_y.register(object, context=Context.EVAL, backend="polars")
def _bessel_y_obj(x: Any, nu: float) -> Any:
    if isinstance(x, pl.Series):
        return x.map_elements(
            lambda v: _bessel_y_impl(v, nu), return_dtype=pl.Float64
        )
    return _bessel_y_impl(x, nu)
