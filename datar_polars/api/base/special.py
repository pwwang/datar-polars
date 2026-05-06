"""Special math functions for the polars backend.

Implements: beta, lgamma, digamma, trigamma, choose, factorial, gamma,
lfactorial, lchoose, lbeta, psigamma.
Requires scipy.
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl

from datar.apis.base import (
    beta,
    lgamma,
    digamma,
    trigamma,
    choose,
    factorial,
    gamma,
    lfactorial,
    lchoose,
    lbeta,
    psigamma,
)

from ...contexts import Context


def _get_special_func(name: str):
    try:
        from scipy import special
    except ImportError as imperr:
        raise ImportError(
            "`special` family requires `scipy` package.\n"
            "Try: pip install -U scipy"
        ) from imperr
    return getattr(special, name)


def _is_iterable(x: Any) -> bool:
    """Check if x is a non-string iterable."""
    if isinstance(x, (str, bytes, pl.Expr)):
        return False
    return hasattr(x, "__iter__")


def _to_list(x: Any) -> list:
    """Convert anything iterable to a plain list."""
    if isinstance(x, pl.Series):
        return x.to_list()
    if _is_iterable(x):
        return list(x)
    return [x]


def _apply_unary(x: Any, fn: Callable) -> Any:
    """Apply fn element-wise: scalar→scalar, list→list, Series→Series."""
    if isinstance(x, pl.Series):
        return pl.Series([fn(v) for v in x.to_list()])
    if _is_iterable(x):
        return [fn(v) for v in x]
    return fn(x)


def _apply_binary(x: Any, y: Any, fn: Callable) -> Any:
    """Apply binary fn element-wise, handling scalar/iterable mixing."""
    x_iter = isinstance(x, pl.Series) or _is_iterable(x)
    y_iter = isinstance(y, pl.Series) or _is_iterable(y)
    if x_iter or y_iter:
        x_vals = _to_list(x)
        y_vals = _to_list(y)
        result = [fn(xv, yv) for xv, yv in zip(x_vals, y_vals)]
        if isinstance(x, pl.Series) or isinstance(y, pl.Series):
            return pl.Series(result)
        return result
    return fn(x, y)


# ---- beta --------------------------------------------------------------


@beta.register(pl.Expr, context=Context.EVAL, backend="polars")
def _beta_expr(x: pl.Expr, y) -> pl.Expr:
    fn = _get_special_func("beta")

    if isinstance(y, pl.Expr):
        return x.map_elements(
            lambda v: float(fn(v[0], v[1])), return_dtype=pl.Float64
        )
    return x.map_elements(lambda v: float(fn(v, y)), return_dtype=pl.Float64)


@beta.register(object, context=Context.EVAL, backend="polars")
def _beta_obj(x: Any, y: Any) -> Any:
    return _apply_binary(x, y, lambda a, b: float(_get_special_func("beta")(a, b)))


# ---- lgamma ------------------------------------------------------------


@lgamma.register(pl.Expr, context=Context.EVAL, backend="polars")
def _lgamma_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(
        lambda v: float(_get_special_func("gammaln")(v)),
        return_dtype=pl.Float64,
    )


@lgamma.register(object, context=Context.EVAL, backend="polars")
def _lgamma_obj(x: Any) -> Any:
    return _apply_unary(
        x, lambda v: float(_get_special_func("gammaln")(v))
    )


# ---- digamma -----------------------------------------------------------


@digamma.register(pl.Expr, context=Context.EVAL, backend="polars")
def _digamma_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(
        lambda v: float(_get_special_func("psi")(v)),
        return_dtype=pl.Float64,
    )


@digamma.register(object, context=Context.EVAL, backend="polars")
def _digamma_obj(x: Any) -> Any:
    return _apply_unary(
        x, lambda v: float(_get_special_func("psi")(v))
    )


# ---- trigamma ----------------------------------------------------------


@trigamma.register(pl.Expr, context=Context.EVAL, backend="polars")
def _trigamma_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(
        lambda v: float(_get_special_func("polygamma")(1, v)),
        return_dtype=pl.Float64,
    )


@trigamma.register(object, context=Context.EVAL, backend="polars")
def _trigamma_obj(x: Any) -> Any:
    return _apply_unary(
        x, lambda v: float(_get_special_func("polygamma")(1, v))
    )


# ---- choose ------------------------------------------------------------


@choose.register(pl.Expr, context=Context.EVAL, backend="polars")
def _choose_expr(n: pl.Expr, k) -> pl.Expr:
    fn = _get_special_func("binom")
    if isinstance(k, pl.Expr):
        return n.map_elements(
            lambda v: float(fn(v[0], v[1])), return_dtype=pl.Float64
        )
    return n.map_elements(lambda v: float(fn(v, k)), return_dtype=pl.Float64)


@choose.register(object, context=Context.EVAL, backend="polars")
def _choose_obj(n: Any, k: Any) -> Any:
    return _apply_binary(
        n, k, lambda a, b: float(_get_special_func("binom")(a, b))
    )


# ---- factorial ---------------------------------------------------------


@factorial.register(pl.Expr, context=Context.EVAL, backend="polars")
def _factorial_expr(x: pl.Expr) -> pl.Expr:
    fn = _get_special_func("factorial")
    return x.map_elements(fn, return_dtype=pl.Float64)


@factorial.register(object, context=Context.EVAL, backend="polars")
def _factorial_obj(x: Any) -> Any:
    return _apply_unary(x, _get_special_func("factorial"))


# ---- gamma -------------------------------------------------------------


@gamma.register(pl.Expr, context=Context.EVAL, backend="polars")
def _gamma_expr(x: pl.Expr) -> pl.Expr:
    return x.map_elements(
        lambda v: float(_get_special_func("gamma")(v)),
        return_dtype=pl.Float64,
    )


@gamma.register(object, context=Context.EVAL, backend="polars")
def _gamma_obj(x: Any) -> Any:
    return _apply_unary(
        x, lambda v: float(_get_special_func("gamma")(v))
    )


# ---- lfactorial --------------------------------------------------------


@lfactorial.register(pl.Expr, context=Context.EVAL, backend="polars")
def _lfactorial_expr(x: pl.Expr) -> pl.Expr:
    fn = _get_special_func("factorial")
    from math import log as _log

    return x.map_elements(lambda v: _log(fn(v)), return_dtype=pl.Float64)


@lfactorial.register(object, context=Context.EVAL, backend="polars")
def _lfactorial_obj(x: Any) -> Any:
    from math import log as _log
    fn = _get_special_func("factorial")

    return _apply_unary(x, lambda v: _log(fn(v)))


# ---- lchoose -----------------------------------------------------------


@lchoose.register(pl.Expr, context=Context.EVAL, backend="polars")
def _lchoose_expr(n: pl.Expr, k) -> pl.Expr:
    fn = _get_special_func("binom")
    from math import log as _log

    if isinstance(k, pl.Expr):
        return n.map_elements(
            lambda v: _log(fn(v[0], v[1])), return_dtype=pl.Float64
        )
    return n.map_elements(lambda v: _log(fn(v, k)), return_dtype=pl.Float64)


@lchoose.register(object, context=Context.EVAL, backend="polars")
def _lchoose_obj(n: Any, k: Any) -> Any:
    from math import log as _log

    return _apply_binary(
        n, k, lambda a, b: _log(_get_special_func("binom")(a, b))
    )


# ---- lbeta -------------------------------------------------------------


@lbeta.register(pl.Expr, context=Context.EVAL, backend="polars")
def _lbeta_expr(x: pl.Expr, y) -> pl.Expr:
    fn = _get_special_func("betaln")

    if isinstance(y, pl.Expr):
        return x.map_elements(
            lambda v: float(fn(v[0], v[1])), return_dtype=pl.Float64
        )
    return x.map_elements(lambda v: float(fn(v, y)), return_dtype=pl.Float64)


@lbeta.register(object, context=Context.EVAL, backend="polars")
def _lbeta_obj(x: Any, y: Any) -> Any:
    return _apply_binary(
        x, y, lambda a, b: float(_get_special_func("betaln")(a, b))
    )


# ---- psigamma ----------------------------------------------------------


@psigamma.register(pl.Expr, context=Context.EVAL, backend="polars")
def _psigamma_expr(x: pl.Expr, deriv: float) -> pl.Expr:
    fn = _get_special_func("polygamma")

    return x.map_elements(
        lambda v: float(fn(round(deriv), v)), return_dtype=pl.Float64
    )


@psigamma.register(object, context=Context.EVAL, backend="polars")
def _psigamma_obj(x: Any, deriv: float) -> Any:
    return _apply_unary(
        x, lambda v: float(_get_special_func("polygamma")(round(deriv), v))
    )
