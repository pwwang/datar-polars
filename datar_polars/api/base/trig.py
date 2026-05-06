"""Trig/math functions for the polars backend.

Implements: acos, acosh, asin, asinh, atan, atanh, atan2,
cos, cosh, cospi, sin, sinh, sinpi, tan, tanh, tanpi.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import (
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    atan2,
    cos,
    cosh,
    cospi,
    sin,
    sinh,
    sinpi,
    tan,
    tanh,
    tanpi,
)

from ...contexts import Context


import math

# ---- acos ---------------------------------------------------------------


@acos.register(pl.Expr, context=Context.EVAL, backend="polars")
def _acos_expr(x: pl.Expr) -> pl.Expr:
    return x.arccos()


@acos.register(object, context=Context.EVAL, backend="polars")
def _acos_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.arccos()
    import math

    return math.acos(x)


# ---- acosh --------------------------------------------------------------


@acosh.register(pl.Expr, context=Context.EVAL, backend="polars")
def _acosh_expr(x: pl.Expr) -> pl.Expr:
    return x.arccosh()


@acosh.register(object, context=Context.EVAL, backend="polars")
def _acosh_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.arccosh()
    import math

    return math.acosh(x)


# ---- asin ---------------------------------------------------------------


@asin.register(pl.Expr, context=Context.EVAL, backend="polars")
def _asin_expr(x: pl.Expr) -> pl.Expr:
    return x.arcsin()


@asin.register(object, context=Context.EVAL, backend="polars")
def _asin_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.arcsin()
    import math

    return math.asin(x)


# ---- asinh --------------------------------------------------------------


@asinh.register(pl.Expr, context=Context.EVAL, backend="polars")
def _asinh_expr(x: pl.Expr) -> pl.Expr:
    return x.arcsinh()


@asinh.register(object, context=Context.EVAL, backend="polars")
def _asinh_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.arcsinh()
    import math

    return math.asinh(x)


# ---- atan ---------------------------------------------------------------


@atan.register(pl.Expr, context=Context.EVAL, backend="polars")
def _atan_expr(x: pl.Expr) -> pl.Expr:
    return x.arctan()


@atan.register(object, context=Context.EVAL, backend="polars")
def _atan_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.arctan()
    import math

    return math.atan(x)


# ---- atanh --------------------------------------------------------------


@atanh.register(pl.Expr, context=Context.EVAL, backend="polars")
def _atanh_expr(x: pl.Expr) -> pl.Expr:
    return x.arctanh()


@atanh.register(object, context=Context.EVAL, backend="polars")
def _atanh_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.arctanh()
    import math

    return math.atanh(x)


# ---- atan2 --------------------------------------------------------------


@atan2.register(pl.Expr, context=Context.EVAL, backend="polars")
def _atan2_expr(y: pl.Expr, x: pl.Expr) -> pl.Expr:
    return pl.arctan2(y, x)


@atan2.register(object, context=Context.EVAL, backend="polars")
def _atan2_obj(y: Any, x: Any) -> Any:
    if isinstance(y, pl.Series) and isinstance(x, pl.Series):
        import numpy as np
        return np.arctan2(y.to_numpy(), x.to_numpy())
    import math

    return math.atan2(y, x)


# ---- cos ----------------------------------------------------------------


@cos.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cos_expr(x: pl.Expr) -> pl.Expr:
    return x.cos()


@cos.register(object, context=Context.EVAL, backend="polars")
def _cos_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cos()
    import math

    return math.cos(x)


# ---- cosh ---------------------------------------------------------------


@cosh.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cosh_expr(x: pl.Expr) -> pl.Expr:
    return x.cosh()


@cosh.register(object, context=Context.EVAL, backend="polars")
def _cosh_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cosh()
    import math

    return math.cosh(x)


# ---- cospi --------------------------------------------------------------


@cospi.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cospi_expr(x: pl.Expr) -> pl.Expr:
    return (x * pl.lit(math.pi)).cos()


@cospi.register(object, context=Context.EVAL, backend="polars")
def _cospi_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return (x * math.pi).cos()
    from math import cos as _cos

    return _cos(x * math.pi)


# ---- sin ----------------------------------------------------------------


@sin.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sin_expr(x: pl.Expr) -> pl.Expr:
    return x.sin()


@sin.register(object, context=Context.EVAL, backend="polars")
def _sin_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.sin()
    import math

    return math.sin(x)


# ---- sinh ---------------------------------------------------------------


@sinh.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sinh_expr(x: pl.Expr) -> pl.Expr:
    return x.sinh()


@sinh.register(object, context=Context.EVAL, backend="polars")
def _sinh_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.sinh()
    import math

    return math.sinh(x)


# ---- sinpi --------------------------------------------------------------


@sinpi.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sinpi_expr(x: pl.Expr) -> pl.Expr:
    return (x * pl.lit(math.pi)).sin()


@sinpi.register(object, context=Context.EVAL, backend="polars")
def _sinpi_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return (x * math.pi).sin()
    from math import sin as _sin

    return _sin(x * math.pi)


# ---- tan ----------------------------------------------------------------


@tan.register(pl.Expr, context=Context.EVAL, backend="polars")
def _tan_expr(x: pl.Expr) -> pl.Expr:
    return x.tan()


@tan.register(object, context=Context.EVAL, backend="polars")
def _tan_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.tan()
    import math

    return math.tan(x)


# ---- tanh ---------------------------------------------------------------


@tanh.register(pl.Expr, context=Context.EVAL, backend="polars")
def _tanh_expr(x: pl.Expr) -> pl.Expr:
    return x.tanh()


@tanh.register(object, context=Context.EVAL, backend="polars")
def _tanh_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.tanh()
    import math

    return math.tanh(x)


# ---- tanpi --------------------------------------------------------------


@tanpi.register(pl.Expr, context=Context.EVAL, backend="polars")
def _tanpi_expr(x: pl.Expr) -> pl.Expr:
    return (x * pl.lit(math.pi)).tan()


@tanpi.register(object, context=Context.EVAL, backend="polars")
def _tanpi_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return (x * math.pi).tan()
    from math import tan as _tan

    return _tan(x * math.pi)
