"""Cumulative functions for the polars backend.

Implements: cumsum, cummax, cummin, cumprod.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import cumsum, cummax, cummin, cumprod

from ...contexts import Context


@cumsum.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cumsum_expr(x: pl.Expr) -> pl.Expr:
    return x.cum_sum()


@cumsum.register(object, context=Context.EVAL, backend="polars")
def _cumsum_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cum_sum()
    import numpy as np

    return np.cumsum(x)


# ---- cummax -------------------------------------------------------------


@cummax.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cummax_expr(x: pl.Expr) -> pl.Expr:
    return x.cum_max()


@cummax.register(object, context=Context.EVAL, backend="polars")
def _cummax_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cum_max()
    import numpy as np

    return np.maximum.accumulate(x)


# ---- cummin -------------------------------------------------------------


@cummin.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cummin_expr(x: pl.Expr) -> pl.Expr:
    return x.cum_min()


@cummin.register(object, context=Context.EVAL, backend="polars")
def _cummin_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cum_min()
    import numpy as np

    return np.minimum.accumulate(x)


# ---- cumprod ------------------------------------------------------------


@cumprod.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cumprod_expr(x: pl.Expr) -> pl.Expr:
    return x.cum_prod()


@cumprod.register(object, context=Context.EVAL, backend="polars")
def _cumprod_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.cum_prod()
    import numpy as np

    return np.cumprod(x)
