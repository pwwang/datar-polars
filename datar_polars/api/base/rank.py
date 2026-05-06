"""Ranking and window functions for the polars backend.

Implements: row_number, min_rank, dense_rank, percent_rank, cume_dist, ntile,
lead, lag.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.dplyr import (
    row_number_,
    min_rank_,
    dense_rank_,
    percent_rank_,
    cume_dist_,
    ntile_,
    lead,
    lag,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble


# ---- row_number ---------------------------------------------------------

@row_number_.register(pl.LazyFrame, context=Context.EVAL, backend="polars")
def _row_number_lazy(x: pl.LazyFrame) -> pl.Expr:
    """row_number() inside a verb — returns row index as Expr."""
    return pl.int_range(pl.len()) + 1


@row_number_.register(pl.DataFrame, context=Context.EVAL, backend="polars")
def _row_number_df(x: pl.DataFrame) -> pl.Expr:
    """row_number() on eager data."""
    return pl.int_range(pl.len()) + 1

@row_number_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _row_number_expr(x: pl.Expr) -> pl.Expr:
    return x.cum_count()


@row_number_.register(object, context=Context.EVAL, backend="polars")
def _row_number_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return pl.int_range(1, len(x) + 1, eager=True)
    return 1


# ---- min_rank -----------------------------------------------------------


@min_rank_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _min_rank_expr(x: pl.Expr, *, na_last: str = "keep") -> pl.Expr:
    return pl.when(x.is_not_null() & x.is_not_nan()).then(x.rank("min"))


@min_rank_.register(object, context=Context.EVAL, backend="polars")
def _min_rank_obj(x: Any, *, na_last: str = "keep") -> Any:
    if isinstance(x, pl.Series):
        ranked = x.rank("min")
        na_mask = x.is_null() | x.is_nan()
        if na_mask.any():
            ranked = ranked.set(na_mask, None)
        return ranked
    import scipy.stats

    return scipy.stats.rankdata(x, method="min")


# ---- dense_rank ---------------------------------------------------------


@dense_rank_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _dense_rank_expr(x: pl.Expr, *, na_last: str = "keep") -> pl.Expr:
    return pl.when(x.is_not_null() & x.is_not_nan()).then(x.rank("dense"))


@dense_rank_.register(object, context=Context.EVAL, backend="polars")
def _dense_rank_obj(x: Any, *, na_last: str = "keep") -> Any:
    if isinstance(x, pl.Series):
        ranked = x.rank("dense")
        na_mask = x.is_null() | x.is_nan()
        if na_mask.any():
            ranked = ranked.set(na_mask, None)
        return ranked
    import scipy.stats

    return scipy.stats.rankdata(x, method="dense")


# ---- percent_rank -------------------------------------------------------


@percent_rank_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _percent_rank_expr(x: pl.Expr, *, na_last: str = "keep") -> pl.Expr:
    rank_min = x.rank("min")
    n_non_na = (x.is_not_null() & x.is_not_nan()).sum()
    return pl.when(x.is_not_null() & x.is_not_nan()).then(
        (rank_min - 1).cast(pl.Float64) / (n_non_na - 1).cast(pl.Float64)
    )


@percent_rank_.register(object, context=Context.EVAL, backend="polars")
def _percent_rank_obj(x: Any, *, na_last: str = "keep") -> Any:
    if isinstance(x, pl.Series):
        rank = x.rank("min")
        na_mask = x.is_null() | x.is_nan()
        n = (~na_mask).sum()
        if n <= 1:
            result = pl.Series("", [0.0] * len(x), dtype=pl.Float64)
            if na_mask.any():
                result = result.set(na_mask, None)
            return result
        result = (rank - 1).cast(pl.Float64) / (n - 1)
        if na_mask.any():
            result = result.set(na_mask, None)
        return result
    return 0.0


# ---- cume_dist ----------------------------------------------------------


@cume_dist_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cume_dist_expr(x: pl.Expr, *, na_last: str = "keep") -> pl.Expr:
    n_non_na = (x.is_not_null() & x.is_not_nan()).sum()
    return pl.when(x.is_not_null() & x.is_not_nan()).then(
        x.rank("min").cast(pl.Float64) / n_non_na.cast(pl.Float64)
    )


@cume_dist_.register(object, context=Context.EVAL, backend="polars")
def _cume_dist_obj(x: Any, *, na_last: str = "keep") -> Any:
    if isinstance(x, pl.Series):
        rank = x.rank("min")
        na_mask = x.is_null() | x.is_nan()
        n = (~na_mask).sum()
        if n == 0:
            return pl.Series([float("nan")] * len(x))
        result = rank.cast(pl.Float64) / n
        if na_mask.any():
            result = result.set(na_mask, None)
        return result
    return 1.0


# ---- ntile --------------------------------------------------------------


@ntile_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _ntile_expr(x: pl.Expr, *, n: int = None) -> pl.Expr:
    if n is None:
        raise ValueError("`n` must be provided for ntile()")
    return pl.when(x.is_not_null() & x.is_not_nan()).then(
        x.qcut(n).to_physical() + 1
    )


@ntile_.register(object, context=Context.EVAL, backend="polars")
def _ntile_obj(x: Any, *, n: int = None) -> Any:
    if n is None:
        raise ValueError("`n` must be provided for ntile()")
    if isinstance(x, pl.Series):
        na_mask = x.is_null() | x.is_nan()
        result = x.qcut(n).to_physical().cast(pl.Int64) + 1
        if na_mask.any():
            result = result.set(na_mask, None)
        return result
    import numpy as np

    arr = np.asarray(x)
    result = np.zeros(len(arr), dtype=int)
    bins = np.array_split(np.arange(len(arr)), n)
    for i, bin_indices in enumerate(bins):
        if len(bin_indices) > 0:
            result[bin_indices] = i + 1
    return result


# ---- lead ---------------------------------------------------------------


@lead.register(pl.Expr, context=Context.EVAL, backend="polars")
def _lead_expr(
    x: pl.Expr,
    n: int = 1,
    default: Any = None,
    order_by: Any = None,
) -> pl.Expr:
    result = x.shift(-n)
    if default is not None:
        result = result.fill_null(default)
    return result


@lead.register(object, context=Context.EVAL, backend="polars")
def _lead_obj(
    x: Any,
    n: int = 1,
    default: Any = None,
    order_by: Any = None,
) -> Any:
    if not isinstance(n, int):
        raise ValueError("`lead-lag` expect an integer for `n`.")
    if isinstance(x, pl.Series):
        result = x.shift(-n)
        if default is not None:
            result = result.fill_null(default)
        return result
    # Convert to Series for uniform handling
    vals = list(x) if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)) else [x]
    s = pl.Series(vals)
    result = s.shift(-n)
    if default is not None:
        result = result.fill_null(default)
    return result


# ---- lag ----------------------------------------------------------------


@lag.register(pl.Expr, context=Context.EVAL, backend="polars")
def _lag_expr(
    x: pl.Expr,
    n: int = 1,
    default: Any = None,
    order_by: Any = None,
) -> pl.Expr:
    result = x.shift(n)
    if default is not None:
        result = result.fill_null(default)
    return result


@lag.register(object, context=Context.EVAL, backend="polars")
def _lag_obj(
    x: Any,
    n: int = 1,
    default: Any = None,
    order_by: Any = None,
) -> Any:
    if isinstance(x, pl.Series):
        result = x.shift(n)
        if default is not None:
            result = result.fill_null(default)
        return result
    # Convert to Series for uniform handling
    vals = list(x) if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)) else [x]
    s = pl.Series(vals)
    result = s.shift(n)
    if default is not None:
        result = result.fill_null(default)
    return result


# ---- bare-named aliases -------------------------------------------------

row_number = row_number_
min_rank = min_rank_
dense_rank = dense_rank_
cume_dist = cume_dist_
percent_rank = percent_rank_
ntile = ntile_
