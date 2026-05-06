"""Windowed rank functions.

See https://github.com/tidyverse/dplyr/blob/master/R/rank.R
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.dplyr import (
    row_number_,
    ntile_,
    min_rank_,
    dense_rank_,
    percent_rank_,
    cume_dist_,
)

from ...polars import Series


# ── row_number ──────────────────────────────────────────────────────────────

@row_number_.register(object, backend="polars")
def _row_number(x: Any) -> Any:
    """Row number of x."""
    if isinstance(x, pl.Expr):
        return x.cum_count()
    if isinstance(x, Series):
        return pl.Series(range(1, len(x) + 1))
    if hasattr(x, "__len__"):
        return list(range(1, len(x) + 1))
    return 1


# ── ntile ───────────────────────────────────────────────────────────────────


@ntile_.register(object, backend="polars")
def _ntile(x: Any, *, n: int = None) -> Any:
    """Rough rank breaking input into n buckets."""
    if n is None:
        raise ValueError("`n` must be provided for ntile().")
    if isinstance(x, pl.Expr):
        # Use polars native qcut — to_physical() gives integer bin codes
        return pl.when(x.is_not_null() & x.is_not_nan()).then(
            x.qcut(n).to_physical() + 1
        )
    if isinstance(x, Series):
        na_mask = x.is_null() | x.is_nan()
        result = x.qcut(n).to_physical().cast(pl.Int64) + 1
        if na_mask.any():
            result = result.set(na_mask, None)
        return result
    # Generic case: use numpy
    import numpy as np

    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    na_mask = np.isnan(arr)
    non_na = (~na_mask).sum()
    if non_na == 0:
        return np.full(arr.size, np.nan)
    n = min(n, non_na)
    ranked = np.zeros(arr.size)
    ranked[~na_mask] = (
        np.floor(
            (np.argsort(np.argsort(arr[~na_mask])) * n / non_na)
        )
        + 1
    )
    ranked[na_mask] = np.nan
    return ranked


# ── min_rank ────────────────────────────────────────────────────────────────


@min_rank_.register(object, backend="polars")
def _min_rank(x: Any, *, na_last: str = "keep") -> Any:
    """Minimum rank."""
    return _rank_impl(x, na_last=na_last, method="min")


# ── dense_rank ──────────────────────────────────────────────────────────────


@dense_rank_.register(object, backend="polars")
def _dense_rank(x: Any, *, na_last: str = "keep") -> Any:
    """Dense rank."""
    return _rank_impl(x, na_last=na_last, method="dense")


# ── percent_rank ────────────────────────────────────────────────────────────


@percent_rank_.register(object, backend="polars")
def _percent_rank(x: Any, *, na_last: str = "keep") -> Any:
    """Percent rank."""
    if isinstance(x, pl.Expr):
        rank_min = x.rank(method="min", descending=False)
        n_non_na = (x.is_not_null() & x.is_not_nan()).sum()
        return pl.when(x.is_not_null() & x.is_not_nan()).then(
            (rank_min - 1).cast(pl.Float64) / (n_non_na - 1).cast(pl.Float64)
        )
    if isinstance(x, Series):
        ranked = x.rank(method="min")
        na_mask = x.is_null() | x.is_nan()
        total = (~na_mask).sum()
        if total <= 1:
            result = pl.Series([0.0] * len(x))
            if na_mask.any():
                result = result.set(na_mask, None)
            return result
        result = (ranked - 1).cast(pl.Float64) / (total - 1)
        if na_mask.any():
            result = result.set(na_mask, None)
        return result
    import numpy as np

    arr = np.asarray(list(x), dtype=float)
    if arr.size <= 1:
        return np.zeros(arr.size)
    na_mask = np.isnan(arr)
    ranked = np.zeros(arr.size)
    non_na = (~na_mask).sum()
    if non_na <= 1:
        result = np.zeros(arr.size)
        result[na_mask] = np.nan
        return result
    ranked[~na_mask] = np.argsort(np.argsort(arr[~na_mask])).astype(float)
    ranked[na_mask] = np.nan
    result = np.zeros(arr.size)
    result[~na_mask] = ranked[~na_mask] / (non_na - 1)
    result[na_mask] = np.nan
    return result


# ── cume_dist ───────────────────────────────────────────────────────────────


@cume_dist_.register(object, backend="polars")
def _cume_dist(x: Any, *, na_last: str = "keep") -> Any:
    """Cumulative distribution."""
    if isinstance(x, pl.Expr):
        n_non_na = (x.is_not_null() & x.is_not_nan()).sum()
        return pl.when(x.is_not_null() & x.is_not_nan()).then(
            x.rank(method="min").cast(pl.Float64) / n_non_na.cast(pl.Float64)
        )
    if isinstance(x, Series):
        ranked = x.rank(method="min")
        na_mask = x.is_null() | x.is_nan()
        total = (~na_mask).sum()
        if total == 0:
            return pl.Series([float("nan")] * len(x))
        result = ranked.cast(pl.Float64) / total
        if na_mask.any():
            result = result.set(na_mask, None)
        return result
    import numpy as np

    arr = np.asarray(list(x), dtype=float)
    na_mask = np.isnan(arr)
    ranked = np.zeros(arr.size)
    non_na = (~na_mask).sum()
    if non_na == 0:
        return np.full(arr.size, np.nan)
    ranked[~na_mask] = np.argsort(np.argsort(arr[~na_mask])).astype(float) + 1
    ranked[na_mask] = np.nan
    result = np.zeros(arr.size)
    result[~na_mask] = ranked[~na_mask] / non_na
    result[na_mask] = np.nan
    return result


# ── shared rank implementation ──────────────────────────────────────────────


def _rank_impl(x: Any, na_last: str = "keep", method: str = "min") -> Any:
    """Generic rank implementation wrapping polars."""
    if isinstance(x, pl.Expr):
        return pl.when(x.is_not_null() & x.is_not_nan()).then(
            x.rank(method=method, descending=False)
        )
    if isinstance(x, Series):
        ranked = x.rank(method=method, descending=False)
        na_mask = x.is_null() | x.is_nan()
        if na_mask.any():
            ranked = ranked.set(na_mask, None)
        return ranked
    import numpy as np

    # Use polars Series rank for generic objects
    s = pl.Series(x)
    result = s.rank(method=method, descending=False)
    if isinstance(x, np.ndarray):
        return result.to_numpy()
    return result.to_list()
