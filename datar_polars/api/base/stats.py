"""Column-wise and row-wise statistics for the polars backend.

Implements: col_sums, row_sums, col_means, row_means,
col_sds, row_sds, col_medians, row_medians.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.apis.base import (
    col_means,
    col_medians,
    col_sds,
    col_sums,
    row_means,
    row_medians,
    row_sds,
    row_sums,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, as_tibble


# ---- col_sums -----------------------------------------------------------


@col_sums.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _col_sums_tibble(x: Tibble, na_rm: bool | None = None) -> pl.Series:
    """Column sums. Returns a Series of sums per column."""
    return x.select(pl.all().sum()).collect().row(0)


@col_sums.register(object, context=Context.EVAL, backend="polars")
def _col_sums_obj(x: Any, na_rm: bool | None = None) -> Any:
    if isinstance(x, pl.Series):
        return x.sum()
    if isinstance(x, pl.Expr):
        return x.sum()
    return sum(x)


# ---- row_sums -----------------------------------------------------------


@row_sums.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _row_sums_tibble(x: Tibble, na_rm: bool | None = None) -> pl.Series:
    """Row sums. Returns a Series of sums per row."""
    return x.select(pl.sum_horizontal(pl.all())).collect().to_series()


@row_sums.register(object, context=Context.EVAL, backend="polars")
def _row_sums_obj(x: Any, na_rm: bool | None = None) -> Any:
    if isinstance(x, pl.Series):
        return x.sum()
    return sum(x) if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)) else x


# ---- col_means ----------------------------------------------------------


@col_means.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _col_means_tibble(x: Tibble, na_rm: bool | None = None) -> pl.Series:
    return x.select(pl.all().mean()).collect().row(0)


@col_means.register(object, context=Context.EVAL, backend="polars")
def _col_means_obj(x: Any, na_rm: bool | None = None) -> Any:
    if isinstance(x, pl.Series):
        return x.mean()
    if isinstance(x, pl.Expr):
        return x.mean()
    import numpy as np
    return np.mean(x)


# ---- row_means ----------------------------------------------------------


@row_means.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _row_means_tibble(x: Tibble, na_rm: bool | None = None) -> pl.Series:
    return x.select(pl.mean_horizontal(pl.all())).collect().to_series()


@row_means.register(object, context=Context.EVAL, backend="polars")
def _row_means_obj(x: Any, na_rm: bool | None = None) -> Any:
    if isinstance(x, pl.Series):
        return x.mean()
    import numpy as np
    return np.mean(x)


# ---- col_sds ------------------------------------------------------------


@col_sds.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _col_sds_tibble(
    x: Tibble, na_rm: bool | None = None, ddof: int = 1
) -> pl.Series:
    return x.select(pl.all().std(ddof=ddof)).collect().row(0)


@col_sds.register(object, context=Context.EVAL, backend="polars")
def _col_sds_obj(
    x: Any, na_rm: bool | None = None, ddof: int = 1
) -> Any:
    if isinstance(x, pl.Series):
        return x.std(ddof=ddof)
    if isinstance(x, pl.Expr):
        return x.std(ddof=ddof)
    import numpy as np
    return np.std(x, ddof=ddof)


# ---- row_sds ------------------------------------------------------------


@row_sds.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _row_sds_tibble(
    x: Tibble, na_rm: bool | None = None, ddof: int = 1
) -> pl.Series:
    # polars doesn't have std_horizontal, compute manually
    ncols = len(x.columns)
    row_mean = pl.mean_horizontal(pl.all())
    sum_sq = sum(
        (pl.col(c) - row_mean).pow(2)
        for c in x.columns
    )
    return x.select(
        (sum_sq / (ncols - ddof)).sqrt()
    ).collect().to_series()


@row_sds.register(object, context=Context.EVAL, backend="polars")
def _row_sds_obj(
    x: Any, na_rm: bool | None = None, ddof: int = 1
) -> Any:
    if isinstance(x, pl.Series):
        return x.std(ddof=ddof)
    import numpy as np
    return np.std(x, ddof=ddof)


# ---- col_medians --------------------------------------------------------


@col_medians.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _col_medians_tibble(x: Tibble, na_rm: bool | None = None) -> pl.Series:
    return x.select(pl.all().median()).collect().row(0)


@col_medians.register(object, context=Context.EVAL, backend="polars")
def _col_medians_obj(x: Any, na_rm: bool | None = None) -> Any:
    if isinstance(x, pl.Series):
        return x.median()
    if isinstance(x, pl.Expr):
        return x.median()
    import numpy as np
    return np.median(x)


# ---- row_medians --------------------------------------------------------


@row_medians.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _row_medians_tibble(x: Tibble, na_rm: bool | None = None) -> pl.Series:
    # polars doesn't have median_horizontal; collect and compute
    pdf = x.collect()
    vals = []
    for i in range(len(pdf)):
        row = [pdf[c][i] for c in x.columns if pdf[c][i] is not None]
        vals.append(sorted(row)[len(row) // 2] if row else None)
    return pl.Series("row_median", vals)


@row_medians.register(object, context=Context.EVAL, backend="polars")
def _row_medians_obj(x: Any, na_rm: bool | None = None) -> Any:
    if isinstance(x, pl.Series):
        return x.median()
    import numpy as np
    return np.median(x)
