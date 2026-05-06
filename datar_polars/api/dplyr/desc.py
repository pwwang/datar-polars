"""Provides desc() for the polars backend

desc() negates values so that sorting ascending produces a descending-order
result inside arrange(). Uses to_physical() so categorical columns (factors)
get their integer codes negated.
"""

import polars as pl

from datar.apis.dplyr import desc

from ...polars import Series


@desc.register(pl.Expr, backend="polars")
def _desc_expr(x: pl.Expr) -> pl.Expr:
    """Negate values to invert sort order for arrange()."""
    return -x.to_physical().cast(pl.Float64)


@desc.register(Series, backend="polars")
def _desc_series(x):
    return -x.to_physical().cast(pl.Float64)


@desc.register(object, backend="polars")
def _desc_obj(x):
    """Negate plain Python values (lists, ranges, scalars)."""
    return -pl.Series(x).to_physical().cast(pl.Float64)
