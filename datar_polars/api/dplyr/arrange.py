"""Arrange rows by column values

See source https://github.com/tidyverse/dplyr/blob/master/R/arrange.R
"""

from __future__ import annotations

from typing import Any

import polars as pl
from pipda import evaluate_expr

from datar.dplyr import arrange

from ...contexts import Context
from ...tibble import (
    Tibble,
    LazyTibble,
    reconstruct_tibble,
    to_lazy,
    to_eager,
)


@arrange.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _arrange_lazy(
    _data: pl.LazyFrame,
    *args: Any,
    _by_group: bool = False,
    **kwargs: Any,
) -> LazyTibble:
    if not args and not kwargs and not _by_group:
        return _data

    gvars = (
        _data._datar.get("groups", [])
        if hasattr(_data, "_datar")
        else []
    )

    data = _data
    sort_cols: list[str] = []

    def _add_sort_expr(data, expr, idx):
        """Add a single sort expression, return (data, col_name)."""
        tmp_name = f"__sort_{idx}"
        if isinstance(expr, pl.Expr):
            data = data.with_columns(expr.alias(tmp_name))
        elif isinstance(expr, str):
            return data, expr  # existing column name
        else:
            data = data.with_columns(
                pl.Series(tmp_name, expr)
                if isinstance(expr, (list, tuple))
                else pl.lit(expr).alias(tmp_name)
            )
        return data, tmp_name

    sort_idx = 0
    for val in args:
        if val is None:
            continue
        evaluated = evaluate_expr(val, data, Context.EVAL)
        # Handle across() returning a list/tuple of Exprs — expand each
        if isinstance(evaluated, (list, tuple)) and all(
            isinstance(e, pl.Expr) for e in evaluated
        ):
            for expr in evaluated:
                data, col = _add_sort_expr(data, expr, sort_idx)
                sort_cols.append(col)
                sort_idx += 1
        elif isinstance(evaluated, (list, tuple)):
            data, col = _add_sort_expr(data, evaluated, sort_idx)
            sort_cols.append(col)
            sort_idx += 1
        else:
            data, col = _add_sort_expr(data, evaluated, sort_idx)
            sort_cols.append(col)
            sort_idx += 1

    for key, val in kwargs.items():
        evaluated = evaluate_expr(val, data, Context.EVAL)
        if isinstance(evaluated, (list, tuple)) and all(
            isinstance(e, pl.Expr) for e in evaluated
        ):
            for expr in evaluated:
                data, col = _add_sort_expr(data, expr, sort_idx)
                sort_cols.append(col)
                sort_idx += 1
        elif isinstance(evaluated, (list, tuple)):
            data, col = _add_sort_expr(data, evaluated, sort_idx)
            sort_cols.append(col)
            sort_idx += 1
        else:
            data, col = _add_sort_expr(data, evaluated, sort_idx)
            sort_cols.append(col)
            sort_idx += 1

    if _by_group:
        gcols = [g for g in gvars if g in data.collect_schema().names()]
        sort_cols = gcols + sort_cols

    if not sort_cols:
        return _data

    result = data.sort(sort_cols)

    tmp_cols = [c for c in sort_cols if c.startswith("__sort_")]
    if tmp_cols:
        result = result.drop(tmp_cols)

    return reconstruct_tibble(result, _data)


@arrange.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _arrange_eager(
    _data: pl.DataFrame,
    *args: Any,
    _by_group: bool = False,
    **kwargs: Any,
) -> Tibble:
    return to_eager(
        _arrange_lazy(
            to_lazy(_data), *args, _by_group=_by_group, **kwargs
        )
    )
