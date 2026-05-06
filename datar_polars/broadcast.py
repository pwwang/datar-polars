"""Broadcasting (value recycling) rules for the polars backend

Polars handles scalar broadcasting natively via pl.lit and expression APIs.
This module provides add_to_tibble / init_tibble_from for parity with
datar-pandas, used in mutate, summarise, tibble, etc.

Refactored for LazyTibble (pl.LazyFrame subclass) lazy-first architecture:
- Accepts LazyTibble/LazyFrame as first arg
- Uses .with_columns() for adding expressions/columns
- Never manually saves/restores _datar (LazyTibble._from_pyldf handles it)
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import polars as pl

from .polars import DataFrame, Series
from .tibble import LazyTibble
from .common import is_scalar
from .utils import name_of


def init_tibble_from(value: Any, name: str) -> LazyTibble:
    """Initialize a LazyTibble from a single value.

    Args:
        value: The value to wrap in a LazyTibble
        name: Column name for the result

    Returns:
        A LazyTibble with one column
    """
    if is_scalar(value):
        return LazyTibble({name: [value]})
    if isinstance(value, Series):
        return LazyTibble(value.to_frame(name=name or value.name or 0))
    if isinstance(value, pl.Expr):
        return LazyTibble({name or "expr": [None]})
    if isinstance(value, (DataFrame, pl.LazyFrame)):
        from .tibble import as_tibble

        result = as_tibble(value)
        if name:
            result = result.rename(
                {col: f"{name}${col}" for col in result.collect_schema().names()}
            )
        return LazyTibble(result)
    if isinstance(value, dict):
        return LazyTibble(value)
    return LazyTibble({name: value})


def add_to_tibble(
    data: Any,
    name: str | None,
    value: Any,
    allow_dup_names: bool = False,
    broadcast_tbl: bool = False,
) -> Any:
    """Add a column or set of columns to a LazyTibble.

    Args:
        data: Target LazyTibble (may be None)
        name: Column name (None to infer from value)
        value: Value to add — Expr, Series, scalar, array, or DataFrame
        allow_dup_names: If True, attempt to keep duplicate column names
        broadcast_tbl: Reserved for parity with the pandas backend

    Returns:
        LazyTibble with the new column(s) added
    """
    if value is None:
        return data

    if data is None:
        return init_tibble_from(value, name or "")

    # A DataFrame/LazyFrame/Tibble value
    if isinstance(value, (DataFrame, pl.LazyFrame)):
        cols = list(value.collect_schema().names())
        if not name:
            for col in cols:
                data = add_to_tibble(data, col, pl.col(col), allow_dup_names)
        elif isinstance(value, pl.LazyFrame):
            for col in cols:
                data = add_to_tibble(
                    data, f"{name}${col}", pl.col(col), allow_dup_names
                )
        else:
            struct_fields = []
            for col in cols:
                col_series = value.get_column(col)
                if len(col_series) == 1:
                    inner = col_series[0]
                    if hasattr(inner, "_pipda_ref"):
                        struct_fields.append(
                            pl.col(str(inner._pipda_ref)).alias(col)
                        )
                    elif isinstance(inner, pl.Expr):
                        struct_fields.append(inner.alias(col))
                    else:
                        struct_fields.append(pl.lit(inner).alias(col))
                else:
                    struct_fields.append(
                        pl.lit(col_series.to_list()).alias(col)
                    )
            data = data.with_columns(pl.struct(struct_fields).alias(name))
        return data

    effective_name = name or name_of(value) or str(value)

    # polars Expr (lazy expression from ContextEval: f.x → pl.col("x"))
    if isinstance(value, pl.Expr):
        if effective_name in data.collect_schema().names():
            if allow_dup_names:
                tmp = f"{effective_name}_{int(time.time())}"
                data = data.with_columns(value.alias(tmp)).rename(
                    {tmp: effective_name}
                )
            else:
                data = data.with_columns(value.alias(effective_name))
        else:
            data = data.with_columns(value.alias(effective_name))
        return data

    # polars Series — convert to expression via pl.lit for lazy compatibility
    if isinstance(value, Series):
        if effective_name in data.collect_schema().names():
            if allow_dup_names:
                tmp = f"{effective_name}_{int(time.time())}"
                data = data.with_columns(
                    pl.lit(value).alias(tmp)
                ).rename({tmp: effective_name})
            else:
                data = data.with_columns(pl.lit(value).alias(effective_name))
        else:
            data = data.with_columns(pl.lit(value).alias(effective_name))
        return data

    # numpy array
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            data = data.with_columns(pl.lit(value.item()).alias(effective_name))
        else:
            data = data.with_columns(
                pl.lit(pl.Series(effective_name, value)).alias(effective_name)
            )
        return data

    # Plain list / tuple
    if isinstance(value, (list, tuple)):
        data = data.with_columns(
            pl.lit(pl.Series(effective_name, value)).alias(effective_name)
        )
        return data

    # Scalar
    data = data.with_columns(pl.lit(value).alias(effective_name))
    return data
