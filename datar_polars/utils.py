"""Utility functions for the polars backend"""

from __future__ import annotations

import math
import textwrap
from typing import Any, Iterable
from functools import singledispatch

import polars as pl
from pipda import ReferenceAttr, ReferenceItem

from .polars import DataFrame, Series
from .collections import Collection

DEFAULT_COLUMN_PREFIX = "_VAR_"


def replace_na_with_none(val: list) -> list:
    """Replace NA (float nan) with None when mixed with non-numeric types.

    NA is float('nan'). In a pure numeric (float-like) column Polars handles
    NaN natively — no replacement needed. In a mixed str+NA column, Polars
    would coerce NaN to the literal string "NaN" (not null), which breaks
    fill() and other operations. We detect that case and replace NA → None.
    """
    if not isinstance(val, (list, tuple)) or not val:
        return val

    # Single pass: detect whether both NA and non-numeric values exist.
    # Early-break once both are found — no need to scan the whole list.
    has_na = False
    has_non_numeric = False
    has_bool = False
    for v in val:
        if isinstance(v, float) and math.isnan(v):
            has_na = True
        elif v is not None and not isinstance(v, (int, float, complex, bool)):
            has_non_numeric = True
        elif isinstance(v, bool):
            has_bool = True
        if has_na and (has_non_numeric or has_bool):
            break

    if has_na and not (has_non_numeric or has_bool):
        return val
    if not has_na:
        return val

    return [
        None if (isinstance(v, float) and math.isnan(v)) else v for v in val
    ]

# Specify a "no default" value so that None can be used as a default value
NO_DEFAULT = object()
# Just for internal and testing uses
NA_character_ = "<NA>"


@singledispatch
def name_of(value: Any) -> str:
    """Get the name of a value"""
    out = str(value)
    out = textwrap.shorten(out, 16, break_on_hyphens=False, placeholder="...")
    return out


name_of.register(Series, lambda x: x.name if x.name else None)
name_of.register(pl.Expr, lambda x: x.meta.output_name() if hasattr(x, "meta") and x.meta.output_name() else None)


def is_scalar(x: Any) -> bool:
    """Check if x is a scalar value (not a list/array/series)"""
    import numpy as np

    if isinstance(x, (str, bytes)):
        return True
    if isinstance(x, (Series, pl.Expr)):
        return False
    if isinstance(x, np.ndarray):
        return x.ndim == 0
    if hasattr(x, "__len__") and not isinstance(x, dict):
        return len(x) == 0
    return True


def is_null(x: Any) -> Any:
    """Check if value is null or None"""
    return x is None or x is pl.Null


def is_integer(x: Any) -> bool:
    """Check if x is an integer or integers"""
    if isinstance(x, (int, pl.Int8, pl.Int16, pl.Int32, pl.Int64)):
        return True
    if hasattr(x, "dtype"):
        return x.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64)
    return False


def is_iterable(x: Any) -> bool:
    """Check if x is an iterable (list/array/series) but not a string/bytes"""
    return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))


def is_expr_list(x: list) -> bool:
    """Check if a list contains only pl.Expr objects (e.g. from c_across)."""
    return len(x) > 0 and all(isinstance(e, pl.Expr) for e in x)


def to_series(x: Any) -> pl.Series:
    """Convert x to a polars Series if it is not already one"""
    if isinstance(x, pl.Series):
        return x
    if isinstance(x, pl.Expr):
        raise TypeError("Cannot convert pl.Expr to Series")
    return pl.Series("", [x] if is_scalar(x) else x, strict=False)


def cat_levels(x: pl.Series) -> list:
    """Get the levels of a factor-like Series (pl.Enum or pl.Categorical).

    A pl.Enum carries a fixed category list; a pl.Categorical does not, so its
    distinct values in order of appearance stand in for the levels.
    Replaces ``Series.cat.get_categories``, removed in polars 2.0.
    """
    if isinstance(x.dtype, pl.Enum):
        return x.dtype.categories.to_list()
    return x.cast(pl.Utf8).drop_nulls().unique(maintain_order=True).to_list()


def ntile_bins(x: pl.Expr | pl.Series, n: int) -> pl.Expr | pl.Series:
    """Split x into n buckets of roughly equal size, numbered from 1.

    Buckets are cut on ordinal rank (dplyr's ntile), not on the values, so
    tied values may land in different buckets and nulls/NaNs stay null.
    Computed directly rather than via ``qcut``: that is deprecated in polars
    2.0 and its replacement ``bin_quantiles`` does not exist in 1.36, and the
    two number their bins differently anyway.
    """
    non_na = x.is_not_null() & x.is_not_nan()
    return (((x.rank("ordinal") - 1) * n / non_na.sum()).floor() + 1).cast(
        pl.Int64
    )


def vars_select(
    all_columns: Iterable[str],
    *columns: int | str,
    raise_nonexists: bool = True,
):
    """Select columns from a pool of column names

    Args:
        all_columns: The column pool to select from
        *columns: arguments to select from the pool
        raise_nonexists: Whether to raise when a column doesn't exist

    Returns:
        The selected indexes for columns

    Raises:
        KeyError: When a column does not exist and raise_nonexists is True
    """
    cols: Any = [
        column._pipda_ref if isinstance(column, (ReferenceAttr, ReferenceItem)) else (
            column.name if isinstance(column, Series) else column
        )
        for column in columns
    ]
    for col in cols:
        if not isinstance(col, str):
            continue
        # Check for duplicate names
        if isinstance(all_columns, list):
            indices = [i for i, c in enumerate(all_columns) if c == col]
            if len(indices) > 1:
                raise ValueError(
                    f'Names must be unique. Name "{col}" found at '
                    f"locations {indices}."
                )

    selected = Collection(*cols, pool=list(all_columns))
    if raise_nonexists and selected.unmatched and selected.unmatched != {None}:
        raise KeyError(f"Columns `{selected.unmatched}` do not exist.")

    from .common import unique

    return unique(selected).astype(int)
