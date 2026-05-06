"""Common utility functions for the polars backend"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import polars as pl


def unique(x: Any) -> np.ndarray:
    """Return unique elements as a numpy array (order-preserving)."""
    if isinstance(x, (pl.Series,)):
        return x.unique(maintain_order=True).to_numpy()
    seen = set()
    result = []
    for item in list(x):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return np.array(result)


def is_scalar(x: Any) -> bool:
    """Check if x is a scalar value"""
    if isinstance(x, (str, bytes)):
        return True
    if isinstance(x, (list, tuple, pl.Series, pl.Expr)):
        return False
    if isinstance(x, np.ndarray):
        return x.ndim == 0
    if hasattr(x, "__len__") and not isinstance(x, dict):
        try:
            return len(x) == 0
        except TypeError:
            # __len__ exists via __getattr__ but doesn't actually work
            # (e.g. pipda Expression objects)
            return False
    return True


def is_null(x: Any) -> Any:
    """Check if values are null"""
    import polars as pl

    if isinstance(x, pl.Series):
        return x.is_null()
    if isinstance(x, pl.DataFrame):
        return x.select(pl.all().is_null())
    return x is None


def is_integer(x: Any) -> bool:
    """Check if x is an integer or integer array"""
    import numpy as np

    if isinstance(x, (int, np.integer)):
        return True
    if hasattr(x, "dtype"):
        return np.issubdtype(x.dtype, np.integer)
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        try:
            arr = np.array(x)
            return np.issubdtype(arr.dtype, np.integer)
        except (TypeError, ValueError):
            return False
    return False


def setdiff(a: Iterable, b: Iterable) -> list:
    """Return items in a not in b"""
    b_set = set(b)
    return [item for item in a if item not in b_set]


def union(a: Iterable, b: Iterable) -> list:
    """Return items in a or b (order preserving, duplicates removed)"""
    seen = set()
    result = []
    for item in list(a) + list(b):
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def intersect(a: Iterable, b: Iterable) -> list:
    """Return items in both a and b (order preserving from a)"""
    b_set = set(b)
    seen = set()
    result = []
    for item in a:
        if item in b_set and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_iterable(x: Any) -> bool:
    """Check if x is a non-string iterable (list, tuple, Series, Collection)."""
    if isinstance(x, (str, bytes, pl.Expr)):
        return False
    return hasattr(x, "__iter__")


def to_series(x: Any, length: int = None) -> pl.Series:
    """Convert input to a pl.Series, broadcasting scalars to the given length."""
    import math

    if isinstance(x, pl.Series):
        return x
    if isinstance(x, pl.Expr):
        raise TypeError("Cannot convert Expr to Series directly.")
    if is_iterable(x):
        vals = list(x)
        if length is not None and len(vals) < length:
            vals = vals * length
        vals = vals[:length] if length else vals
        vals = [None if isinstance(v, float) and math.isnan(v) else v for v in vals]
        return pl.Series(vals, strict=False)
    if length is not None:
        return pl.Series([x] * length)
    return pl.Series([x])
