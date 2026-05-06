"""Lazy and collect verbs for the polars backend.

Provides lazy() to convert a DataFrame to a LazyTibble and
collect() to materialize a LazyTibble back to an eager Tibble.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pipda import register_verb
from pipda.utils import TypeHolder

from ..contexts import Context
from ..tibble import (
    Tibble,
    LazyTibble,
    as_tibble,
    to_lazy,
    to_eager,
)


# ---- lazy() ------------------------------------------------------------


@register_verb(TypeHolder, context=Context.PENDING, ast_fallback="normal")
def lazy(_data):
    """Mark data for lazy evaluation.

    Converts a DataFrame to a LazyTibble.
    For LazyFrame inputs, wraps as LazyTibble.
    """
    raise NotImplementedError(
        "lazy() is only available with the polars backend."
    )


@lazy.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _lazy_lf(_data: pl.LazyFrame) -> LazyTibble:
    """Wrap a LazyFrame as LazyTibble."""
    return to_lazy(_data)


@lazy.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _lazy_df(_data: pl.DataFrame) -> LazyTibble:
    """Convert a DataFrame to LazyTibble."""
    return to_lazy(_data)


@lazy.register(object, context=Context.PENDING, backend="polars")
def _lazy_any(_data: Any) -> LazyTibble:
    """Wrap any input as a LazyTibble."""
    return to_lazy(_data)


# ---- collect() ---------------------------------------------------------


@register_verb(TypeHolder, context=Context.PENDING, ast_fallback="normal")
def collect(_data):
    """Materialize a lazy query into an eager Tibble.

    For LazyFrame inputs, calls .collect() and wraps as Tibble.
    For DataFrame inputs, wraps as Tibble.
    """
    raise NotImplementedError(
        "collect() is only available with the polars backend."
    )


@collect.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _collect_lf(_data: pl.LazyFrame) -> Tibble:
    """Materialize a LazyFrame to a Tibble."""
    return to_eager(_data)


@collect.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _collect_df(_data: pl.DataFrame) -> Tibble:
    """Ensure a DataFrame is a Tibble."""
    if isinstance(_data, Tibble):
        return _data
    return as_tibble(_data)


@collect.register(object, context=Context.PENDING, backend="polars")
def _collect_any(_data: Any) -> Any:
    """Pass through already-materialized data."""
    return _data
