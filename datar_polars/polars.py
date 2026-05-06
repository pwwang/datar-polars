"""Polars backend types and utilities

Provides Tibble (eager pl.DataFrame subclass), LazyTibble (lazy pl.LazyFrame
subclass), and related utilities for the polars backend.
"""

import polars as pl

from .tibble import Tibble, LazyTibble  # noqa: F401

DataFrame = pl.DataFrame
LazyFrame = pl.LazyFrame
Series = pl.Series

# Re-export commonly used functions
concat = pl.concat
read_csv = pl.read_csv

__all__ = [
    "Tibble",
    "LazyTibble",
    "DataFrame",
    "LazyFrame",
    "Series",
    "concat",
    "read_csv",
]
