import polars as pl
from datar.apis.base import (
    is_character,
    is_numeric,
)

from ...factory import func_bootstrap


@func_bootstrap(is_character)
def _is_character_polars(s: pl.Series) -> pl.Series:
    return s.dtype == pl.datatypes.Utf8


@func_bootstrap(is_numeric)
def _is_numeric_polars(s: pl.Series) -> pl.Series:
    return s.dtype in (
        pl.datatypes.Int8,
        pl.datatypes.Int16,
        pl.datatypes.Int32,
        pl.datatypes.Int64,
        pl.datatypes.UInt8,
        pl.datatypes.UInt16,
        pl.datatypes.UInt32,
        pl.datatypes.UInt64,
        pl.datatypes.Float32,
        pl.datatypes.Float64,
    )
