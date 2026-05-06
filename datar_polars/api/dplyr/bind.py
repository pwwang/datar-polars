"""Bind multiple data frames by row and column

See https://github.com/tidyverse/dplyr/blob/master/R/bind.r
"""

from __future__ import annotations
from typing import Any, Callable, Optional, Union

import polars as pl
from datar.core.names import repair_names
from datar.apis.dplyr import bind_rows, bind_cols

from ...polars import DataFrame
from ...contexts import Context
from ...tibble import Tibble, reconstruct_tibble, as_tibble
from ...common import is_scalar


def _construct_tibble(data: Any) -> Tibble:
    """Convert arbitrary data to a Tibble for binding."""
    if isinstance(data, Tibble):
        return data
    if isinstance(data, dict):
        data = data.copy()
        for key, val in data.items():
            data[key] = [val] if is_scalar(val) else val
        return Tibble(data)
    if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
        return as_tibble(data)
    if isinstance(data, list):
        return Tibble(data)
    return as_tibble(data)


# ── bind_rows ───────────────────────────────────────────────────────────────


@bind_rows.register((Tibble, DataFrame, list, dict, type(None)), backend="polars")
def _bind_rows(
    *datas: Tibble | DataFrame | list | dict | None,
    _id: Optional[str] = None,
    _copy: bool = True,
    **kwargs: Any,
) -> Tibble:
    if _id is not None and not isinstance(_id, str):
        raise ValueError("`_id` must be a scalar string.")

    if not datas:
        _data = None
    else:
        _data, datas = datas[0], datas[1:]

    key_data: dict = {}
    if isinstance(_data, list):
        _data = [d for d in _data if d is not None]
        for i, dat in enumerate(_data):
            key_data[i] = _construct_tibble(dat)
    elif _data is not None:
        key_data[0] = _construct_tibble(_data)

    for i, dat in enumerate(datas):
        if isinstance(dat, list):
            for df in dat:
                key_data[len(key_data)] = _construct_tibble(df)
        elif dat is not None:
            key_data[len(key_data)] = _construct_tibble(dat)

    for key, val in kwargs.items():
        if val is not None:
            key_data[key] = _construct_tibble(val)

    if not key_data:
        return Tibble()

    # Collect all Tibbles to DataFrames for concatenation
    collected = [v.collect() for v in key_data.values()]

    if _id is not None:
        # Add _id column with key names
        dfs_with_id = []
        for key, df in zip(key_data.keys(), collected):
            df_with_id = df.with_columns(pl.lit(key).alias(_id))
            dfs_with_id.append(df_with_id)
        result = pl.concat(dfs_with_id, how="diagonal_relaxed")
    else:
        result = pl.concat(collected, how="diagonal_relaxed")

    # Wrap back to lazy Tibble
    result_tbl = Tibble(result.lazy())
    return reconstruct_tibble(result_tbl)


# ── bind_cols ───────────────────────────────────────────────────────────────


@bind_cols.register((Tibble, DataFrame, dict, type(None)), backend="polars")
def _bind_cols(
    *datas: Tibble | DataFrame | dict | None,
    _name_repair: Union[str, Callable] = "unique",
    _copy: bool = True,
) -> Tibble:
    ds = [
        _construct_tibble(d) for d in datas if d is not None
    ]

    if not ds:
        return Tibble()

    # Collect all to DataFrames
    collected = [d.collect() for d in ds]

    # Repair column names BEFORE horizontal concat — polars rejects duplicates
    all_cols = []
    for cdf in collected:
        all_cols.extend(cdf.collect_schema().names())
    new_names = repair_names(all_cols, repair=_name_repair)
    if new_names != all_cols:
        # Rename each dataframe's columns to the repaired names
        offset = 0
        renamed = []
        for cdf in collected:
            ncols = len(cdf.collect_schema().names())
            cdf_names = new_names[offset : offset + ncols]
            renamed.append(
                cdf.rename(dict(zip(cdf.collect_schema().names(), cdf_names)))
            )
            offset += ncols
        collected = renamed

    # Horizontal concatenation
    result = pl.concat(collected, how="horizontal")

    result_tbl = Tibble(result.lazy())
    return reconstruct_tibble(result_tbl)
