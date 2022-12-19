"""Arrange rows by column values

See source https://github.com/tidyverse/dplyr/blob/master/R/arrange.R
"""
from __future__ import annotations
from typing import Any

from polars import DataFrame, Series
from datar.apis.dplyr import arrange

from ...contexts import Context
from ...utils import union
from ...tibble import TibbleGrouped, TibbleRowwise
from .mutate import mutate


@arrange.register(DataFrame, context=Context.PENDING, backend="polars")
def _arrange(
    _data: DataFrame,
    *args: Any,
    _by_group: bool = False,
    **kwargs: Any,
) -> DataFrame:
    if not args and not kwargs:
        return _data

    data = _data.with_column(
        Series(range(_data.shape[0])).alias("__arrange_id__")
    )
    gvars = (
        []
        if _data.datar.grouper is None
        else _data.datar.grouper.group_vars
    )
    sorting_df = mutate(
        _data,
        *args,
        __ast_fallback="normal",
        __backend="polars",
        **kwargs,
    )
    sorting_df = sorting_df.with_column(
        Series(range(_data.shape[0])).alias("__arrange_id__")
    )
    if _by_group:
        sorting_cols = union(gvars, sorting_df.datar.meta["mutated_cols"])
    else:
        sorting_cols = sorting_df.datar.meta["mutated_cols"]

    sorting_df = sorting_df.sort(list(sorting_cols), nulls_last=True).select(
        "__arrange_id__"
    )
    out = sorting_df.join(data, how="left", on="__arrange_id__")
    out = out.drop("__arrange_id__")
    if isinstance(_data, TibbleGrouped):
        return out.datar.group_by(*gvars, sort=_data.datar.grouper.sort)
    if isinstance(_data, TibbleRowwise):
        return out.datar.rowwise(*gvars)
    return out
