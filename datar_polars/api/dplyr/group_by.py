"""Group by verbs and functions
See source https://github.com/tidyverse/dplyr/blob/master/R/group-by.r
"""
from __future__ import annotations
from typing import Any

from datar.apis.dplyr import (
    mutate,
    group_by,
    ungroup,
    rowwise,
)
from polars import DataFrame, LazyFrame
from polars.internals.dataframe.groupby import GroupBy
from polars.internals.lazyframe.groupby import LazyGroupBy

from ...tibble import Tibble, TibbleRowwise, LazyTibbleRowwise
from ...contexts import Context
from ...utils import vars_select, setdiff, union
from .select import _eval_select  # pyright: ignore


@group_by.register(
    (DataFrame, LazyFrame),
    context=Context.PENDING,
    backend="polars",
)
def _group_by(
    _data: DataFrame,
    *args: Any,
    _add: bool = False,  # not working, since _data is not grouped
    _drop: bool = None,
    _sort: bool = False,
    _dropna: bool = False,
    **kwargs: Any,
) -> DataFrame | LazyFrame | GroupBy | LazyGroupBy:
    _data = mutate(
        _data,
        *args,
        __ast_fallback="normal",
        __backend="polars",
        **kwargs,
    )

    new_cols = _data._datar["mutated_cols"]
    if len(new_cols) == 0:
        return _data

    return _data.groupby(new_cols, maintain_order=_sort)


@group_by.register(GroupBy, context=Context.SELECT, backend="polars")
def _group_by_grouped(
    _data: GroupBy,
    *args: Any,
    _add: bool = False,
    _drop: bool = None,
    _sort: bool = False,
    _dropna: bool = False,
    **kwargs: Any,
) -> GroupBy:
    if kwargs:
        raise ValueError("Cannot rename columns in group_by() for GroupBy.")

    all_columns = _data._df.columns()
    selected_idx, new_names = _eval_select(all_columns, *args)
    selected_cols = [all_columns[int(i)] for i in selected_idx]

    ungrouped = ungroup(_data, __ast_fallback="normal", __backend="polars")
    gvars = union(_data.by, selected_cols) if _add else selected_cols
    return group_by(
        ungrouped,
        *gvars,
        _drop=_drop,
        _sort=_sort,
        _dropna=_dropna,
        __ast_fallback="normal",
        __backend="polars",
    )


@rowwise.register(DataFrame, context=Context.SELECT, backend="polars")
def _rowwise(_data: DataFrame, *cols: str | int) -> TibbleRowwise:
    if cols:
        raise ValueError(
            "Setting grouping variables is not supported in rowwise(). "
        )

    return TibbleRowwise(_data)


@rowwise.register(GroupBy, context=Context.SELECT, backend="polars")
def _rowwise_grouped(_data: GroupBy, *cols: str | int) -> TibbleRowwise:
    # grouped dataframe's columns are unique already
    if cols:
        raise ValueError(
            "Setting grouping variables is not supported in rowwise(). "
        )

    ungrouped = ungroup(_data, __ast_fallback="normal", __backend="polars")
    return TibbleRowwise(ungrouped)


@rowwise.register(
    (LazyTibbleRowwise, TibbleRowwise),
    context=Context.SELECT,
    backend="polars",
)
def _rowwise_rowwise(
    _data: LazyTibbleRowwise | TibbleRowwise,
    *cols: str | int,
) -> LazyTibbleRowwise | TibbleRowwise:
    if cols:
        raise ValueError(
            "Setting grouping variables is not supported in rowwise(). "
        )
    return _data


@ungroup.register(DataFrame, context=Context.SELECT, backend="polars")
def _ungroup(x: Any, *cols: str | int) -> Tibble:
    if cols:
        raise ValueError("`*cols` is not empty.")
    return Tibble(x)


@ungroup.register(LazyFrame, context=Context.SELECT, backend="polars")
def _ungroup_lazy(x: Any, *cols: str | int) -> Tibble:
    if cols:
        raise ValueError("`*cols` is not empty.")
    return x


@ungroup.register(LazyGroupBy, context=Context.SELECT, backend="polars")
def _ungroup_lazygroup(
    x: LazyGroupBy,
    *cols: str | int,
) -> LazyFrame | LazyGroupBy:
    if cols:
        raise ValueError("`*cols` is not empty.")
    return x.apply(lambda df: df, None)


@ungroup.register(GroupBy, context=Context.SELECT, backend="polars")
def _ungroup_groupby(x: GroupBy, *cols: str | int) -> DataFrame | GroupBy:
    # If there is a better way?
    ungrouped = x.apply(lambda df: df)
    if not cols:
        return ungrouped

    all_columns = x._df.columns()
    old_groups = x.by
    to_remove = vars_select(all_columns, *cols)
    new_groups = setdiff(old_groups, [all_columns[int(i)] for i in to_remove])

    return ungrouped.groupby(new_groups)
