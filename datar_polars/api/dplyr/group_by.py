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
from polars import DataFrame

from ...tibble import Tibble, TibbleGrouped, TibbleRowwise
from ...contexts import Context
from ...utils import vars_select, setdiff, union
from .select import _eval_select  # pyright: ignore


@group_by.register(DataFrame, context=Context.PENDING, backend="polars")
def _group_by(
    _data: DataFrame,
    *args: Any,
    _add: bool = False,  # not working, since _data is not grouped
    _drop: bool = None,
    # when try to retrieve df from gf, ordered changed
    _sort: bool = True,
    **kwargs: Any,
) -> DataFrame | TibbleGrouped:
    _data = mutate(
        _data,
        *args,
        __ast_fallback="normal",
        __backend="polars",
        **kwargs,
    )

    new_cols = _data.datar.meta["mutated_cols"]
    if len(new_cols) == 0:
        return _data

    return _data.datar.group_by(*new_cols, sort=_sort)


@group_by.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.SELECT,
    backend="polars",
)
def _group_by_grouped(
    _data: TibbleGrouped | TibbleRowwise,
    *args: Any,
    _add: bool = False,
    _drop: bool = None,
    _sort: bool = True,
    **kwargs: Any,
) -> TibbleGrouped:
    all_columns = _data._df.columns()
    selected_idx, new_names = _eval_select(all_columns, *args, **kwargs)
    selected_old_cols = [all_columns[int(i)] for i in selected_idx]
    data = _data.rename(new_names)
    selected_cols = [
        new_names.get(col, col)
        if new_names
        else col
        for col in selected_old_cols
    ]

    if _add:
        gvars = union(
            [
                new_names.get(var, var)
                if new_names
                else var
                for var in _data.datar.grouper.group_vars
            ],
            selected_cols,
        )
    else:
        gvars = selected_cols

    return data.datar.group_by(*gvars, sort=_sort)


@rowwise.register(DataFrame, context=Context.SELECT, backend="polars")
def _rowwise(
    _data: DataFrame,
    *cols: str | int,
) -> TibbleRowwise:
    idxes = vars_select(_data.columns, *cols)
    return _data.datar.rowwise(*(_data.columns[int(i)] for i in idxes))


@rowwise.register(TibbleGrouped, context=Context.SELECT, backend="polars")
def _rowwise_grouped(_data: TibbleGrouped, *cols: str | int) -> TibbleRowwise:
    # grouped dataframe's columns are unique already
    if cols:
        raise ValueError(
            "Can't re-group when creating rowwise data. "
            "Either first `ungroup()` or call `rowwise()` without arguments."
        )

    return _data.datar.rowwise(*_data.group_vars)


@ungroup.register(DataFrame, context=Context.SELECT, backend="polars")
def _ungroup(x: DataFrame | TibbleRowwise, *cols: str | int) -> Tibble:
    if cols:
        raise ValueError("`*cols` is not empty.")
    return x.datar.ungroup() if isinstance(x, TibbleRowwise) else x


@ungroup.register(TibbleGrouped, context=Context.SELECT, backend="polars")
def _ungroup_groupby(
    x: TibbleGrouped,
    *cols: str | int,
) -> DataFrame | TibbleGrouped:
    # If there is a better way?
    ungrouped = x.datar.ungroup()
    if not cols:
        return ungrouped

    all_columns = ungrouped.columns()
    old_groups = x.datar.grouper.group_vars
    to_remove = vars_select(all_columns, *cols)
    new_groups = setdiff(old_groups, [all_columns[int(i)] for i in to_remove])

    return ungrouped.datar.group_by(new_groups, sort=x.datar.grouper.sort)
