from __future__ import annotations

from typing import Sequence, List

from datar.apis.dplyr import (
    group_data,
    group_keys,
    group_rows,
    group_indices,
    group_vars,
    group_size,
    n_groups,
)
from polars import DataFrame, LazyFrame, Series
from polars.internals.dataframe.groupby import GroupBy
from polars.internals.lazyframe.groupby import LazyGroupBy

# from ...utils import dict_get
from ...contexts import Context
from ...tibble import Tibble, TibbleRowwise


@group_data.register(
    (DataFrame, LazyFrame),
    context=Context.EVAL_EXPR,
    backend="polars",
)
def _group_data(_data: DataFrame) -> None:
    return None


@group_data.register(GroupBy, context=Context.EVAL_EXPR, backend="polars")
def _group_data_grouped(_data: GroupBy) -> Tibble:
    gpdata = group_keys(_data, __ast_fallback="normal", __backend="polars")
    return gpdata.with_column(
        Series(
            name="_rows",
            values=group_rows(
                _data,
                __ast_fallback="normal",
                __backend="polars",
            ),
        )
    )


@group_keys.register(
    (DataFrame, LazyFrame, TibbleRowwise),
    context=Context.EVAL_EXPR,
    backend="polars",
)
def _group_keys(_data: DataFrame) -> None:
    return None


@group_keys.register(GroupBy, context=Context.EVAL_EXPR, backend="polars")
def _group_keys_grouped(_data: GroupBy) -> Tibble:
    return _data.apply(lambda df: df[_data.by].unique())


@group_keys.register(LazyGroupBy, context=Context.EVAL_EXPR, backend="polars")
def _group_keys_lazygrouped(_data: LazyGroupBy) -> Tibble:
    return _data.apply(lambda df: df[_data.by].unique(), None)


@group_rows.register(DataFrame, context=Context.EVAL_EXPR, backend="polars")
def _group_rows(_data: DataFrame) -> List[List[int]]:
    rows = list(range(_data.shape[0]))
    return [rows]


@group_rows.register(GroupBy, context=Context.EVAL_EXPR, backend="polars")
def _group_rows_grouped(_data: GroupBy) -> List[List[int]]:
    return _data._groups()["groups"].to_list()


@group_indices.register(DataFrame, context=Context.EVAL_EXPR, backend="polars")
def _group_indices(_data: DataFrame) -> List[int]:
    return [0] * _data.shape[0]


@group_indices.register(GroupBy, context=Context.EVAL_EXPR, backend="polars")
def _group_indices_gruoped(_data: GroupBy) -> List[int]:
    ret = {}
    for row in group_data(
        _data,
        __ast_fallback="normal",
        __backend="polars",
    ).itertuples():
        for index in row[-1]:
            ret[index] = row.Index
    return [ret[key] for key in sorted(ret)]


@group_vars.register(DataFrame, context=Context.EVAL_EXPR, backend="polars")
def _group_vars(_data: DataFrame) -> Sequence[str]:
    return []


@group_vars.register(GroupBy, context=Context.EVAL_EXPR, backend="polars")
def _group_vars_gb(_data: GroupBy) -> Sequence[str]:
    return _data.by


@group_size.register(DataFrame, context=Context.EVAL_EXPR, backend="polars")
def _group_size(_data: DataFrame) -> Sequence[int]:
    """Gives the size of each group"""
    return [_data.shape[0]]


@group_size.register(GroupBy, context=Context.EVAL_EXPR, backend="polars")
def _group_size_grouped(_data: GroupBy) -> Sequence[int]:
    return _data._groups()["groups"].apply(lambda s: s.len()).to_list()


@n_groups.register(DataFrame, context=Context.EVAL_EXPR, backend="polars")
def _n_groups(_data: DataFrame) -> int:
    """Gives the total number of groups."""
    return 1


@n_groups.register(GroupBy, context=Context.EVAL_EXPR, backend="polars")
def _n_groups_grouped(_data: GroupBy) -> int:
    return _data._groups().shape[0]


@n_groups.register(TibbleRowwise, context=Context.EVAL_EXPR, backend="polars")
def _n_groups_rowwise(_data: TibbleRowwise) -> int:
    return _data.shape[0]
