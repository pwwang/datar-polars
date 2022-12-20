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
from polars import DataFrame

# from ...utils import dict_get
from ...contexts import Context
from ...extended import DFGrouper
from ...tibble import Tibble, TibbleGrouped, TibbleRowwise


@group_data.register(DataFrame, context=Context.EVAL, backend="polars")
def _group_data(_data: DataFrame) -> None:
    return DFGrouper(_data).group_data


@group_data.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.EVAL,
    backend="polars",
)
def _group_data_grouped(_data: Tibble) -> Tibble:
    return _data.datar.grouper.group_data


@group_keys.register(DataFrame, context=Context.EVAL, backend="polars")
def _group_keys(_data: DataFrame) -> None:
    return DFGrouper(_data).group_keys


@group_keys.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.EVAL,
    backend="polars",
)
def _group_keys_grouped(_data: Tibble) -> Tibble:
    return _data.datar.grouper.group_keys


@group_rows.register(DataFrame, context=Context.EVAL, backend="polars")
def _group_rows(_data: DataFrame) -> List[List[int]]:
    return DFGrouper(_data).group_rows


@group_rows.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.EVAL,
    backend="polars",
)
def _group_rows_grouped(_data: Tibble) -> List[List[int]]:
    return _data.datar.grouper.group_rows


@group_indices.register(DataFrame, context=Context.EVAL, backend="polars")
def _group_indices(_data: DataFrame) -> List[int]:
    return DFGrouper(_data).group_indices


@group_indices.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.EVAL,
    backend="polars",
)
def _group_indices_gruoped(_data: Tibble) -> List[int]:
    return _data.datar.grouper.group_indices


@group_vars.register(DataFrame, context=Context.EVAL, backend="polars")
def _group_vars(_data: DataFrame) -> Sequence[str]:
    return []


@group_vars.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.EVAL,
    backend="polars",
)
def _group_vars_gb(_data: Tibble) -> Sequence[str]:
    return _data.datar.grouper.group_vars


@group_size.register(DataFrame, context=Context.EVAL, backend="polars")
def _group_size(_data: DataFrame) -> Sequence[int]:
    """Gives the size of each group"""
    return DFGrouper(_data).group_size


@group_size.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.EVAL,
    backend="polars",
)
def _group_size_grouped(_data: Tibble) -> Sequence[int]:
    return _data.datar.grouper.group_size


@n_groups.register(DataFrame, context=Context.EVAL, backend="polars")
def _n_groups(_data: DataFrame) -> int:
    """Gives the total number of groups."""
    return 1


@n_groups.register(
    (TibbleGrouped, TibbleRowwise),
    context=Context.EVAL,
    backend="polars",
)
def _n_groups_grouped(_data: Tibble) -> int:
    return _data.datar.grouper.n_groups
