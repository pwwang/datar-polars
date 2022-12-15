"""Subset columns using their names and types

See source https://github.com/tidyverse/dplyr/blob/master/R/select.R
"""
from __future__ import annotations
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from datar.apis.dplyr import select
from datar.core.utils import logger

from polars import DataFrame, LazyFrame, col
from polars.internals.dataframe.groupby import GroupBy

from ...contexts import Context
from ...utils import vars_select, intersect, setdiff, union
from ...collections import Collection, Inverted


@select.register(
    GroupBy,
    context=Context.SELECT,
    backend="polars",
)
def _select_gb(
    _data: GroupBy,
    *args: str | Iterable | Inverted,
    **kwargs: Mapping[str, str],
) -> GroupBy:
    old_gvars = _data.by
    ungrouped = _data.apply(lambda x: x)
    all_columns = ungrouped.columns
    selected_idx, new_names = _eval_select(
        all_columns,
        *args,
        **kwargs,
        _group_vars=old_gvars,
    )
    cols = []
    new_gvars = []
    for i in selected_idx:
        selected_col = all_columns[int(i)]
        if new_names and selected_col in new_names:
            cols.append(col(selected_col).alias(new_names[selected_col]))
            new_gvars.append(new_names[selected_col])
        else:
            cols.append(selected_col)
            new_gvars.append(selected_col)
    data = ungrouped.select(cols)
    gvars = intersect(union(old_gvars, new_gvars), data.columns)
    return data.groupby(gvars)


@select.register(
    (DataFrame, LazyFrame),
    context=Context.SELECT,
    backend="polars",
)
def _select(
    _data: DataFrame,
    *args: str | Iterable | Inverted,
    **kwargs: Mapping[str, str],
) -> DataFrame:
    all_columns = _data.columns
    selected_idx, new_names = _eval_select(all_columns, *args, **kwargs)
    cols = []
    for i in selected_idx:
        selected_col = all_columns[int(i)]
        if new_names and selected_col in new_names:
            cols.append(col(selected_col).alias(new_names[selected_col]))
        else:
            cols.append(selected_col)
    return _data.select(cols)


def _eval_select(
    _all_columns: List[str],
    *args: Any,
    _group_vars: Sequence[str] = (),
    _missing_gvars_inform: bool = True,
    **kwargs: Any,
) -> Tuple[Sequence[int], Mapping[str, str]]:
    """Evaluate selections to get locations

    Returns:
        A tuple of (selected column indices, dict of old-to-new renamings)
    """
    selected_idx = vars_select(
        _all_columns,
        *args,
        *kwargs.values(),
    )

    if _missing_gvars_inform:
        missing = setdiff(
            _group_vars,
            [_all_columns[int(i)] for i in selected_idx],
        )
        if len(missing) > 0:
            logger.info("Adding missing grouping variables: %s", missing)

    selected_idx = union(
        Collection(*_group_vars, pool=_all_columns),
        selected_idx,
    )

    if not kwargs:
        return selected_idx, None

    rename_idx = vars_select(_all_columns, *kwargs.values())
    new_names = dict(zip([_all_columns[i] for i in rename_idx], kwargs))
    return selected_idx, new_names
