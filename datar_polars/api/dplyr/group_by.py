"""Group by verbs and functions

See source https://github.com/tidyverse/dplyr/blob/master/R/group-by.r
"""

from __future__ import annotations

from typing import Any, Optional

from datar.dplyr import (
    group_by,
    ungroup,
    rowwise,
    group_by_drop_default,
    group_vars,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...utils import vars_select
from ...common import setdiff


@group_by.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _group_by(
    _data: Tibble,
    *args: Any,
    _add: bool = False,
    _drop: Optional[bool] = None,
    **kwargs: Any,
) -> Tibble:
    existing_groups = list(_data._datar.get("groups") or []) if hasattr(_data, "_datar") else []
    result = _data.clone()

    if args:
        new_groups = list(vars_select(list(result.collect_schema().names()), *args))
        new_group_names = [list(result.collect_schema().names())[i] for i in new_groups]
    elif kwargs:
        new_group_names = list(kwargs.keys())
    else:
        new_group_names = []

    if _add and existing_groups:
        group_names = existing_groups + [
            g for g in new_group_names if g not in existing_groups
        ]
    else:
        group_names = new_group_names

    result._datar["groups"] = group_names if group_names else None
    result._datar["_drop"] = _drop if _drop is not None else True
    return reconstruct_tibble(result, _data)


@ungroup.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _ungroup(
    x: Tibble,
    *cols: str | int,
) -> Tibble:
    old_groups = list(x._datar.get("groups") or []) if hasattr(x, "_datar") else []
    result = x.clone()

    if not cols:
        result._datar["groups"] = None
        return reconstruct_tibble(result, x)

    to_remove_idx = vars_select(list(x.collect_schema().names()), *cols, raise_nonexists=True)
    to_remove = [list(x.collect_schema().names())[i] for i in to_remove_idx]
    new_groups = setdiff(old_groups, to_remove)
    result._datar["groups"] = new_groups if new_groups else None
    return reconstruct_tibble(result, x)


@rowwise.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _rowwise(
    _data: Tibble,
    *cols: str | int,
) -> Tibble:
    result = _data.clone()
    result._datar["rowwise"] = True

    if cols:
        idx = vars_select(list(result.collect_schema().names()), *cols)
        gvars = [list(result.collect_schema().names())[i] for i in idx]
        result._datar["groups"] = gvars

    return reconstruct_tibble(result, _data)


@group_vars.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _group_vars(_data: Tibble) -> list:
    if hasattr(_data, "_datar") and _data._datar.get("groups") is not None:
        return list(_data._datar["groups"])
    return []


@group_by_drop_default.register((Tibble, LazyTibble), backend="polars")
def _group_by_drop_default(_tbl: Tibble) -> bool:
    if hasattr(_tbl, "_datar") and "_drop" in _tbl._datar:
        return _tbl._datar["_drop"]
    return True
