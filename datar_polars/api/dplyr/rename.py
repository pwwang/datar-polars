"""Rename columns

https://github.com/tidyverse/dplyr/blob/master/R/rename.R
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from datar.apis.dplyr import group_vars, rename, rename_with

from ...polars import DataFrame
from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...utils import vars_select
from .select import _eval_select


@rename.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _rename(_data: Tibble, **kwargs: str) -> Tibble:
    """Rename columns of a Tibble.

    Args:
        _data: The Tibble to rename columns of.
        **kwargs: new_name=old_name pairs.

    Returns:
        A new Tibble with renamed columns.
    """
    gvars = group_vars(
        _data,
        __ast_fallback="normal",
        __backend="polars",
    )
    all_columns = _data.collect_schema().names()
    selected, new_names = _eval_select(
        all_columns,
        _group_vars=gvars,
        _missing_gvars_inform=False,
        **kwargs,
    )
    rename_map = {} if new_names is None else new_names

    # Build rename mapping: old -> new
    polars_rename = {}
    for i, col in enumerate(all_columns):
        if i in selected and col in rename_map:
            polars_rename[col] = rename_map[col]

    out = _data.rename(polars_rename)

    # Update group vars if any were renamed
    if polars_rename and hasattr(_data, "_datar") and _data._datar.get("groups"):
        new_gvars = [rename_map.get(g, g) for g in gvars]
        out = reconstruct_tibble(out, _data)
        out._datar["groups"] = new_gvars

    return reconstruct_tibble(out, _data)


@rename_with.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _rename_with(
    _data: Tibble,
    _fn: Callable,
    *args: Any,
    **kwargs: Any,
) -> Tibble:
    """Rename columns using a function.

    Args:
        _data: The Tibble to rename columns of.
        _fn: Function that takes a column name and returns a new name.
        *args: Columns to apply function to.
        **kwargs: Additional keyword arguments passed to _fn.

    Returns:
        A new Tibble with renamed columns.
    """
    if not args:
        cols = _data.collect_schema().names()
    else:
        cols = args[0]
        args = args[1:]

    if isinstance(cols, Sequence) and not isinstance(cols, str):
        selected = vars_select(_data.collect_schema().names(), *cols)
    else:
        selected = vars_select(_data.collect_schema().names(), cols)

    all_columns = _data.collect_schema().names()
    cols_to_rename = [all_columns[i] for i in selected]
    new_columns = {
        _fn(col, *args, **kwargs): col for col in cols_to_rename
    }

    return rename(
        _data,
        **new_columns,
        __ast_fallback="normal",
        __backend="polars",
    )
