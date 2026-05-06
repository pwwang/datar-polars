"""Subset columns using their names and types

See source https://github.com/tidyverse/dplyr/blob/master/R/select.R
"""

from __future__ import annotations

from types import GeneratorType
from typing import Any, Mapping, Sequence, Tuple

import polars as pl

from datar.core.utils import logger
from datar.dplyr import select

from ...contexts import Context
from ...tibble import (
    Tibble,
    LazyTibble,
    reconstruct_tibble,
    to_lazy,
    to_eager,
)
from ...utils import vars_select
from ...common import setdiff, union, intersect
from ...collections import Collection, Intersect


def _get_gvars(data) -> list:
    """Return grouping variable names from a Tibble/LazyTibble."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


@select.register(pl.LazyFrame, context=Context.SELECT, backend="polars")
def _select_lazy(
    _data: pl.LazyFrame,
    *args: Any,
    **kwargs: Any,
) -> LazyTibble:
    all_columns = _data.collect_schema().names()
    gvars = _get_gvars(_data)

    selected_idx, new_names = _eval_select(
        all_columns,
        *args,
        _group_vars=gvars,
        **kwargs,
    )
    selected_cols = [all_columns[i] for i in selected_idx]

    out = _data.select(selected_cols)

    if new_names:
        out = out.rename(new_names)
        # Update group_vars if any group columns were renamed
        if gvars:
            updated_gvars = [new_names.get(g, g) for g in gvars]
            if hasattr(out, "_datar"):
                out._datar["groups"] = updated_gvars

    return reconstruct_tibble(out, _data)


@select.register(pl.DataFrame, context=Context.SELECT, backend="polars")
def _select_eager(
    _data: pl.DataFrame,
    *args: Any,
    **kwargs: Any,
) -> Tibble:
    return to_eager(
        _select_lazy(to_lazy(_data), *args, **kwargs)
    )


def _eval_select(
    _all_columns: list,
    *args: Any,
    _group_vars: Sequence[str],
    _missing_gvars_inform: bool = True,
    **kwargs: Any,
) -> Tuple[list, Mapping[str, str] | None]:
    """Evaluate selections to column indices.

    Returns:
        A tuple of *(selected column indices, dict of old-to-new rename pairs)*.
    """
    normalized_args = tuple(_materialize_select_arg(arg) for arg in args)
    normalized_kwargs = {
        key: _materialize_select_arg(val) for key, val in kwargs.items()
    }

    selected_idx = list(
        vars_select(_all_columns, *normalized_args, *normalized_kwargs.values())
    )

    if _missing_gvars_inform:
        selected_names = [_all_columns[i] for i in selected_idx]
        missing = setdiff(_group_vars, selected_names)
        if missing:
            logger.info("Adding missing grouping variables: %s", ", ".join(missing))

    # Always include group variables
    gvar_indices = [
        _all_columns.index(g) for g in _group_vars if g in _all_columns
    ]
    selected_idx = union(gvar_indices, selected_idx)

    if not kwargs:
        return selected_idx, None

    # kwargs are rename pairs:  select(df, new=old)  →  kwargs = {"new": "old"}
    rename_idx = list(vars_select(_all_columns, *normalized_kwargs.values()))
    new_names = dict(
        zip([_all_columns[i] for i in rename_idx], kwargs.keys())
    )
    return selected_idx, new_names


def _materialize_select_arg(arg: Any) -> Any:
    """Convert one-shot generators inside tidyselect arguments to tuples.

    Some notebook expressions like c[f.name:f.mass] arrive as a Collection
    wrapping a generator produced by pipda's lazy evaluation. Materializing
    those generators preserves the intended selection before vars_select()
    expands the collection against the column pool.
    """
    if isinstance(arg, Intersect):
        return Intersect(
            _materialize_select_arg(arg.elems[0]),
            _materialize_select_arg(arg.elems[1]),
            pool=arg.pool,
        )
    if isinstance(arg, Collection):
        return type(arg)(
            *(_materialize_select_arg(elem) for elem in list(arg)),
            pool=arg.pool,
        )
    if isinstance(arg, GeneratorType):
        return tuple(_materialize_select_arg(elem) for elem in arg)
    return arg
