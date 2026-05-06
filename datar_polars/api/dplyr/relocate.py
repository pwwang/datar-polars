"""Relocate columns.

See https://github.com/tidyverse/dplyr/blob/master/R/relocate.R
"""

from __future__ import annotations

from typing import Any, Optional

from datar.apis.dplyr import relocate, group_vars

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...utils import vars_select
from ...common import setdiff, union, intersect


def _get_gvars(data: Tibble) -> list:
    """Get group variable names."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


@relocate.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _relocate(
    _data: Tibble,
    *args: Any,
    _before: Optional[int | str] = None,
    _after: Optional[int | str] = None,
    **kwargs: Any,
) -> Tibble:
    """Relocate columns in a Tibble.

    Args:
        _data: The Tibble.
        *args: Columns to move (tidy-selection).
        _before: Destination column to place before.
        _after: Destination column to place after.
        **kwargs: Rename pairs (new=old).

    Returns:
        Tibble with columns relocated.
    """
    gvars = _get_gvars(_data)
    all_columns = _data.collect_schema().names()

    if _before is not None and _after is not None:
        raise ValueError("Must supply only one of `_before` and `_after`.")

    # Resolve columns to move
    if args:
        to_move_idx = vars_select(all_columns, *args)
        to_move = [all_columns[i] for i in to_move_idx]
    else:
        to_move = []

    # Also handle kwargs (rename pairs)
    if kwargs:
        kw_idx = vars_select(all_columns, *kwargs.values())
        kw_cols = [all_columns[i] for i in kw_idx]
        to_move.extend(kw_cols)
        # Build rename mapping
        rename_map = {
            all_columns[i]: new_name
            for i, new_name in zip(kw_idx, kwargs.keys())
        }
    else:
        rename_map = {}

    if not to_move:
        return _data

    # Build remaining (excluding moved columns) first
    remaining = [c for c in all_columns if c not in to_move]

    # Determine where to insert in remaining
    if _before is not None:
        before_idx = vars_select(all_columns, _before)
        target = all_columns[before_idx[0]]
        where = remaining.index(target) if target in remaining else min(before_idx)
    elif _after is not None:
        after_idx = vars_select(all_columns, _after)
        target = all_columns[after_idx[0]]
        where = remaining.index(target) + 1 if target in remaining else max(after_idx) + 1
    else:
        where = 0

    # Insert to_move at 'where' (counting only non-moved columns)
    lhs = remaining[:where]
    rhs = remaining[where:]
    new_order = lhs + to_move + rhs

    # Rename if needed
    if rename_map:
        result = _data.select(new_order).rename(rename_map)
    else:
        result = _data.select(new_order)

    return reconstruct_tibble(result, _data)
