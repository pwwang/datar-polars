"""Relocate columns"""
from __future__ import annotations
from typing import Any

from datar.apis.dplyr import relocate

from polars import DataFrame, col

from ...contexts import Context
from ...utils import setdiff, union
from .select import _eval_select  # pyright: ignore


@relocate.register(DataFrame, context=Context.SELECT, backend="polars")
def _relocate(
    _data: DataFrame,
    *args: Any,
    _before: int | str = None,
    _after: int | str = None,
    **kwargs: Any,
) -> DataFrame:

    all_columns = _data.columns
    to_move, new_names = _eval_select(all_columns, *args, **kwargs)

    to_move = list(to_move)
    if _before is not None and _after is not None:
        raise ValueError("Must supply only one of `_before` and `_after`.")

    # length = len(all_columns)
    if _before is not None:
        where = _eval_select(all_columns, _before)[0].min()
        if where not in to_move:
            to_move.append(where)

    elif _after is not None:
        where = _eval_select(all_columns, _after)[0].max()
        if where not in to_move:
            to_move.insert(0, where)
    else:
        where = 0
        if where not in to_move:
            to_move.append(where)

    lhs = setdiff(range(where), to_move)
    rhs = setdiff(range(where + 1, len(all_columns)), to_move)
    pos = union(lhs, union(to_move, rhs))

    cols = []
    for i in pos:
        selected_col = all_columns[int(i)]
        if new_names and selected_col in new_names:
            cols.append(col(selected_col).alias(new_names[selected_col]))
        else:
            cols.append(selected_col)

    return _data.select(cols)
