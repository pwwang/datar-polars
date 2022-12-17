"""Create, modify, and delete columns

See source https://github.com/tidyverse/dplyr/blob/master/R/mutate.R
"""
from __future__ import annotations

from contextlib import suppress

from pipda import evaluate_expr, ReferenceAttr, ReferenceItem

from datar.core.utils import arg_match
from datar.dplyr import mutate, transmute, relocate

from polars import DataFrame, NotFoundError

from ...contexts import Context
from ...collections import Collection
from ...utils import (
    name_of,
    setdiff,
    union,
    intersect,
)
from ...tibble import Tibble, add_to_tibble


@mutate.register(DataFrame, context=Context.PENDING, backend="polars")
def _mutate(
    _data: DataFrame,
    *args,
    _keep: str = "all",
    _before: int | str = None,
    _after: int | str = None,
    **kwargs,
):
    keep = arg_match(_keep, "_keep", ["all", "unused", "used", "none"])
    # so we can attach metadata
    data = Tibble(_data)
    data.datar.meta["used_refs"] = set()
    all_columns = data.columns

    mutated_cols = []
    for val in args:
        if (
            isinstance(val, (ReferenceItem, ReferenceAttr))
            and val._pipda_level == 1
            and val._pipda_ref in data.columns
        ):
            mutated_cols.append(val._pipda_ref)
            continue

        bkup_name = name_of(val)
        val = evaluate_expr(val, data, Context.EVAL_DATA)
        if val is None:
            continue

        if isinstance(val, DataFrame):
            mutated_cols.extend(val.columns)
            data = add_to_tibble(data, None, val, broadcast_tbl=False)
        else:
            key = name_of(val) or bkup_name
            mutated_cols.append(key)
            data = add_to_tibble(data, key, val, broadcast_tbl=False)

    for key, val in kwargs.items():
        val = evaluate_expr(val, data, Context.EVAL_DATA)
        if val is None:
            with suppress(NotFoundError):
                data.drop_in_place(key)
        else:
            data = add_to_tibble(data, key, val, broadcast_tbl=False)
            if isinstance(val, DataFrame):
                mutated_cols.extend({f"{key}${col}" for col in val.columns})
            else:
                mutated_cols.append(key)

    # names start with "_" are temporary names if they are used
    used_refs = data.datar.meta["used_refs"]
    tmp_cols = [
        mcol
        for mcol in mutated_cols
        if mcol.startswith("_")
        and mcol in used_refs
        and mcol not in _data.columns
    ]
    # columns can be removed later
    # df >> mutate(Series(1, name="z"), z=None)
    mutated_cols = intersect(mutated_cols, data.columns)
    mutated_cols = setdiff(mutated_cols, tmp_cols)
    # new cols always at last
    # data.columns.difference() does not keep order

    data = data.select(setdiff(data.columns, tmp_cols))

    if _before is not None or _after is not None:
        new_cols = setdiff(mutated_cols, _data.columns)
        data = relocate(
            data,
            *new_cols,
            _before=_before,
            _after=_after,
            __ast_fallback="normal",
            __backend="polars",
        )

    if keep == "all":
        keep = data.columns
    elif keep == "unused":
        unused = setdiff(all_columns, list(used_refs))
        keep = intersect(data.columns, Collection(unused, mutated_cols))
    elif keep == "used":
        keep = intersect(data.columns, Collection(used_refs, mutated_cols))
    else:  # keep == 'none':
        keep = union(mutated_cols, intersect(mutated_cols, data.columns))

    data = data.select(keep)

    # used for group_by
    data.datar.meta["mutated_cols"] = mutated_cols
    return data


@transmute.register(DataFrame, context=Context.PENDING, backend="polars")
def _transmute(
    _data,
    *args,
    _before=None,
    _after=None,
    **kwargs,
):
    return mutate(
        _data,
        *args,
        _keep="none",
        _before=_before,
        _after=_after,
        __ast_fallback="normal",
        __backend="polars",
        **kwargs,
    )
