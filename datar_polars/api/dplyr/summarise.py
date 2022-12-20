"""Summarise each group to fewer rows"""
from __future__ import annotations
from itertools import chain
from typing import Any

from polars import DataFrame, Series
from polars.internals.dataframe.groupby import GroupBy
from pipda import evaluate_expr
from datar import get_option
from datar.core.utils import arg_match, logger
from datar.apis.dplyr import summarise

from ...contexts import Context
from ...tibble import add_to_tibble
from ...utils import name_of, setdiff, to_expr
from ...tibble import Tibble, TibbleRowwise

from .group_by import ungroup
from .group_data import group_vars, group_keys


@summarise.register(DataFrame, context=Context.PENDING, backend="polars")
def _summarise_frame(
    _data: DataFrame | GroupBy,
    *args: Any,
    _groups: str = None,
    **kwargs: Any,
) -> Tibble:
    out = _summarise_build_frame(_data, *args, **kwargs)

    if _groups == "rowwise":
        out = TibbleRowwise(out)

    return out


def _summarise_build_frame(
    _data: DataFrame,
    *args: Any,
    **kwargs: Any,
) -> Tibble:
    """Build summarise result"""
    if not isinstance(_data, TibbleRowwise):
        _data = Tibble(_data)

    _data.datar.meta["used_refs"] = set()
    outframe = Tibble(_summarise_holder=0)
    outframe._datar["summarise_source"] = _data
    for key, val in chain(enumerate(args), kwargs.items()):
        try:
            val = evaluate_expr(val, outframe, Context.EVAL_DATA)
        except KeyError:
            val = evaluate_expr(val, _data, Context.EVAL_DATA)

        if val is None:
            continue

        if isinstance(key, int):
            if isinstance(val, (DataFrame, Series)) and len(val) == 0:
                continue
            key = name_of(val)

        newframe = add_to_tibble(outframe, key, val, broadcast_tbl=True)
        outframe = newframe

    tmp_cols = [
        mcol
        for mcol in outframe.columns
        if mcol.startswith("_")
        and mcol not in _data._datar["used_refs"]
    ]
    outframe = outframe[setdiff(outframe.columns, tmp_cols)]
    return outframe


@summarise.register(GroupBy, context=Context.PENDING, backend="polars")
def _summarise_groupby(
    _data: GroupBy,
    *args: Any,
    _groups: str = None,
    **kwargs: Any,
) -> Tibble:
    _groups = arg_match(
        _groups, "_groups", ["drop", "drop_last", "keep", "rowwise", None]
    )

    gvars = group_vars(_data, __ast_fallback="normal", __backend="polars")
    out = _summarise_build_groupby(_data, *args, **kwargs)
    if _groups is None:
        _groups = "drop_last"

    if _groups == "drop_last":
        if get_option("dplyr_summarise_inform"):
            logger.info(
                "`summarise()` has grouped output by "
                "%s (override with `_groups` argument)",
                gvars[:-1],
            )
        out = out.groupby(list(gvars[:-1]), maintain_order=_data.maintain_order)

    elif _groups == "keep":
        if get_option("dplyr_summarise_inform"):
            logger.info(
                "`summarise()` has grouped output by "
                "%s (override with `_groups` argument)",
                gvars,
            )
        out = out.group_by(gvars, mainon_order=_data.maintain_order)

    elif _groups == "rowwise":
        out = TibbleRowwise(out)

    # else: # drop
    return out


def _summarise_build_groupby(
    _data: GroupBy,
    *args: Any,
    **kwargs: Any,
) -> Tibble:
    """Build summarise result"""
    outframe = group_keys(
        _data,
        __ast_fallback="normal",
        __backend="polars",
    )

    _data._datar = {"used_refs": set()}
    outframe._datar = {"summarise_source": _data}
    for key, val in chain(enumerate(args), kwargs.items()):
        val = evaluate_expr(val, _data, Context.EVAL)

        if val is None:
            continue

        if isinstance(key, int):
            if isinstance(val, (DataFrame, Series)) and len(val) == 0:
                continue
            key = name_of(val)

        out = _data.agg(to_expr(val, name=key))
        outframe = outframe.with_column(out[key])

    gvars = group_vars(_data, __ast_fallback="normal", __backend="polars")
    tmp_cols = [
        mcol
        for mcol in outframe.columns
        if mcol.startswith("_")
        and mcol not in _data._datar["used_refs"]
        and mcol not in gvars
    ]
    outframe = ungroup(outframe, __ast_fallback="normal", __backend="polars")
    outframe = outframe[setdiff(outframe.columns, tmp_cols)]
    return outframe
