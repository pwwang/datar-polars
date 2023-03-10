from __future__ import annotations
import builtins
import re
from typing import Callable, List, Sequence

import numpy as np
from polars import DataFrame, Expr, col as pl_col
from datar.apis.dplyr import (
    where,
    everything,
    last_col,
    starts_with,
    ends_with,
    contains,
    matches,
    all_of,
    any_of,
    num_range,
)
from datar_numpy.utils import make_array

from ...contexts import Context
from ...utils import vars_select, is_scalar, is_logical, setdiff, intersect

from .group_data import group_vars


@where.register(DataFrame, context=Context.EVAL, backend="polars")
def _where(_data: DataFrame, fn: Callable) -> List[str]:
    columns = _data >> everything()
    mask = []
    for col in columns:
        if getattr(fn, "_pipda_functype", None) == "verb" and fn.dependent:
            dat = fn(_data[col])._pipda_eval(_data)
            mask.append(dat)
        elif getattr(fn, "_pipda_functype", None) == "dispatchable":
            if (
                "polars" not in fn.registry
                or Expr not in fn.registry["polars"].registry
            ):
                mask.append(fn(_data[col]))
            else:
                # Havn't found using predicate on an Expr as a Series
                # to filter the column names
                dat = _data.select(
                    fn(pl_col(col)).alias("selected")
                )["selected"]
                mask.append(dat)
        else:
            mask.append(fn(_data[col]))

    mask = [
        flag
        if is_scalar(flag) and is_logical(flag)
        else all(flag)
        for flag in mask
    ]
    return np.array(columns)[mask].tolist()


@everything.register(DataFrame, context=Context.EVAL, backend="polars")
def _everything(_data: DataFrame) -> List[str]:
    return list(
        setdiff(
            _data.columns,
            group_vars(_data, __ast_fallback="normal", __backend="polars"),
        )
    )


@last_col.register(DataFrame, context=Context.SELECT, backend="polars")
def _last_col(
    _data: DataFrame,
    offset: int = 0,
    vars: Sequence[str] = None,
) -> str:
    vars = vars or _data.columns
    return vars[-(offset + 1)]


@starts_with.register(DataFrame, context=Context.SELECT, backend="polars")
def _starts_with(
    _data: DataFrame,
    match: str | Sequence[str],
    ignore_case: bool = True,
    vars: Sequence[str] = None,
) -> List[str]:
    return _filter_columns(
        vars or _data.columns,
        match,
        ignore_case,
        lambda mat, cname: cname.startswith(mat),
    )


@ends_with.register(DataFrame, context=Context.SELECT, backend="polars")
def _ends_with(
    _data: DataFrame,
    match: str | Sequence[str],
    ignore_case: bool = True,
    vars: Sequence[str] = None,
) -> List[str]:
    return _filter_columns(
        vars or _data.columns,
        match,
        ignore_case,
        lambda mat, cname: cname.endswith(mat),
    )


@contains.register(DataFrame, context=Context.SELECT, backend="polars")
def _contains(
    _data: DataFrame,
    match: str,
    ignore_case: bool = True,
    vars: Sequence[str] = None,
) -> List[str]:
    return _filter_columns(
        vars or _data.columns,
        match,
        ignore_case,
        lambda mat, cname: mat in cname,
    )


@matches.register(DataFrame, context=Context.SELECT, backend="polars")
def _matches(
    _data: DataFrame,
    match: str,
    ignore_case: bool = True,
    vars: Sequence[str] = None,
) -> List[str]:
    return _filter_columns(
        vars or _data.columns,
        match,
        ignore_case,
        re.search,
    )


@all_of.register(DataFrame, context=Context.EVAL, backend="polars")
def _all_of(
    _data: DataFrame,
    x: Sequence[int | str],
) -> List[str]:
    all_columns = _data.columns
    x = all_columns[vars_select(all_columns, x)]
    # where do errors raise?

    # nonexists = setdiff(x, all_columns)
    # if nonexists:
    #     raise ColumnNotExistingError(
    #         "Can't subset columns that don't exist. "
    #         f"Columns {nonexists} not exist."
    #     )

    return x.tolist()


@any_of.register(DataFrame, context=Context.SELECT, backend="polars")
def _any_of(
    _data: DataFrame,
    x: Sequence[int | str],
    vars: Sequence[str] = None,
) -> List[str]:
    if vars is not None:  # pragma: no cover
        vars = make_array(vars)
    else:
        vars = _data.columns
    x = vars_select(vars, x, raise_nonexists=False)
    return list(vars[x])


@num_range.register(str, backend="polars")
def _num_range(
    prefix: str,
    range: Sequence[int],
    width: int = None,
) -> List[str]:
    zfill = lambda elem: (  # noqa: E731
        elem if not width else str(elem).zfill(width)
    )
    return [f"{prefix}{zfill(elem)}" for elem in builtins.range(range)]


def _filter_columns(
    all_columns: Sequence[str],
    match: Sequence[str] | str,
    ignore_case: bool,
    func: Callable[[str, str], bool],
) -> List[str]:
    """Filter the columns with given critera

    Args:
        all_columns: The column pool to filter
        match: Strings. If len>1, the union of the matches is taken.
        ignore_case: If True, the default, ignores case when matching names.
        func: A function to define how to filter.

    Returns:
        A list of matched vars
    """
    if is_scalar(match):
        match = [match]  # type: ignore

    ret = []
    for mat in match:  # order kept this way
        for column in all_columns:
            if column in ret:
                continue
            if func(
                mat.lower() if ignore_case else mat,
                column.lower() if ignore_case else column,
            ):
                ret.append(column)

    return ret
