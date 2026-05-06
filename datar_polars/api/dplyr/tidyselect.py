"""Tidyselect helpers for column selection.

These are used INSIDE select/mutate/etc expressions to reference columns.
"""

from __future__ import annotations

import builtins
import re
from typing import Any, Callable, List, Optional, Sequence, cast

import polars as pl

from datar.dplyr import (
    ungroup,
    group_vars,
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

from ...polars import DataFrame
from ...contexts import Context
from ...tibble import Tibble, LazyTibble
from ...common import is_scalar, setdiff
from ...utils import vars_select


# ── where ───────────────────────────────────────────────────────────────────


@where.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _where(_data: Tibble, fn: Callable) -> List[str]:
    """Select columns where fn returns True."""
    columns = _everything(_data)
    _data_ungrouped = ungroup(
        _data,
        __ast_fallback="normal",
        __backend="polars",
    )
    # Collect if lazy — LazyFrame does not support df[col] subscripting
    if isinstance(_data_ungrouped, pl.LazyFrame):
        _data_ungrouped = _data_ungrouped.collect()
    mask = []
    for col in columns:
        col_data = _data_ungrouped.get_column(col)
        if getattr(fn, "_pipda_functype", None) == "verb" and fn.dependent:
            dat = fn(col_data)._pipda_eval(_data_ungrouped)
            mask.append(dat)
        elif getattr(fn, "_pipda_functype", None) == "pipeable":
            mask.append(fn(col_data, __ast_fallback="normal"))
        else:
            mask.append(fn(col_data))

    mask = [
        flag if is_scalar(flag) else all(flag) for flag in mask
    ]
    return [c for c, m in zip(columns, mask) if m]


# ── everything ──────────────────────────────────────────────────────────────


@everything.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _everything(_data: Tibble) -> List[str]:
    """Select all columns except grouping variables."""
    return list(
        setdiff(
            _data.collect_schema().names(),
            group_vars(
                _data,
                __ast_fallback="normal",
                __backend="polars",
            ),
        )
    )


# ── last_col ────────────────────────────────────────────────────────────────


@last_col.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _last_col(
    _data: Tibble,
    offset: int = 0,
    vars: Optional[Sequence[str]] = None,
) -> str:
    """Select the last column (with optional offset)."""
    vars = list(vars) if vars is not None else _data.collect_schema().names()
    return vars[-(offset + 1)]


# ── starts_with ─────────────────────────────────────────────────────────────


@starts_with.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _starts_with(
    _data: Tibble,
    match: str | Sequence[str],
    ignore_case: bool = True,
    vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select columns that start with a prefix."""
    return _filter_columns(
        list(vars) if vars is not None else _data.collect_schema().names(),
        match,
        ignore_case,
        lambda mat, cname: cname.startswith(mat),
    )


# ── ends_with ───────────────────────────────────────────────────────────────


@ends_with.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _ends_with(
    _data: Tibble,
    match: str | Sequence[str],
    ignore_case: bool = True,
    vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select columns that end with a suffix."""
    return _filter_columns(
        list(vars) if vars is not None else _data.collect_schema().names(),
        match,
        ignore_case,
        lambda mat, cname: cname.endswith(mat),
    )


# ── contains ────────────────────────────────────────────────────────────────


@contains.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _contains(
    _data: Tibble,
    match: str,
    ignore_case: bool = True,
    vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select columns that contain a substring."""
    return _filter_columns(
        list(vars) if vars is not None else _data.collect_schema().names(),
        match,
        ignore_case,
        lambda mat, cname: mat in cname,
    )


# ── matches ─────────────────────────────────────────────────────────────────


@matches.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _matches(
    _data: Tibble,
    match: str,
    ignore_case: bool = True,
    vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """Select columns that match a regex."""
    # Compile the regex with flags
    flags = re.IGNORECASE if ignore_case else 0
    return _filter_columns(
        list(vars) if vars is not None else _data.collect_schema().names(),
        match,
        ignore_case,
        lambda mat, cname: re.search(mat, cname, flags) is not None,
    )


# ── all_of ──────────────────────────────────────────────────────────────────


@all_of.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _all_of(
    _data: Tibble,
    x: Sequence[int | str],
) -> List[str]:
    """Strict column selection — all must exist."""
    all_columns = _data.collect_schema().names()
    x_selected = [all_columns[i] for i in vars_select(all_columns, *x)]
    # Raise error if any don't exist (vars_select handles this)
    return cast(List[str], list(x_selected))


# ── any_of ──────────────────────────────────────────────────────────────────


@any_of.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _any_of(
    _data: Tibble,
    x: Sequence[int | str],
    vars: Optional[Sequence[str]] = None,
) -> List[str]:
    """Loose column selection — ignore columns that don't exist."""
    if vars is not None:
        var_pool = list(vars)
    else:
        var_pool = _data.collect_schema().names()
    idx = vars_select(var_pool, *x, raise_nonexists=False)
    return [var_pool[i] for i in idx]


# ── num_range ───────────────────────────────────────────────────────────────


@num_range.register(str, backend="polars")
def _num_range(
    prefix: str,
    range_val: int,
    width: Optional[int] = None,
) -> List[str]:
    """Generate a sequence of column names: prefix1, prefix2, ..."""
    zfill = (
        (lambda elem: elem)
        if not width
        else (lambda elem: str(elem).zfill(width))
    )
    return [f"{prefix}{zfill(elem)}" for elem in builtins.range(range_val)]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _filter_columns(
    all_columns: Sequence[str],
    match: Sequence[str] | str,
    ignore_case: bool,
    func: Callable[[str, str], bool],
) -> List[str]:
    """Filter columns by a matching function.

    Args:
        all_columns: The column names to filter from.
        match: String or sequence of strings to match.
        ignore_case: Whether to ignore case.
        func: Function(mat, cname) -> bool.

    Returns:
        Matching column names in order.
    """
    if is_scalar(match):
        match = [match]  # type: ignore

    ret: List[str] = []
    for mat in match:
        for column in all_columns:
            if column in ret:
                continue
            mat_cmp = mat.lower() if ignore_case else mat
            col_cmp = column.lower() if ignore_case else column
            if func(mat_cmp, col_cmp):
                ret.append(column)
    return ret
