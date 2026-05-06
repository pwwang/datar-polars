"""Iterate over groups and with_groups.

https://github.com/tidyverse/dplyr/blob/master/R/group_split.R
https://github.com/tidyverse/dplyr/blob/master/R/group_map.R
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import polars as pl

from datar.apis.dplyr import (
    ungroup,
    group_by,
    with_groups,
    group_map,
    group_modify,
    group_split,
    group_trim,
    group_walk,
    group_vars,
    group_rows,
    group_keys,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble, as_tibble


# ── _helpers ─────────────────────────────────────────────────────────────────


def _get_gvars(data: Tibble) -> list:
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


def _collect_row_indices(data: Tibble) -> list:
    """Return list of row-index-lists, one per group."""
    gvars = _get_gvars(data)
    if not gvars:
        n = data.select(pl.len()).collect().item()
        return [list(range(n))]
    df = data.select(gvars).collect()
    groups: dict[tuple, list] = {}
    for i, row in enumerate(df.iter_rows()):
        key = tuple(row)
        groups.setdefault(key, []).append(i)
    # Preserve first-occurrence order
    seen: set[tuple] = set()
    result: list[list[int]] = []
    for row in df.iter_rows():
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            result.append(groups[key])
    return result


def _eager(data: Tibble) -> Tibble:
    """Collect if lazy, otherwise return as-is."""
    if isinstance(data, LazyTibble):
        return data.collect()
    return data


# ── with_groups ─────────────────────────────────────────────────────────────


@with_groups.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _with_groups(
    _data: Tibble,
    _groups: Any,
    _func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Temporarily regroup data, apply a function, then return result.

    Args:
        _data: The Tibble.
        _groups: Grouping columns (or None to ungroup).
        _func: Function to apply to regrouped data.
        *args: Additional args for _func.
        **kwargs: Additional kwargs for _func.

    Returns:
        Result of _func applied to regrouped data.
    """
    if _groups is None:
        grouped = ungroup(
            _data,
            __ast_fallback="normal",
            __backend="polars",
        )
    else:
        grouped = group_by(
            _data,
            _groups,
            __ast_fallback="normal",
            __backend="polars",
        )

    return _func(grouped, *args, **kwargs)


# ── group_split ─────────────────────────────────────────────────────────────


@group_split.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_split(
    _data: Tibble,
    *args: Any,
    _keep: bool = True,
    **kwargs: Any,
):
    """Split a grouped frame into a list of data frames, one per group."""
    if args:
        _data = group_by(
            _data, *args, __ast_fallback="normal", __backend="polars", **kwargs
        )

    row_sets = _collect_row_indices(_data)
    eager = _eager(_data)
    gvars = _get_gvars(_data)
    all_cols = eager.collect_schema().names()

    for idx in row_sets:
        chunk = eager[idx]
        if not _keep and gvars:
            keep_cols = [c for c in all_cols if c not in gvars]
            chunk = chunk.select(keep_cols)
        yield Tibble(chunk, _datar={"backend": "polars"})


# ── group_map ───────────────────────────────────────────────────────────────


@group_map.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_map(
    _data: Tibble,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
):
    """Apply a function to each group, yielding results."""
    row_sets = _collect_row_indices(_data)
    eager = _eager(_data)
    gvars = _get_gvars(_data)
    all_cols = eager.collect_schema().names()
    keys_df = eager.select(gvars).unique(maintain_order=True) if gvars else None

    try:
        sig = inspect.signature(_f)
        n_params = len(sig.parameters)
    except (ValueError, TypeError):
        n_params = 1

    for i, idx in enumerate(row_sets):
        chunk = eager[idx]
        if not _keep and gvars:
            keep_cols = [c for c in all_cols if c not in gvars]
            chunk = chunk.select(keep_cols)
        if n_params > 1 and keys_df is not None and len(keys_df) > 0:
            key_row = keys_df.row(i)
            yield _f(chunk, key_row, *args, **kwargs)
        else:
            yield _f(chunk, *args, **kwargs)


# ── group_modify ────────────────────────────────────────────────────────────


@group_modify.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_modify(
    _data: Tibble,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
) -> Tibble:
    """Apply a function to each group, combining results into a data frame."""
    gvars = _get_gvars(_data)

    if not gvars:
        result = _f(_data, *args, **kwargs)
        if not isinstance(result, (Tibble, pl.DataFrame)):
            result = as_tibble(result)
        return reconstruct_tibble(result, _data)

    results: list[pl.DataFrame] = []

    for chunk_or_result in _group_map(
        _data, _f, *args, _keep=_keep, **kwargs
    ):
        if not isinstance(chunk_or_result, (pl.DataFrame, Tibble)):
            chunk_or_result = as_tibble(chunk_or_result)
        results.append(chunk_or_result)

    if not results:
        return Tibble({})

    combined = pl.concat(results, how="vertical")
    return reconstruct_tibble(combined, _data)


# ── group_trim ──────────────────────────────────────────────────────────────


@group_trim.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_trim(
    _data: Tibble,
    _drop: bool | None = None,
) -> Tibble:
    """Drop unused factor levels from grouping columns.

    In polars, there are no R-style factor levels that track unused
    categories, so this is essentially a no-op that returns the data
    with groups regenerated from actual present values.
    """
    gvars = _get_gvars(_data)
    if not gvars:
        return _data
    # Re-group with drop=True forces recalculation from actual data
    return group_by(
        _data,
        *gvars,
        __ast_fallback="normal",
        __backend="polars",
    )


# ── group_walk ──────────────────────────────────────────────────────────────


@group_walk.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_walk(
    _data: Tibble,
    _f: Callable,
    *args: Any,
    _keep: bool = False,
    **kwargs: Any,
) -> None:
    """Apply a function to each group (for side effects), returning None."""
    for _ in _group_map(_data, _f, *args, _keep=_keep, **kwargs):
        pass
