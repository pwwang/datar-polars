"""Group data metadata functions.

See https://github.com/tidyverse/dplyr/blob/master/R/group-data.R
"""

from __future__ import annotations

from typing import Any, List, Sequence

from datar.apis.dplyr import (
    group_data,
    group_keys,
    group_rows,
    group_indices,
    group_vars,
    group_size,
    n_groups,
    group_cols,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble


def _get_gvars(data: Tibble) -> list:
    """Get group variable names."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


def _collect_group_indices(data: Tibble) -> list:
    """Collect row indices for each group."""
    gvars = _get_gvars(data)
    if not gvars:
        n = data.select(pl.len()).collect().item()
        return [list(range(n))]

    import polars as pl

    df = data.select(gvars).collect()
    # Get group boundaries
    groups = {}
    for i, row in enumerate(df.iter_rows()):
        key = tuple(row)
        groups.setdefault(key, []).append(i)
    return list(groups.values())


import polars as pl


# ── group_data ──────────────────────────────────────────────────────────────


@group_data.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_data(_data: Tibble) -> Tibble:
    """Return a tibble with group data (keys and rows)."""
    gvars = _get_gvars(_data)
    rows = _group_rows_impl(_data)

    if not gvars:
        # Ungrouped: single group
        return Tibble({"_rows": [rows[0]]})

    # Grouped: keys + _rows
    keys = _group_keys_impl(_data)
    keys_df = keys.collect()
    keys_df = keys_df.with_columns(
        pl.Series("_rows", rows)
    )
    return Tibble(keys_df.lazy(), _datar=_data._datar.copy() if hasattr(_data, "_datar") else {})


# ── group_keys ──────────────────────────────────────────────────────────────


def _group_keys_impl(_data: Tibble) -> pl.LazyFrame:
    """Internal: compute group keys."""
    gvars = _get_gvars(_data)
    if not gvars:
        return pl.DataFrame({}).lazy()

    drop = _data._datar.get("_drop", True) if hasattr(_data, "_datar") else True
    if not drop:
        schema = _data.collect_schema()
        frames = []
        for gv in gvars:
            dtype = schema[gv]
            if isinstance(dtype, pl.Enum):
                frames.append(
                    pl.LazyFrame({gv: pl.Series(gv, dtype.categories, dtype=dtype)})
                )
            else:
                frames.append(
                    _data.select(gv).unique(maintain_order=True)
                )
        if len(frames) == 1:
            return frames[0]
        result = frames[0]
        for f in frames[1:]:
            result = result.join(f, how="cross")
        return result

    return _data.select(gvars).unique(maintain_order=True)


@group_keys.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_keys(_data: Tibble) -> Tibble:
    """Return a tibble of group keys (one row per group)."""
    result = _group_keys_impl(_data)
    return Tibble(result, _datar=_data._datar.copy() if hasattr(_data, "_datar") else {})


# ── group_rows ──────────────────────────────────────────────────────────────


def _group_rows_impl(_data: Tibble) -> List[List[int]]:
    """Internal: compute group row indices."""
    gvars = _get_gvars(_data)
    if not gvars:
        n = _data.select(pl.len()).collect().item()
        return [list(range(n))]

    df = _data.select(gvars).collect()
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(df.iter_rows()):
        key = tuple(row)
        groups.setdefault(key, []).append(i)

    # Use _group_keys_impl to get canonical key order (respects _drop=False
    # for unused enum/factor levels)
    keys_df = _group_keys_impl(_data).collect()
    result: list[list[int]] = []
    gvar_count = len(gvars)
    for row in keys_df.iter_rows():
        if gvar_count == 1:
            key = (row[0],)
        else:
            key = tuple(row)
        result.append(groups.get(key, []))
    return result


@group_rows.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_rows(_data: Tibble) -> List[List[int]]:
    """Return a list of group row indices."""
    return _group_rows_impl(_data)


# ── group_indices ───────────────────────────────────────────────────────────


@group_indices.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_indices(_data: Tibble) -> List[int]:
    """Return group index for each row."""
    gvars = _get_gvars(_data)
    if not gvars:
        n = _data.select(pl.len()).collect().item()
        return [0] * n

    rows = _group_rows_impl(_data)
    n = sum(len(r) for r in rows)
    result = [0] * n
    for i, group_rows_ in enumerate(rows):
        for j in group_rows_:
            result[j] = i
    return result


# ── group_vars ──────────────────────────────────────────────────────────────


@group_vars.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_vars(_data: Tibble) -> Sequence[str]:
    """Return group variable names."""
    return _get_gvars(_data)


# ── group_size ──────────────────────────────────────────────────────────────


@group_size.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_size(_data: Tibble) -> Sequence[int]:
    """Return the size of each group."""
    rows = _group_rows_impl(_data)
    return [len(r) for r in rows]


# ── n_groups ────────────────────────────────────────────────────────────────


@n_groups.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _n_groups(_data: Tibble) -> int:
    """Return the total number of groups."""
    gvars = _get_gvars(_data)
    if not gvars:
        return 1
    return _group_keys_impl(_data).select(pl.len()).collect().item()


# ── group_cols ──────────────────────────────────────────────────────────────


@group_cols.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _group_cols(_data: Tibble) -> list[int]:
    """Return the column indices of grouping variables."""
    gvars = _get_gvars(_data)
    all_cols = _data.collect_schema().names()
    return [all_cols.index(g) for g in gvars if g in all_cols]
