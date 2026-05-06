"""Mutating joins

See source https://github.com/tidyverse/dplyr/blob/master/R/join.R
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl

from datar.dplyr import (
    inner_join,
    left_join,
    right_join,
    full_join,
    semi_join,
    anti_join,
    nest_join,
    cross_join,
)
from datar.core.utils import NotImplementedByCurrentBackendError

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...common import is_scalar, intersect, setdiff, union


def _resolve_by(
    x: Tibble,
    y: Tibble,
    by: Any,
) -> tuple:
    """Resolve join `by` to (left_on, right_on, on) for polars .join()."""
    if by is None:
        on = intersect(list(x.collect_schema().names()), list(y.collect_schema().names()))
        if not on:
            return None, None, None
        return None, None, on
    if isinstance(by, dict):
        return list(by.keys()), list(by.values()), None
    if isinstance(by, pl.Expr):
        return None, None, [by.meta.output_name()]
    if is_scalar(by):
        return None, None, [by]
    return None, None, list(by)


def _join_prep(
    x: Tibble,
    y: Tibble,
    by: Any,
    suffix: tuple = ("_x", "_y"),
    keep: bool = False,
) -> tuple:
    """Prepare Tibbles and join parameters."""
    left_on, right_on, on = _resolve_by(x, y, by)

    suffix_str = suffix[1]

    if keep and on:
        # Rename join columns on left so both sides are preserved
        rename_map = {}
        for col in on:
            rename_map[col] = f"{col}{suffix[0]}"
            if left_on is not None:
                left_on = [rename_map.get(c, c) for c in left_on]
        if rename_map:
            x = x.rename(rename_map)
            if on is not None:
                on = [rename_map.get(c, c) for c in on]

    return x, y, left_on, right_on, on, suffix_str


def _join(
    x: Tibble,
    y: Tibble,
    how: str,
    by: Any = None,
    copy: bool = False,
    suffix: tuple = ("_x", "_y"),
    keep: bool = False,
) -> Tibble:
    # LazyFrames are immutable — no need to clone
    x_df = x
    y_df = y

    # Cast null-type columns to match the other side's types for joins
    x_cols = list(x_df.collect_schema().names())
    y_cols = list(y_df.collect_schema().names())
    x_schema = x_df.collect_schema()
    y_schema = y_df.collect_schema()

    for col in x_cols:
        if x_schema[col] == pl.Null and col in y_cols:
            x_df = x_df.with_columns(pl.col(col).cast(y_schema[col]))
    for col in y_cols:
        if y_schema[col] == pl.Null and col in x_cols:
            y_df = y_df.with_columns(pl.col(col).cast(x_schema[col]))

    x_df, y_df, left_on, right_on, on, suffix_str = _join_prep(
        x_df, y_df, by, suffix, keep
    )

    if on is None and left_on is None and right_on is None:
        # Cross join (no common columns)
        result = x_df.join(y_df, how="cross", suffix=suffix_str)
        return reconstruct_tibble(result, x)

    maintain_order = "right" if how == "right" else "left"

    if left_on is not None and right_on is not None:
        result = x_df.join(
            y_df,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffix=suffix_str,
            coalesce=True,
            maintain_order=maintain_order,
        )
    else:
        result = x_df.join(
            y_df,
            on=on,
            how=how,
            suffix=suffix_str,
            coalesce=True,
            maintain_order=maintain_order,
        )

    return reconstruct_tibble(result, x)


@inner_join.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _inner_join(
    x: Tibble,
    y: Tibble,
    *,
    by: Any = None,
    copy: bool = False,
    suffix: tuple = ("_x", "_y"),
    keep: bool = False,
) -> Tibble:
    return _join(x, y, how="inner", by=by, copy=copy, suffix=suffix, keep=keep)


@left_join.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _left_join(
    x: Tibble,
    y: Tibble,
    *,
    by: Any = None,
    copy: bool = False,
    suffix: tuple = ("_x", "_y"),
    keep: bool = False,
) -> Tibble:
    return _join(x, y, how="left", by=by, copy=copy, suffix=suffix, keep=keep)


@right_join.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _right_join(
    x: Tibble,
    y: Tibble,
    *,
    by: Any = None,
    copy: bool = False,
    suffix: tuple = ("_x", "_y"),
    keep: bool = False,
) -> Tibble:
    return _join(x, y, how="right", by=by, copy=copy, suffix=suffix, keep=keep)


@full_join.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _full_join(
    x: Tibble,
    y: Tibble,
    *,
    by: Any = None,
    copy: bool = False,
    suffix: tuple = ("_x", "_y"),
    keep: bool = False,
) -> Tibble:
    return _join(x, y, how="full", by=by, copy=copy, suffix=suffix, keep=keep)


@semi_join.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _semi_join(
    x: Tibble,
    y: Tibble,
    *,
    by: Any = None,
    copy: bool = False,
) -> Tibble:
    return _join(x, y, how="semi", by=by, copy=copy)


@anti_join.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _anti_join(
    x: Tibble,
    y: Tibble,
    *,
    by: Any = None,
    copy: bool = False,
) -> Tibble:
    return _join(x, y, how="anti", by=by, copy=copy)


@nest_join.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _nest_join(
    x: Tibble,
    y: Tibble,
    *,
    by: Any = None,
    copy: bool = False,
    keep: bool = False,
    name: Optional[str] = None,
) -> Tibble:
    x_df = x.collect()
    y_df = y.collect()
    if copy:
        x_df = x_df.clone()

    left_on, right_on, on = _resolve_by(x, y, by)

    if left_on is not None and right_on is not None:
        key_pairs = list(zip(left_on, right_on))
    elif on:
        key_pairs = [(k, k) for k in on]
    else:
        key_pairs = []

    y_out_cols = list(y_df.collect_schema().names())
    if not keep:
        right_keys = {rk for _, rk in key_pairs}
        y_out_cols = [c for c in y_out_cols if c not in right_keys]

    nested_dfs = []
    for row in x_df.iter_rows(named=True):
        mask = pl.lit(True)
        for lk, rk in key_pairs:
            mask = mask & (pl.col(rk) == row[lk])
        matched = y_df.filter(mask).select(y_out_cols)
        nested_dfs.append(matched)

    y_name = name or "_y_joined"
    result = x_df.with_columns(
        pl.Series(y_name, nested_dfs, dtype=pl.Object).alias(y_name)
    )
    return reconstruct_tibble(result, x)


@cross_join.register((Tibble, LazyTibble), backend="polars")
def _cross_join(
    x: Tibble,
    y: Tibble,
    *,
    copy: bool = False,
    suffix: tuple = ("_x", "_y"),
) -> Tibble:
    result = x.join(y, how="cross", suffix=suffix[1])
    return reconstruct_tibble(result, x)
