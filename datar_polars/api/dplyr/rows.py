"""Provide functions to manipulate multiple rows.

https://github.com/tidyverse/dplyr/blob/master/R/rows.R
"""

from __future__ import annotations

from typing import Any

from datar.core.utils import logger, arg_match
from datar.apis.dplyr import (
    bind_rows,
    rows_insert,
    rows_append,
    rows_update,
    rows_patch,
    rows_upsert,
    rows_delete,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...common import is_scalar, setdiff


def _get_gvars(data: Tibble) -> list:
    """Get group variable names."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


def _rows_check_key(by: Any, x: Tibble, y: Tibble) -> list[str]:
    """Validate and return the join key columns."""
    if by is None:
        by = list(y.columns)[0]
        logger.info("Matching, by=%r", by)

    if is_scalar(by):
        by = [by]

    for b in by:
        if not isinstance(b, str):
            raise ValueError("`by` must be a string or a list of strings.")

    bad = setdiff(y.collect_schema().names(), x.collect_schema().names())
    if bad:
        raise ValueError(
            f"All columns in `y` must exist in `x`. Missing: {bad}"
        )

    return list(by)


def _rows_check_key_df(df: Tibble, by: list[str], df_name: str) -> None:
    """Check that key columns exist in the dataframe."""
    missing = setdiff(by, df.collect_schema().names())
    if missing:
        raise ValueError(
            f"All `by` columns must exist in `{df_name}`. Missing: {missing}"
        )


def _rows_find_matching_keys(
    x: Tibble, y: Tibble, by: list[str]
) -> tuple:
    """Find rows in x matching keys from y. Returns (x_matched_df, y_unique_keys)."""
    import polars as pl

    x_df = x.collect()
    y_df = y.collect()

    # Find matching keys
    matched = x_df.join(y_df.select(by).unique(), on=by, how="semi")
    return matched, y_df.select(by).unique()


# ── rows_insert ─────────────────────────────────────────────────────────────


@rows_insert.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _rows_insert(
    x: Tibble,
    y: Tibble,
    by: Any = None,
    conflict: str = "error",
    **kwargs: Any,
) -> Tibble:
    """Insert rows from y into x."""
    if kwargs:
        raise ValueError(f"Unsupported arguments: {list(kwargs.keys())}")

    conflict = arg_match(conflict, "conflict", ["error", "ignore"])

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, "x")
    _rows_check_key_df(y, key, "y")

    import polars as pl

    x_df = x.collect()
    y_df = y.collect()

    # Find rows in y whose keys don't exist in x
    existing_keys = x_df.select(key).unique()
    new_rows = y_df.join(existing_keys, on=key, how="anti")

    if conflict == "error":
        conflict_rows = y_df.join(existing_keys, on=key, how="semi")
        if conflict_rows.shape[0] > 0:
            raise ValueError("Attempting to insert duplicate rows.")

    # Bind
    result = pl.concat([x_df, new_rows], how="diagonal_relaxed")
    return Tibble(result.lazy())


# ── rows_append ─────────────────────────────────────────────────────────────


@rows_append.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _rows_append(
    x: Tibble,
    y: Tibble,
    **kwargs: Any,
) -> Tibble:
    """Append rows from y to x."""
    if kwargs:
        raise ValueError(f"Unsupported arguments: {list(kwargs.keys())}")

    _rows_check_key_df(x, list(y.collect_schema().names()), "x")

    import polars as pl

    result = pl.concat([x.collect(), y.collect()], how="diagonal_relaxed")
    return Tibble(result.lazy())


# ── rows_update ─────────────────────────────────────────────────────────────


@rows_update.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _rows_update(
    x: Tibble,
    y: Tibble,
    by: Any = None,
    unmatched: str = "error",
    **kwargs: Any,
) -> Tibble:
    """Update rows in x with values from y."""
    if kwargs:
        raise ValueError(f"Unsupported arguments: {list(kwargs.keys())}")

    unmatched = arg_match(unmatched, "unmatched", ["error", "ignore"])

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, "x")
    _rows_check_key_df(y, key, "y")

    import polars as pl

    x_df = x.collect()
    y_df = y.collect()

    # Check all y keys exist in x
    existing_keys = x_df.select(key).unique()
    missing_keys = y_df.select(key).unique().join(existing_keys, on=key, how="anti")

    if missing_keys.shape[0] > 0 and unmatched == "error":
        raise ValueError("Attempting to update missing rows.")

    # Check y keys are unique
    if y_df.shape[0] != y_df.select(key).unique().shape[0]:
        raise ValueError("`y` key values must be unique.")

    # For each key in y, update the corresponding row in x
    # Use an anti-join to get rows not being updated, then concatenate updated rows
    keep_rows = x_df.join(y_df.select(key).unique(), on=key, how="anti")

    # Rows being updated: keep x's columns not in y, then add y's values
    x_cols_not_in_y = [c for c in x_df.columns if c not in y_df.columns]
    updated = x_df.join(y_df.select(key).unique(), on=key, how="semi")
    # Drop y-overlapping columns from updated x, then join with y
    updated = updated.select(key + x_cols_not_in_y).join(y_df, on=key, how="left")

    result = pl.concat([keep_rows, updated], how="diagonal_relaxed")
    return Tibble(result.lazy())


# ── rows_patch ──────────────────────────────────────────────────────────────


@rows_patch.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _rows_patch(
    x: Tibble,
    y: Tibble,
    by: Any = None,
    unmatched: str = "error",
    **kwargs: Any,
) -> Tibble:
    """Patch rows in x with values from y (only filling NAs)."""
    if kwargs:
        raise ValueError(f"Unsupported arguments: {list(kwargs.keys())}")

    unmatched = arg_match(unmatched, "unmatched", ["error", "ignore"])

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, "x")
    _rows_check_key_df(y, key, "y")

    import polars as pl

    x_df = x.collect()
    y_df = y.collect()

    # Check all y keys exist in x
    existing_keys = x_df.select(key).unique()
    missing_keys = y_df.select(key).unique().join(existing_keys, on=key, how="anti")

    if missing_keys.shape[0] > 0:
        if unmatched == "error":
            raise ValueError("`y` must contain keys that already exist in `x`.")
        y_df = y_df.join(existing_keys, on=key, how="semi")

    if y_df.shape[0] == 0:
        raise ValueError("Attempting to patch missing rows.")

    if y_df.shape[0] != y_df.select(key).unique().shape[0]:
        raise ValueError("`y` key values must be unique.")

    # For matching rows, coalesce: fill x's NAs with y's values
    other_cols = [c for c in y_df.columns if c not in key]

    # Anti-join: rows not in y
    keep_rows = x_df.join(y_df.select(key).unique(), on=key, how="anti")

    # Rows to patch
    to_patch = x_df.join(y_df.select(key).unique(), on=key, how="semi")

    # Coalesce each column
    for col in other_cols:
        col_in_patch = to_patch.select(key + [col]).join(
            y_df.select(key + [col]), on=key, how="left", suffix="_y"
        )
        filled = []
        for row_data in col_in_patch.iter_rows():
            val_x = row_data[1]
            val_y = row_data[2]
            if val_x is None:
                filled.append(val_y)
            else:
                filled.append(val_x)
        to_patch = to_patch.with_columns(
            pl.Series(col, filled).alias(col)
        )

    result = pl.concat([keep_rows, to_patch], how="diagonal_relaxed")
    return Tibble(result.lazy())


# ── rows_upsert ─────────────────────────────────────────────────────────────


@rows_upsert.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _rows_upsert(
    x: Tibble,
    y: Tibble,
    by: Any = None,
    **kwargs: Any,
) -> Tibble:
    """Upsert: update existing rows, insert new rows."""
    if kwargs:
        raise ValueError(f"Unsupported arguments: {list(kwargs.keys())}")

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, "x")
    _rows_check_key_df(y, key, "y")

    import polars as pl

    x_df = x.collect()
    y_df = y.collect()

    if y_df.shape[0] != y_df.select(key).unique().shape[0]:
        raise ValueError("`y` key values must be unique.")

    # Separate new rows (not in x) from update rows (in x)
    existing_keys = x_df.select(key).unique()
    new_rows = y_df.join(existing_keys, on=key, how="anti")
    update_rows = y_df.join(existing_keys, on=key, how="semi")

    # Keep x rows not being updated
    keep_rows = x_df.join(y_df.select(key).unique(), on=key, how="anti")

    # Updated rows
    x_cols_not_in_y = [c for c in x_df.columns if c not in y_df.columns]
    if update_rows.shape[0] > 0:
        updated = x_df.join(update_rows.select(key).unique(), on=key, how="semi")
        updated = updated.select(key + x_cols_not_in_y).join(
            update_rows, on=key, how="left"
        )
    else:
        updated = pl.DataFrame(schema=x_df.schema)

    result = pl.concat([keep_rows, updated, new_rows], how="diagonal_relaxed")
    return Tibble(result.lazy())


# ── rows_delete ─────────────────────────────────────────────────────────────


@rows_delete.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _rows_delete(
    x: Tibble,
    y: Tibble,
    by: Any = None,
    unmatched: str = "error",
    **kwargs: Any,
) -> Tibble:
    """Delete rows in x that match keys in y."""
    if kwargs:
        raise ValueError(f"Unsupported arguments: {list(kwargs.keys())}")

    unmatched = arg_match(unmatched, "unmatched", ["error", "ignore"])

    key = _rows_check_key(by, x, y)
    _rows_check_key_df(x, key, "x")
    _rows_check_key_df(y, key, "y")

    import polars as pl

    x_df = x.collect()
    y_df = y.collect()

    extra_cols = [c for c in y_df.columns if c not in key]
    if extra_cols:
        logger.info("Ignoring extra columns: %s", extra_cols)
        y_df = y_df.select(key)

    # Check all y keys exist in x
    existing_keys = x_df.select(key).unique()
    missing_keys = y_df.select(key).unique().join(existing_keys, on=key, how="anti")

    if missing_keys.shape[0] > 0 and unmatched == "error":
        raise ValueError("Attempting to delete missing rows.")

    # Anti-join to remove matching rows
    result = x_df.join(y_df.select(key).unique(), on=key, how="anti")
    return Tibble(result.lazy())
