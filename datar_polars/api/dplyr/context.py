"""Context dependent expressions

See source https://github.com/tidyverse/dplyr/blob/master/R/context.R
"""

from __future__ import annotations

from typing import Any, List, Optional

import polars as pl

from datar.apis.dplyr import (
    cur_data_all,
    cur_data,
    cur_group,
    cur_group_id,
    cur_group_rows,
    cur_column,
    n,
    consecutive_id,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...common import setdiff


def _get_gvars(data: Tibble) -> list:
    """Get group variable names."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


def _group_rows_impl(data: Tibble) -> List[List[int]]:
    """Compute group row indices."""
    gvars = _get_gvars(data)
    if not gvars:
        nrows = data.select(pl.len()).collect().item()
        return [list(range(nrows))]

    df = data.select(gvars).collect()
    groups: dict = {}
    for i, row in enumerate(df.iter_rows()):
        key = tuple(row)
        groups.setdefault(key, []).append(i)

    seen: list = []
    result: list = []
    for i, row in enumerate(df.iter_rows()):
        key = tuple(row)
        if key not in seen:
            seen.append(key)
            result.append(groups[key])
    return result


# ── n ────────────────────────────────────────────────────────────────────────


@n.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _n(_data: Tibble) -> int:
    """Return the number of rows in the current group."""
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    gvars = _get_gvars(_data)
    if not gvars:
        return _data.select(pl.len()).collect().item()
    return _data.select(gvars).collect().shape[0]


# ── _MultiValueExpr ───────────────────────────────────────────────────────────


class _MultiValueExpr:
    """Wrapper for a pl.Expr that returns multiple values per group.

    Used so that summarise/_build_agg_exprs can track which result columns
    should be exploded (multi-row semantics) vs. which are intentionally
    list-valued (e.g. cur_group_rows).

    For example, ``quantile(f.x, c(0.25, 0.75))`` returns a
    ``_MultiValueExpr`` containing a ``pl.concat_list(...)`` expression.
    After ``group_by().agg()``, only columns tagged this way are exploded.
    """

    def __init__(self, expr: "pl.Expr") -> None:
        self.expr = expr


# ── cur_data_all ─────────────────────────────────────────────────────────────


class _CurDataResult:
    """Holds pre-computed per-group DataFrames for cur_data / cur_data_all.

    These cannot be stored as Polars expression literals (nested object
    types are not supported), so we compute them eagerly and inject them
    after aggregation.

    Proxies shape and columns for ContextEval attribute access
    (e.g. ``cur_data_all().shape[0]``).
    """

    def __init__(self, dfs: list[pl.DataFrame]) -> None:
        self.dfs = dfs

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the first (or only) group DataFrame."""
        return self.dfs[0].shape

    @property
    def columns(self) -> list[str]:
        """Columns of the first (or only) group DataFrame."""
        return self.dfs[0].columns


def _build_cur_data_dfs(
    _data: Tibble,
    include_gvars: bool,
) -> _CurDataResult:
    """Eagerly compute per-group DataFrames for cur_data(_all)."""
    gvars = _get_gvars(_data)
    if not gvars:
        df = _data.collect()
        if not include_gvars:
            df = df.select(
                [c for c in df.columns if c not in gvars]
            )
        return _CurDataResult([df])

    all_cols = _data.collect_schema().names()
    keep_cols = (
        all_cols
        if include_gvars
        else [c for c in all_cols if c not in gvars]
    )

    select_cols = keep_cols + [gv for gv in gvars if gv not in keep_cols]
    df = _data.select(select_cols).collect()
    group_dfs: list[pl.DataFrame] = []
    for _, group_df in df.group_by(gvars, maintain_order=True):
        group_dfs.append(group_df.select(keep_cols))
    return _CurDataResult(group_dfs)


@cur_data_all.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _cur_data_all(_data: Tibble) -> _CurDataResult:
    """Return per-group DataFrames including grouping variables."""
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    return _build_cur_data_dfs(_data, include_gvars=True)


# ── cur_data ─────────────────────────────────────────────────────────────────


@cur_data.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _cur_data(_data: Tibble) -> _CurDataResult:
    """Return per-group DataFrames excluding grouping variables."""
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    return _build_cur_data_dfs(_data, include_gvars=False)


# ── cur_group ────────────────────────────────────────────────────────────────


@cur_group.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _cur_group(_data: Tibble) -> _GroupEvalExpr:
    """Return a pl.Expr that computes the group key for each row/group."""
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    gvars = _get_gvars(_data)
    if not gvars:
        return _GroupEvalExpr(pl.lit(None))
    return _GroupEvalExpr(pl.struct(gvars))


# ── cur_group_id ─────────────────────────────────────────────────────────────


@cur_group_id.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _cur_group_id(_data: Tibble) -> pl.Expr:
    """Return a pl.Expr that computes group IDs (0-based) for each row.

    Builds a when/then chain mapping each unique group combination
    (in order of first appearance) to its index.
    """
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    gvars = _get_gvars(_data)
    if not gvars:
        return pl.lit(0)

    unique_df = _data.select(gvars).unique(maintain_order=True).collect()

    expr: pl.Expr | None = None
    for i in range(unique_df.height):
        row = unique_df.row(i)
        cond = pl.col(gvars[0]) == row[0]
        for j in range(1, len(gvars)):
            cond = cond & (pl.col(gvars[j]) == row[j])
        if expr is None:
            expr = pl.when(cond).then(pl.lit(i))
        else:
            expr = expr.when(cond).then(pl.lit(i))

    return expr.otherwise(pl.lit(-1))


# ── cur_group_rows ───────────────────────────────────────────────────────────


class _GroupEvalExpr:
    """Wraps a pl.Expr for per-row group-context functions.

    Summarise should apply .first() inside agg() to deduplicate the
    per-row expression result; mutate should use the raw expression.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self.expr = expr


@cur_group_rows.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _cur_group_rows(_data: Tibble) -> _GroupEvalExpr:
    """Return a pl.Expr that computes row-index lists for each group."""
    _data = getattr(_data, "_datar", {}).get("summarise_source", _data)
    gvars = _get_gvars(_data)
    if not gvars:
        nrows = _data.select(pl.len()).collect().item()
        return _GroupEvalExpr(pl.lit(list(range(nrows))))

    rows = _group_rows_impl(_data)
    unique_df = _data.select(gvars).unique(maintain_order=True).collect()

    expr: pl.Expr | None = None
    for i in range(unique_df.height):
        row = unique_df.row(i)
        cond = pl.col(gvars[0]) == row[0]
        for j in range(1, len(gvars)):
            cond = cond & (pl.col(gvars[j]) == row[j])
        if expr is None:
            expr = pl.when(cond).then(pl.lit(rows[i]))
        else:
            expr = expr.when(cond).then(pl.lit(rows[i]))

    return _GroupEvalExpr(expr.otherwise(pl.lit([])))


# ── cur_column ───────────────────────────────────────────────────────────────


class CurColumn:
    """Marker for the current column name in across() calls."""

    @classmethod
    def replace_args(cls, args: tuple, column: str) -> tuple:
        """Replace CurColumn instances with the real column name in args."""
        return tuple(column if isinstance(arg, cls) else arg for arg in args)

    @classmethod
    def replace_kwargs(cls, kwargs: dict, column: str) -> dict:
        """Replace CurColumn instances with the real column name in kwargs."""
        return {
            key: column if isinstance(val, cls) else val
            for key, val in kwargs.items()
        }


@cur_column.register(backend="polars")
def _cur_column() -> CurColumn:
    """Return a CurColumn marker for use in across()."""
    return CurColumn()


# ── consecutive_id ───────────────────────────────────────────────────────────


@consecutive_id.register(object, backend="polars")
def _consecutive_id_obj(
    x: Any,
    *args: Any,
) -> Any:
    """Generate consecutive IDs for object dispatch."""
    return consecutive_id.dispatch(pl.Series)(
        pl.Series(x) if not isinstance(x, pl.Series) else x,
        *args,
    )


@consecutive_id.register(pl.Series, backend="polars")
def _consecutive_id_series(
    x: pl.Series,
    *args: Any,
) -> pl.Series:
    """Generate consecutive IDs for a Series.

    Each time a value changes, the ID increments.
    """
    if args:
        # Multiple series: combine them
        dfs = [x.to_frame("_x0")]
        for i, arg in enumerate(args):
            if isinstance(arg, pl.Series):
                dfs.append(arg.to_frame(f"_x{i+1}"))
            else:
                dfs.append(pl.Series(f"_x{i+1}", arg).to_frame())
        combined = pl.concat(dfs, how="horizontal")
        series = combined.select(
            pl.concat_str(pl.all(), separator="\x00").alias("_combined")
        ).get_column("_combined")
    else:
        series = x

    # Use rle_id for efficient consecutive ID generation
    result = series.rle_id()
    return result
