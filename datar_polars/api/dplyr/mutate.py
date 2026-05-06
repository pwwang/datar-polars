"""Create, modify, and delete columns

See source https://github.com/tidyverse/dplyr/blob/master/R/mutate.R
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, cast

import polars as pl
from pipda import evaluate_expr, ReferenceAttr, ReferenceItem

from datar.core.utils import arg_match
from datar.dplyr import mutate, transmute

from ...contexts import Context
from ...collections import Collection
from ...utils import name_of
from ...broadcast import add_to_tibble
from ...tibble import (
    Tibble,
    LazyTibble,
    reconstruct_tibble,
    to_lazy,
    to_eager,
)
from ...common import setdiff, union, intersect
from ..forcats import _EVAL_DATA
from .context import _GroupEvalExpr


def _get_gvars(data) -> list:
    """Return grouping variable names."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


def _relocate_polars(
    data,
    new_cols: list,
    _before: int | str = None,
    _after: int | str = None,
):
    """Move *new_cols* before/after a reference column."""
    remaining = [c for c in data.collect_schema().names() if c not in new_cols]

    if isinstance(_before, ReferenceAttr):
        _before = str(_before._pipda_ref)
    if isinstance(_after, ReferenceAttr):
        _after = str(_after._pipda_ref)

    if _before is not None:
        idx = remaining.index(_before) if isinstance(_before, str) else _before
        order = remaining[:idx] + new_cols + remaining[idx:]
    elif _after is not None:
        idx = (
            remaining.index(_after) + 1
            if isinstance(_after, str)
            else _after + 1
        )
        order = remaining[:idx] + new_cols + remaining[idx:]
    else:
        order = new_cols + remaining

    return data.select(order)


def _mutate_impl(
    _data,
    *args: Any,
    _keep: str = "all",
    _before: int | str = None,
    _after: int | str = None,
    **kwargs: Any,
):
    """Shared implementation for mutate on lazy data."""
    keep = arg_match(_keep, "_keep", ["all", "unused", "used", "none", "trans"])
    gvars = _get_gvars(_data)
    all_columns = _data.collect_schema().names()

    data = _data
    if not hasattr(data, "_datar") or data._datar is None:
        import copy
        if hasattr(_data, "_datar"):
            data._datar = copy.copy(_data._datar)
        else:
            data._datar = {}
    data._datar["used_refs"] = set()

    mutated_cols: list = []

    # Detect rowwise mode
    is_rowwise = (
        data._datar.get("rowwise", False)
        if hasattr(data, "_datar")
        else False
    )

    if is_rowwise:
        # Row-wise mode: evaluate per row with scalar values
        collected = data.collect()
        col_names = list(collected.columns)
        result_rows = []

        for i in range(collected.shape[0]):
            row_dict = dict(zip(col_names, collected.row(i)))

            # Evaluate positional args
            for val in args:
                if (
                    isinstance(val, (ReferenceItem, ReferenceAttr))
                    and val._pipda_level == 1
                    and val._pipda_ref in col_names
                ):
                    continue  # just a column ref, no new column
                evaluated = evaluate_expr(val, row_dict, cast(Any, Context.EVAL))
                if evaluated is not None:
                    key = name_of(val)
                    row_dict[key] = evaluated

            # Evaluate keyword args
            for key, val_expr in kwargs.items():
                evaluated = evaluate_expr(val_expr, row_dict, cast(Any, Context.EVAL))
                row_dict[key] = evaluated

            result_rows.append(row_dict)

        new_columns = [c for c in result_rows[0].keys() if c not in col_names] if result_rows else []
        mutated_cols = list(new_columns)

        result_df = pl.DataFrame(result_rows)
        result_df = result_df.select(
            [c for c in result_df.columns if c in col_names or c in new_columns]
        )

        data = Tibble(result_df)
        data = reconstruct_tibble(data, _data)
        data._datar["mutated_cols"] = mutated_cols
        return data

    # ---- positional args ---------------------------------------------------
    for val in args:
        if (
            isinstance(val, (ReferenceItem, ReferenceAttr))
            and val._pipda_level == 1
            and val._pipda_ref in data.collect_schema().names()
        ):
            mutated_cols.append(val._pipda_ref)
            continue

        bkup_name = name_of(val)
        _EVAL_DATA.set(data)
        val = evaluate_expr(val, data, cast(Any, Context.EVAL))
        if val is None:
            continue

        if isinstance(val, (list, tuple)) and val:
            if all(isinstance(v, pl.Expr) for v in val):
                # List of expressions (from across()) — expand each
                for expr in val:
                    key = (
                        expr.meta.output_name()
                        if hasattr(expr, "meta")
                        and expr.meta.output_name()
                        else name_of(expr)
                    )
                    mutated_cols.append(key)
                    data = add_to_tibble(
                        data, key, expr, broadcast_tbl=False
                    )
            else:
                # Plain list/tuple — treat as a column value
                key = name_of(val) or bkup_name
                mutated_cols.append(key)
                data = add_to_tibble(data, key, val, broadcast_tbl=False)
        elif isinstance(val, (pl.DataFrame, pl.LazyFrame)):
            mutated_cols.extend(list(val.columns))
            data = add_to_tibble(data, None, val, broadcast_tbl=False)
        elif isinstance(val, _GroupEvalExpr):
            key = name_of(val) or bkup_name
            mutated_cols.append(key)
            data = add_to_tibble(
                data, key, val.expr, broadcast_tbl=False
            )
        else:
            key = name_of(val) or bkup_name
            mutated_cols.append(key)
            data = add_to_tibble(data, key, val, broadcast_tbl=False)

    # ---- keyword args ------------------------------------------------------
    for key, val in kwargs.items():
        _EVAL_DATA.set(data)
        val = evaluate_expr(val, data, cast(Any, Context.EVAL))
        if val is None:
            schema_names = data.collect_schema().names()
            if key in schema_names:
                with suppress(Exception):
                    data = data.drop(key)
        elif isinstance(val, (list, tuple)) and val:
            if all(isinstance(v, pl.Expr) for v in val):
                # List of expressions (from across()) — expand each
                for expr in val:
                    expr_key = (
                        expr.meta.output_name()
                        if hasattr(expr, "meta") and expr.meta.output_name()
                        else name_of(expr)
                    )
                    mutated_cols.append(expr_key)
                    data = add_to_tibble(
                        data, expr_key, expr, broadcast_tbl=False
                    )
            else:
                # Plain list/tuple — treat as a column value
                data = add_to_tibble(data, key, val, broadcast_tbl=False)
                mutated_cols.append(key)
        elif isinstance(val, _GroupEvalExpr):
            data = add_to_tibble(
                data, key, val.expr, broadcast_tbl=False
            )
            mutated_cols.append(key)
        elif isinstance(val, pl.Expr):
            data = add_to_tibble(data, key, val, broadcast_tbl=False)
            mutated_cols.append(key)
        else:
            data = add_to_tibble(data, key, val, broadcast_tbl=False)
            if isinstance(val, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(val, pl.LazyFrame):
                    mutated_cols.extend(
                        {f"{key}${col}" for col in
                         val.collect_schema().names()}
                    )
                else:
                    mutated_cols.append(key)
            else:
                mutated_cols.append(key)

    # ---- temporary columns (names starting with "_") -----------------------
    used_refs = data._datar.get("used_refs", set())
    tmp_cols = [
        mcol
        for mcol in mutated_cols
        if mcol.startswith("_")
        and mcol in used_refs
        and mcol not in _data.collect_schema().names()
    ]
    mutated_cols = intersect(mutated_cols, list(data.collect_schema().names()))
    mutated_cols = setdiff(mutated_cols, tmp_cols)

    for tc in tmp_cols:
        with suppress(Exception):
            data = data.drop(tc)

    # ---- reorder if requested ----------------------------------------------
    if _before is not None or _after is not None:
        new_cols = list(setdiff(mutated_cols, _data.collect_schema().names()))
        data = _relocate_polars(data, new_cols, _before=_before, _after=_after)

    # ---- keep logic --------------------------------------------------------
    current_cols = list(data.collect_schema().names())
    if keep == "all":
        keep_cols = current_cols
    elif keep == "unused":
        unused = setdiff(all_columns, list(used_refs))
        keep_cols = intersect(
            current_cols, list(gvars) + list(unused) + list(mutated_cols)
        )
    elif keep == "used":
        keep_cols = intersect(
            current_cols, list(gvars) + list(used_refs) + list(mutated_cols)
        )
    elif keep == "trans":
        keep_cols = union(
            setdiff(gvars, mutated_cols),
            intersect(mutated_cols, current_cols),
        )
    else:  # "none"
        keep_cols = intersect(
            current_cols, list(gvars) + list(mutated_cols)
        )

    data = data.select(keep_cols)

    data = reconstruct_tibble(data, _data)
    data._datar["mutated_cols"] = mutated_cols
    return data


# ---- Verb registrations -------------------------------------------------


@mutate.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _mutate_lazy(
    _data: pl.LazyFrame,
    *args: Any,
    _keep: str = "all",
    _before: int | str = None,
    _after: int | str = None,
    **kwargs: Any,
):
    return _mutate_impl(
        _data, *args, _keep=_keep, _before=_before, _after=_after, **kwargs
    )


@mutate.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _mutate_eager(
    _data: pl.DataFrame,
    *args: Any,
    _keep: str = "all",
    _before: int | str = None,
    _after: int | str = None,
    **kwargs: Any,
) -> Tibble:
    return to_eager(
        _mutate_impl(
            to_lazy(_data),
            *args,
            _keep=_keep,
            _before=_before,
            _after=_after,
            **kwargs,
        )
    )


@transmute.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _transmute_lazy(
    _data: pl.LazyFrame,
    *args: Any,
    _before: int | str = None,
    _after: int | str = None,
    **kwargs: Any,
):
    return mutate(
        _data,
        *args,
        _keep="trans",
        _before=_before,
        _after=_after,
        __ast_fallback="normal",
        __backend="polars",
        **kwargs,
    )


@transmute.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _transmute_eager(
    _data: pl.DataFrame,
    *args: Any,
    _before: int | str = None,
    _after: int | str = None,
    **kwargs: Any,
) -> Tibble:
    return to_eager(
        _transmute_lazy(
            to_lazy(_data),
            *args,
            _before=_before,
            _after=_after,
            **kwargs,
        )
    )
