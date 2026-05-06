"""Subset rows using their positions

https://github.com/tidyverse/dplyr/blob/master/R/slice.R
"""

from __future__ import annotations

import builtins
from math import ceil, floor
from typing import TYPE_CHECKING, Any, Iterable, Optional, Union

import polars as pl
from pipda import Expression
from datar.core.utils import logger
from datar.apis.dplyr import (
    slice_,
    slice_head,
    slice_tail,
    slice_sample,
    slice_min,
    slice_max,
)

from ...contexts import Context
from ...tibble import (
    Tibble,
    LazyTibble,
    reconstruct_tibble,
    to_lazy,
    to_eager,
)
from ...common import is_scalar
from ...collections import Collection

if TYPE_CHECKING:
    pass


def _n_from_prop(
    total: int,
    n: Optional[int | float] = None,
    prop: Optional[float] = None,
) -> int:
    """Get n from a proportion"""
    if n is None and prop is None:
        return 1
    if n is not None and not isinstance(n, (int, float)):
        raise TypeError(f"Expect `n` a number, got {type(n)}.")
    if prop is not None and not isinstance(prop, (int, float)):
        raise TypeError(f"Expect `prop` a number, got {type(prop)}.")
    if prop is not None:
        if prop < 0:
            return max(ceil((1.0 + prop) * total), 0)
        return floor(float(total) * min(prop, 1.0))

    n_val = 1.0 if n is None else float(n)
    if n_val < 0:
        return max(ceil(total + n_val), 0)
    return min(floor(n_val), total)


# ── Shared helpers ─────────────────────────────────────────────────────────


def _count_rows(data: pl.LazyFrame) -> int:
    """Count rows in a LazyFrame. For DataFrame, use len()."""
    return data.select(pl.len()).collect().item()


def _evaluate_exprs(rows: tuple, _data: pl.LazyFrame) -> tuple:
    """Evaluate pipda expressions in rows to concrete values.

    In Context.PENDING, verb arguments like n()-1 arrive as raw pipda
    Expression objects. These must be evaluated against _data to produce
    concrete row indices (integers) before Collection can expand them.
    """
    out = []
    for r in rows:
        if isinstance(r, Expression):
            evald = r._pipda_eval(_data, Context.EVAL)
            if isinstance(evald, pl.Expr):
                evald = _data.select(
                    evald.alias("__slice_val")
                ).collect().item()
            out.append(evald)
        else:
            out.append(r)
    return tuple(out)


def _sanitize_rows(rows: Iterable, nrows: int) -> list:
    """Sanitize rows passed to slice_."""
    rows_list = Collection(*rows, pool=nrows)
    if rows_list.error:
        raise rows_list.error from None
    return list(rows_list)


# ── slice_ ──────────────────────────────────────────────────────────────────


@slice_.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _slice_lazy(
    _data: pl.LazyFrame,
    *rows: Union[int, str],
    _preserve: bool = False,
):
    if _preserve:
        logger.warning("`slice()` doesn't support `_preserve` argument yet.")

    if not rows:
        return reconstruct_tibble(_data, _data)

    rows = _evaluate_exprs(rows, _data)
    nrows = _count_rows(_data)
    rows_idx = _sanitize_rows(rows, nrows)

    if len(rows_idx) == 0:
        result = _data.clear()
        return reconstruct_tibble(result, _data)

    result = (
        _data.with_row_index(name="_datar_row")
        .filter(pl.col("_datar_row").is_in(list(rows_idx)))
        .drop("_datar_row")
    )
    return reconstruct_tibble(result, _data)


@slice_.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _slice_eager(
    _data: pl.DataFrame,
    *rows: Union[int, str],
    _preserve: bool = False,
) -> Tibble:
    return to_eager(
        _slice_lazy(to_lazy(_data), *rows, _preserve=_preserve)
    )


# ── slice_head ──────────────────────────────────────────────────────────────


@slice_head.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _slice_head_lazy(
    _data: pl.LazyFrame,
    n: Optional[int] = None,
    prop: Optional[float] = None,
):
    nrows = _count_rows(_data)
    n_val = _n_from_prop(nrows, n, prop)
    result = _data.slice(0, n_val)
    return reconstruct_tibble(result, _data)


@slice_head.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _slice_head_eager(
    _data: pl.DataFrame,
    n: Optional[int] = None,
    prop: Optional[float] = None,
) -> Tibble:
    return to_eager(
        _slice_head_lazy(to_lazy(_data), n=n, prop=prop)
    )


# ── slice_tail ──────────────────────────────────────────────────────────────


@slice_tail.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _slice_tail_lazy(
    _data: pl.LazyFrame,
    n: int = 1,
    prop: Optional[float] = None,
):
    nrows = _count_rows(_data)
    n_val = _n_from_prop(nrows, n, prop)
    if n_val == 0:
        result = _data.clear()
    else:
        result = _data.slice(nrows - n_val, n_val)
    return reconstruct_tibble(result, _data)


@slice_tail.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _slice_tail_eager(
    _data: pl.DataFrame,
    n: int = 1,
    prop: Optional[float] = None,
) -> Tibble:
    return to_eager(
        _slice_tail_lazy(to_lazy(_data), n=n, prop=prop)
    )


# ── slice_sample ────────────────────────────────────────────────────────────


@slice_sample.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _slice_sample_lazy(
    _data: pl.LazyFrame,
    n: int = 1,
    prop: Optional[float] = None,
    weight_by: Optional[Iterable[Union[int, float]]] = None,
    replace: bool = False,
    random_state: Any = None,
):
    nrows = _count_rows(_data)
    n_val = _n_from_prop(nrows, n, prop)

    if n_val == 0:
        result = _data.clear()
        return reconstruct_tibble(result, _data)

    df = _data.collect()
    sampled = df.sample(
        n=n_val,
        with_replacement=replace,
        seed=random_state,
    )
    result = LazyTibble(sampled.lazy(), _datar=_data._datar.copy())
    return reconstruct_tibble(result, _data)


@slice_sample.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _slice_sample_eager(
    _data: pl.DataFrame,
    n: int = 1,
    prop: Optional[float] = None,
    weight_by: Optional[Iterable[Union[int, float]]] = None,
    replace: bool = False,
    random_state: Any = None,
) -> Tibble:
    return to_eager(
        _slice_sample_lazy(
            to_lazy(_data),
            n=n,
            prop=prop,
            weight_by=weight_by,
            replace=replace,
            random_state=random_state,
        )
    )


# ── slice_min ───────────────────────────────────────────────────────────────


def _extract_sort_col(order_by: Any, _data: pl.LazyFrame) -> str:
    """Extract sort column name from order_by argument."""
    if isinstance(order_by, str):
        return order_by
    if isinstance(order_by, Expression):
        try:
            from pipda import evaluate_expr

            col_expr = evaluate_expr(order_by, _data, Context.EVAL)
            if isinstance(col_expr, pl.Expr):
                return (
                    col_expr.meta.output_name()
                    or col_expr.meta.root_names()[0]
                )
        except Exception:
            pass
    return str(order_by)


@slice_min.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _slice_min_lazy(
    _data: pl.LazyFrame,
    order_by: Any,
    n: int = 1,
    prop: Optional[float] = None,
    with_ties: Union[bool, str] = True,
):
    nrows = _count_rows(_data)
    n_val = _n_from_prop(nrows, n, prop)
    sort_col = _extract_sort_col(order_by, _data)
    result = _data.sort(sort_col, nulls_last=True).slice(0, n_val)
    return reconstruct_tibble(result, _data)


@slice_min.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _slice_min_eager(
    _data: pl.DataFrame,
    order_by: Any,
    n: int = 1,
    prop: Optional[float] = None,
    with_ties: Union[bool, str] = True,
) -> Tibble:
    return to_eager(
        _slice_min_lazy(
            to_lazy(_data),
            order_by=order_by,
            n=n,
            prop=prop,
            with_ties=with_ties,
        )
    )


# ── slice_max ───────────────────────────────────────────────────────────────


@slice_max.register(pl.LazyFrame, context=Context.PENDING, backend="polars")
def _slice_max_lazy(
    _data: pl.LazyFrame,
    order_by: Any,
    n: int = 1,
    prop: Optional[float] = None,
    with_ties: Union[bool, str] = True,
):
    nrows = _count_rows(_data)
    n_val = _n_from_prop(nrows, n, prop)
    sort_col = _extract_sort_col(order_by, _data)
    result = _data.sort(
        sort_col, descending=True, nulls_last=True
    ).slice(0, n_val)
    return reconstruct_tibble(result, _data)


@slice_max.register(pl.DataFrame, context=Context.PENDING, backend="polars")
def _slice_max_eager(
    _data: pl.DataFrame,
    order_by: Any,
    n: int = 1,
    prop: Optional[float] = None,
    with_ties: Union[bool, str] = True,
) -> Tibble:
    return to_eager(
        _slice_max_lazy(
            to_lazy(_data),
            order_by=order_by,
            n=n,
            prop=prop,
            with_ties=with_ties,
        )
    )
