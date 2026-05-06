"""Subset distinct/unique rows

See source https://github.com/tidyverse/dplyr/blob/master/R/distinct.R
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.dplyr import distinct, mutate
from datar.apis.dplyr import n_distinct

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...common import union, setdiff, intersect
from ...utils import vars_select


# ---- n_distinct ---------------------------------------------------------


@n_distinct.register(pl.Expr, context=Context.EVAL, backend="polars")
def _n_distinct_expr(_data: pl.Expr, na_rm: bool = True) -> pl.Expr:
    if na_rm:
        return _data.drop_nulls().n_unique()
    return _data.n_unique()


@n_distinct.register(object, context=Context.EVAL, backend="polars")
def _n_distinct_obj(_data: Any, na_rm: bool = True) -> Any:
    import math

    if isinstance(_data, pl.Series):
        if na_rm:
            return _data.drop_nulls().n_unique()
        return _data.n_unique()
    if hasattr(_data, "__iter__") and not isinstance(_data, (str, bytes)):
        s = pl.Series(list(_data))
        if na_rm:
            return s.drop_nulls().n_unique()
        return s.n_unique()
    # Scalar
    if _data is None:
        return 0
    if isinstance(_data, float) and math.isnan(_data):
        return 0 if na_rm else 1
    return 1


@n_distinct.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _n_distinct_tibble(_data: Tibble, na_rm: bool = True) -> int:
    return _data.unique().collect().height


# ---- distinct -----------------------------------------------------------


@distinct.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _distinct(
    _data: Tibble,
    *args: Any,
    _keep_all: bool = False,
    **kwargs: Any,
) -> Tibble:
    gvars = (
        _data._datar.get("groups", [])
        if hasattr(_data, "_datar") and _data._datar.get("groups") is not None
        else []
    )

    if not args and not kwargs:
        out = _data.unique(maintain_order=True)
        return reconstruct_tibble(out, _data)

    all_cols = _data.collect_schema().names()

    # Fast path: only simple column references in args, no kwargs expressions
    if (
        not kwargs
        and all(
            isinstance(a, (str,))
            or (hasattr(a, "_pipda_ref") and isinstance(a._pipda_ref, str))
            for a in args
        )
    ):
        selected_idx = vars_select(all_cols, *args)
        selected_cols = [all_cols[i] for i in selected_idx]
        subset = union(gvars, selected_cols)
        out = _data.unique(subset=subset, maintain_order=True)
        if not _keep_all:
            out = out.select(
                [c for c in subset if c in out.collect_schema().names()]
            )
        return reconstruct_tibble(out, _data)

    # General path: use mutate to evaluate expressions, then distinct
    mutated = mutate(
        _data,
        *args,
        **kwargs,
        _keep="none",
        __ast_fallback="normal",
        __backend="polars",
    )
    mutated_cols = mutated.collect_schema().names()
    out = mutated.unique(maintain_order=True)

    if not _keep_all:
        keep_cols = list(kwargs.keys())
        if args:
            arg_idx = vars_select(all_cols, *args)
            keep_cols = union(keep_cols, [all_cols[i] for i in arg_idx])
        keep_cols = union(gvars, keep_cols)
        out = out.select(
            [c for c in keep_cols if c in mutated_cols]
        )

    return reconstruct_tibble(out, _data)
