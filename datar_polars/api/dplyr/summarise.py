"""Summarise each group to fewer rows

See source https://github.com/tidyverse/dplyr/blob/master/R/summarise.R
"""

from __future__ import annotations

from typing import Any, Optional

import polars as pl
from polars.exceptions import ColumnNotFoundError
from pipda import evaluate_expr

from datar.core.utils import arg_match, logger
from datar.dplyr import summarise, reframe

from ...contexts import Context
from ...tibble import Tibble, LazyTibble, reconstruct_tibble
from ...utils import name_of
from ...common import is_scalar
from .context import _GroupEvalExpr, _CurDataResult, _MultiValueExpr


def _get_gvars(data: Tibble) -> list:
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


def _build_agg_exprs(
    data: Tibble,
    args: tuple,
    kwargs: dict,
) -> tuple[list[pl.Expr], dict[str, list[pl.DataFrame]], set[str], dict[str, pl.Expr]]:
    """Evaluate summarise expressions.

    Returns:
        - list of named pl.Expr for agg
        - dict of column names to per-group DataFrames (_CurDataResult)
        - set of column names that should be exploded (multi-row semantics)
        - dict of dependent exprs (reference columns created in this call)
    """
    exprs: list[pl.Expr] = []
    deferred: dict[str, list[pl.DataFrame]] = {}
    expand_cols: set[str] = set()
    dependent: dict[str, pl.Expr] = {}

    new_keys: set[str] = set(kwargs.keys())
    existing_cols: set[str] = set(data.collect_schema().names())

    for i, val in enumerate(args):
        if val is None:
            continue
        evaluated = evaluate_expr(val, data, Context.EVAL)
        if evaluated is None:
            continue
        if isinstance(evaluated, (Tibble, LazyTibble, pl.DataFrame, pl.LazyFrame)):
            evaluated = _frame_to_expr_items(evaluated)
        if isinstance(evaluated, (list, tuple)):
            for expr in evaluated:
                if isinstance(expr, _MultiValueExpr):
                    expr_key = expr.expr.meta.output_name() or f"expr_{i}"
                    expand_cols.add(expr_key)
                    exprs.append(expr.expr.alias(expr_key))
                elif isinstance(expr, _GroupEvalExpr):
                    expr_key = expr.expr.meta.output_name() or f"expr_{i}"
                    exprs.append(expr.expr.first().alias(expr_key))
                elif isinstance(expr, pl.Expr):
                    key = name_of(expr) or f"expr_{i}"
                    exprs.append(expr.alias(key))
                elif expr is not None:
                    key = name_of(expr) or f"expr_{i}"
                    if is_scalar(expr):
                        exprs.append(pl.lit(expr).alias(key))
                    else:
                        exprs.append(pl.lit(expr).alias(key))
            continue
        key = name_of(evaluated) or f"expr_{i}"
        if isinstance(evaluated, _CurDataResult):
            deferred[key] = evaluated.dfs
        elif isinstance(evaluated, _MultiValueExpr):
            expand_cols.add(key)
            exprs.append(evaluated.expr.alias(key))
        elif isinstance(evaluated, _GroupEvalExpr):
            exprs.append(evaluated.expr.first().alias(key))
        elif isinstance(evaluated, pl.Expr):
            exprs.append(evaluated.alias(key))
        elif isinstance(evaluated, pl.DataFrame):
            exprs.append(pl.lit(evaluated.to_dicts()).alias(key))
        elif is_scalar(evaluated):
            exprs.append(pl.lit(evaluated).alias(key))
        else:
            exprs.append(pl.lit(evaluated).alias(key))

    for key, val in kwargs.items():
        if val is None:
            exprs.append(pl.lit(None).alias(key))
            continue
        evaluated = evaluate_expr(val, data, Context.EVAL)
        if evaluated is None:
            exprs.append(pl.lit(None).alias(key))
        elif isinstance(evaluated, (Tibble, LazyTibble, pl.DataFrame, pl.LazyFrame)):
            for expr in _frame_to_expr_items(evaluated, prefix=key):
                if isinstance(expr, _MultiValueExpr):
                    expr_key = expr.expr.meta.output_name() or key
                    expand_cols.add(expr_key)
                    exprs.append(expr.expr.alias(expr_key))
                elif isinstance(expr, _GroupEvalExpr):
                    expr_key = expr.expr.meta.output_name() or key
                    exprs.append(expr.expr.first().alias(expr_key))
                elif isinstance(expr, pl.Expr):
                    expr_key = expr.meta.output_name() or key
                    exprs.append(expr.alias(expr_key))
        elif isinstance(evaluated, _CurDataResult):
            deferred[key] = evaluated.dfs
        elif isinstance(evaluated, _MultiValueExpr):
            expand_cols.add(key)
            exprs.append(evaluated.expr.alias(key))
        elif isinstance(evaluated, (list, tuple)):
            # Check if all items are non-Expr scalars → multi-row literal
            has_expr = any(
                isinstance(e, (pl.Expr, _MultiValueExpr, _GroupEvalExpr))
                for e in evaluated
            )
            if not has_expr:
                vals = list(evaluated)
                expand_cols.add(key)
                exprs.append(pl.lit(pl.Series(vals)).alias(key))
            else:
                for expr in evaluated:
                    if isinstance(expr, _MultiValueExpr):
                        expr_key = expr.expr.meta.output_name() or key
                        expand_cols.add(expr_key)
                        exprs.append(expr.expr.alias(expr_key))
                    elif isinstance(expr, _GroupEvalExpr):
                        expr_key = expr.expr.meta.output_name() or key
                        exprs.append(expr.expr.first().alias(expr_key))
                    elif isinstance(expr, pl.Expr):
                        expr_key = (
                            expr.meta.output_name()
                            if hasattr(expr, "meta") and expr.meta.output_name()
                            else name_of(expr)
                        )
                        exprs.append(expr.alias(expr_key))
                    elif expr is not None:
                        exprs.append(pl.lit(expr, allow_object=True).alias(key))
        elif isinstance(evaluated, _GroupEvalExpr):
            exprs.append(evaluated.expr.first().alias(key))
        elif isinstance(evaluated, pl.Expr):
            new_refs = {
                root
                for root in set(evaluated.meta.root_names())
                if root in new_keys and root not in existing_cols
            }
            if new_refs:
                dependent[key] = evaluated
            else:
                exprs.append(evaluated.alias(key))
        elif isinstance(evaluated, pl.DataFrame):
            exprs.append(pl.lit(evaluated.to_dicts()).alias(key))
        elif is_scalar(evaluated):
            exprs.append(pl.lit(evaluated).alias(key))
        else:
            exprs.append(pl.lit(evaluated).alias(key))

    return exprs, deferred, expand_cols, dependent


def _frame_to_expr_items(
    frame: Tibble | LazyTibble | pl.DataFrame | pl.LazyFrame,
    prefix: str | None = None,
) -> list[Any]:
    """Convert a tibble/dataframe helper result into named expressions.

    This lets reframe/summarise splice helper-returned tibbles like
    quantile_df(f.height) into regular aggregation expressions.
    """
    if isinstance(frame, (LazyTibble, pl.LazyFrame)):
        frame = frame.collect()
    elif isinstance(frame, LazyTibble):
        frame = frame.collect()

    items: list[Any] = []
    for col in frame.columns:
        name = f"{prefix}_{col}" if prefix else col
        values = frame.get_column(col).to_list()
        if not values:
            continue

        if all(isinstance(value, _MultiValueExpr) for value in values):
            items.append(_MultiValueExpr(values[0].expr.alias(name)))
            continue

        if all(isinstance(value, _GroupEvalExpr) for value in values):
            items.append(_GroupEvalExpr(values[0].expr.alias(name)))
            continue

        if all(isinstance(value, pl.Expr) for value in values):
            items.append(values[0].alias(name))
            continue

        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            items.append(_MultiValueExpr(pl.lit(list(values[0])).alias(name)))
            continue

        if len(values) > 1:
            items.append(_MultiValueExpr(pl.lit(list(values)).alias(name)))
            continue

        items.append(pl.lit(values[0], allow_object=True).alias(name))

    return items


@summarise.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _summarise(
    _data: Tibble,
    *args: Any,
    _groups: Optional[str] = None,
    **kwargs: Any,
) -> Tibble:
    _groups = arg_match(
        _groups, "_groups", ["drop", "drop_last", "keep", "rowwise", None]
    )
    gvars = _get_gvars(_data)

    # Build aggregation expressions
    agg_exprs, deferred, expand_cols, dependent = _build_agg_exprs(
        _data, args, kwargs
    )

    if not agg_exprs and not deferred:
        result = pl.DataFrame([{}])
        return reconstruct_tibble(result, _data)

    try:
        if agg_exprs:
            if gvars:
                result = _data.group_by(gvars, maintain_order=True).agg(agg_exprs)
                # Explode only tracked multi-value columns (e.g. multi-prob quantile)
                if expand_cols:
                    result = result.explode(list(expand_cols))
            else:
                result = _data.select(agg_exprs)
                # Explode only tracked multi-value columns
                if expand_cols:
                    result = result.explode(list(expand_cols))
        else:
            # Only deferred columns (e.g. cur_data) — build a skeleton result
            if gvars:
                result = _data.select(gvars).unique(
                    maintain_order=True
                ).collect().lazy()
            else:
                result = pl.LazyFrame([{}])
    except ColumnNotFoundError as e:
        raise KeyError(str(e)) from e

    # Apply dependent expressions (reference columns created in this call)
    if dependent:
        dep_exprs = [expr.alias(key) for key, expr in dependent.items()]
        result = result.with_columns(dep_exprs)

    # Inject deferred _CurDataResult columns
    if deferred:
        result = result.collect() if not isinstance(result, pl.DataFrame) else result
        for col_name, dfs in deferred.items():
            result = result.with_columns(
                pl.Series(col_name, dfs, dtype=pl.Object)
            )

    # Drop temporary columns (names beginning with _)
    _tmp_cols = [c for c in result.columns if c.startswith("_") and c not in gvars]
    if _tmp_cols:
        result = result.drop(_tmp_cols)

    if gvars:
        # Handle _groups parameter
        if _groups is None:
            if len(gvars) > 1:
                _groups = "drop_last"
            else:
                _groups = "keep"

        if _groups == "drop_last" and len(gvars) > 1:
            new_groups = gvars[:-1]
            logger.info(
                "`summarise()` has grouped output by %s "
                "(override with `_groups` argument)",
                new_groups,
            )
            result._datar = {"groups": new_groups if new_groups else None}
        elif _groups == "keep" and gvars:
            logger.info(
                "`summarise()` has grouped output by %s "
                "(override with `_groups` argument)",
                gvars,
            )
            result._datar = {"groups": gvars}
        elif _groups == "drop":
            result._datar = {"groups": None}

    return reconstruct_tibble(result, _data)


@reframe.register((Tibble, LazyTibble), context=Context.PENDING, backend="polars")
def _reframe(
    _data: Tibble,
    *args: Any,
    **kwargs: Any,
) -> Tibble:
    gvars = _get_gvars(_data)

    agg_exprs, deferred, expand_cols, dependent = _build_agg_exprs(
        _data, args, kwargs
    )

    if not agg_exprs and not deferred:
        result = pl.DataFrame([{}])
        return reconstruct_tibble(result, _data)

    try:
        if agg_exprs:
            if gvars:
                result = _data.group_by(gvars, maintain_order=True).agg(agg_exprs)
                if expand_cols:
                    result = result.explode(list(expand_cols))
            else:
                result = _data.select(agg_exprs)
                if expand_cols:
                    result = result.explode(list(expand_cols))
        else:
            if gvars:
                result = _data.select(gvars).unique(
                    maintain_order=True
                ).collect().lazy()
            else:
                result = pl.LazyFrame([{}])
    except ColumnNotFoundError as e:
        raise KeyError(str(e)) from e

    # Apply dependent expressions (reference columns created in this call)
    if dependent:
        dep_exprs = [expr.alias(key) for key, expr in dependent.items()]
        result = result.with_columns(dep_exprs)

    if deferred:
        result = result.collect() if not isinstance(result, pl.DataFrame) else result
        for col_name, dfs in deferred.items():
            result = result.with_columns(
                pl.Series(col_name, dfs, dtype=pl.Object)
            )

    # Drop temporary columns (names beginning with _)
    _tmp_cols = [c for c in result.columns if c.startswith("_") and c not in gvars]
    if _tmp_cols:
        result = result.drop(_tmp_cols)

    return reconstruct_tibble(result, _data)
