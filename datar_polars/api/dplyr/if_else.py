"""Vectorised if and multiple if-else

https://github.com/tidyverse/dplyr/blob/master/R/if_else.R
https://github.com/tidyverse/dplyr/blob/master/R/case_when.R
"""

from __future__ import annotations

from typing import Any

import polars as pl
from datar.apis.dplyr import if_else, case_when, case_match

from ...common import is_iterable, to_series
from ...contexts import Context


# ---- if_else ------------------------------------------------------------


@if_else.register(pl.Expr, context=Context.EVAL, backend="polars")
def _if_else_expr(
    condition: pl.Expr,
    true,
    false,
    missing=None,
) -> pl.Expr:
    """Vectorised if-else for lazy Expr inputs."""
    cond = condition.fill_null(False).cast(pl.Boolean)
    true_val = pl.lit(true) if not isinstance(true, pl.Expr) else true
    false_val = pl.lit(false) if not isinstance(false, pl.Expr) else false
    result = pl.when(cond).then(true_val).otherwise(false_val)
    if missing is not None:
        missing_val = pl.lit(missing) if not isinstance(missing, pl.Expr) else missing
        result = pl.when(condition.is_null()).then(missing_val).otherwise(result)
    return result


@if_else.register(object, context=Context.EVAL, backend="polars")
def _if_else(condition, true, false, missing=None):
    """Vectorised if-else. Returns scalar for scalar inputs, Series for vectors."""
    has_expr = (
        isinstance(condition, pl.Expr)
        or isinstance(true, pl.Expr)
        or isinstance(false, pl.Expr)
    )

    # Expr path — return Expr for lazy evaluation
    if has_expr:
        cond = pl.lit(condition) if not isinstance(condition, pl.Expr) else condition
        if isinstance(cond, pl.Series):
            cond = pl.lit(cond)
        true_val = pl.lit(true) if not isinstance(true, pl.Expr) else true
        false_val = pl.lit(false) if not isinstance(false, pl.Expr) else false
        result = pl.when(cond).then(true_val).otherwise(false_val)
        if missing is not None:
            missing_val = (
                pl.lit(missing) if not isinstance(missing, pl.Expr) else missing
            )
            result = pl.when(cond.is_null()).then(missing_val).otherwise(result)
        return result

    # Non-Expr path — return Series (or scalar for pure scalar)
    has_vector = (
        is_iterable(condition)
        or is_iterable(true)
        or is_iterable(false)
        or isinstance(condition, pl.Series)
        or isinstance(true, pl.Series)
        or isinstance(false, pl.Series)
    )

    if not has_vector:
        if missing is not None and condition is None:
            return missing
        return true if condition else false

    # Determine length from the first vector input and validate sizes
    length = None
    for name, v in [("condition", condition), ("true", true), ("false", false)]:
        vlen = None
        if isinstance(v, pl.Series):
            vlen = len(v)
        elif is_iterable(v):
            vlen = len(list(v))
        if vlen is not None:
            if length is None:
                length = vlen
            elif vlen != length:
                raise ValueError(
                    f"`{name}` has size {vlen}, "
                    f"but the condition has size {length}."
                )

    cond_s = to_series(condition, length)
    true_s = to_series(true, length)
    false_s = to_series(false, length)

    cond_bool = cond_s.cast(pl.Boolean)

    if missing is not None:
        missing_s = to_series(missing, length)
        result = pl.select(
            pl.when(cond_s.is_null())
            .then(missing_s)
            .otherwise(pl.when(cond_bool).then(true_s).otherwise(false_s))
        ).to_series()
    else:
        result = pl.select(
            pl.when(cond_bool).then(true_s).otherwise(false_s)
        ).to_series()
    return result


# ---- case_when ----------------------------------------------------------


@case_when.register(pl.Expr, context=Context.EVAL, backend="polars")
def _case_when_expr(*args: Any, _default: Any = None) -> pl.Expr:
    """Lazy case_when returning pl.Expr."""
    if not args:
        raise TypeError("No cases provided.")
    if isinstance(args[0], (str, bytes)):
        raise TypeError("`case_when` expects conditions and values, not a string.")

    cases = []
    default = _default
    i = 0
    has_tuple = False
    while i < len(args):
        arg = args[i]
        if isinstance(arg, tuple) and len(arg) == 2:
            cases.append(arg)
            i += 1
            has_tuple = True
        elif i + 1 < len(args):
            next_arg = args[i + 1]
            if isinstance(next_arg, tuple) and len(next_arg) == 2:
                raise ValueError("Case-value not paired.")
            else:
                cases.append((arg, next_arg))
                i += 2
        else:
            if not has_tuple:
                raise ValueError("Case-value not paired.")
            default = arg
            i += 1

    if not cases:
        raise ValueError("No cases provided.")

    result = pl.lit(default) if default is not None else pl.lit(None)
    for cond_i, val_i in reversed(cases):
        cond_expr = cond_i if isinstance(cond_i, pl.Expr) else pl.lit(cond_i)
        val_expr = val_i if isinstance(val_i, pl.Expr) else pl.lit(val_i)
        result = pl.when(cond_expr).then(val_expr).otherwise(result)
    return result


@case_when.register(object, context=Context.EVAL, backend="polars")
def _case_when(*args: Any, _default: Any = None) -> Any:
    """Vectorised case_when. Supports both tuple and flat calling conventions.

    Tuple form: case_when((cond, val), (cond, val), ..., default_scalar)
    Flat form:  case_when(cond, val, cond, val, ..., default)
    """
    if not args:
        raise TypeError("No cases provided.")
    if isinstance(args[0], (str, bytes)):
        raise TypeError("`case_when` expects conditions and values, not a string.")

    cases = []
    default = _default
    i = 0
    has_tuple = False
    while i < len(args):
        arg = args[i]
        if isinstance(arg, tuple) and len(arg) == 2:
            cases.append(arg)
            i += 1
            has_tuple = True
        elif i + 1 < len(args):
            next_arg = args[i + 1]
            if isinstance(next_arg, tuple) and len(next_arg) == 2:
                raise ValueError("Case-value not paired.")
            else:
                cases.append((arg, next_arg))
                i += 2
        else:
            if not has_tuple:
                raise ValueError("Case-value not paired.")
            default = arg
            i += 1

    if not cases:
        raise ValueError("No cases provided.")

    has_expr = False
    length = None
    for cond_i, val_i in cases:
        if isinstance(cond_i, pl.Expr) or isinstance(val_i, pl.Expr):
            has_expr = True
        if isinstance(cond_i, pl.Series):
            length = max(length, len(cond_i)) if length else len(cond_i)
        elif is_iterable(cond_i):
            length = max(length, len(list(cond_i))) if length else len(list(cond_i))
        if isinstance(val_i, pl.Series):
            length = max(length, len(val_i)) if length else len(val_i)
        elif is_iterable(val_i):
            length = max(length, len(list(val_i))) if length else len(list(val_i))

    if has_expr:
        default_expr = pl.lit(default) if default is not None else pl.lit(None)
        result = default_expr
        for cond_i, val_i in reversed(cases):
            cond_expr = cond_i if isinstance(cond_i, pl.Expr) else pl.lit(cond_i)
            val_expr = val_i if isinstance(val_i, pl.Expr) else pl.lit(val_i)
            result = pl.when(cond_expr).then(val_expr).otherwise(result)
        return result

    # Non-Expr path: build using Series — iterate reversed so first case wins
    result = None
    for cond_i, val_i in reversed(cases):
        cond_s = to_series(cond_i, length).cast(pl.Boolean)
        val_s = to_series(val_i, length)
        if length is None:
            length = len(cond_s)
        if result is None:
            result = pl.Series([None] * length)
        result = pl.select(
            pl.when(cond_s).then(val_s).otherwise(result)
        ).to_series()

    if result is None:
        raise ValueError("No valid conditions provided to case_when.")

    if default is not None:
        default_s = to_series(default, length)
        result = result.fill_null(default_s)

    return result


# ---- case_match ---------------------------------------------------------


@case_match.register(pl.Expr, context=Context.EVAL, backend="polars")
def _case_match_expr(
    _x: pl.Expr, *args: Any, _default=None, _dtypes=None
) -> pl.Expr:
    if len(args) % 2 != 0 or len(args) == 0:
        raise ValueError("condition-value not paired.")

    result = None
    for i in range(0, len(args), 2):
        match_vals = args[i]
        replacement = args[i + 1]

        if match_vals is None:
            cond = _x.is_null()
        else:
            match_list = (
                [match_vals]
                if not hasattr(match_vals, "__iter__")
                or isinstance(match_vals, (str, bytes))
                else list(match_vals)
            )
            cond = _x.is_in(match_list) | _x.is_null()

        repl_expr = (
            replacement
            if isinstance(replacement, pl.Expr)
            else pl.lit(replacement)
        )
        if result is None:
            result = pl.when(cond).then(repl_expr)
        else:
            result = pl.when(cond).then(repl_expr).otherwise(result)

    if result is None:
        raise ValueError("condition-value not paired.")

    if _default is not None:
        default_expr = (
            _default if isinstance(_default, pl.Expr) else pl.lit(_default)
        )
        result = result.otherwise(default_expr)
    else:
        result = result.otherwise(None)

    if _dtypes is not None:
        result = result.cast(_dtypes)

    return result


@case_match.register(object, context=Context.EVAL, backend="polars")
def _case_match_obj(
    _x: Any, *args: Any, _default=None, _dtypes=None
) -> Any:
    if len(args) % 2 != 0 or len(args) == 0:
        raise ValueError("condition-value not paired.")

    x_series = to_series(_x)
    length = len(x_series)
    result = pl.Series([None] * length)

    for i in range(0, len(args), 2):
        match_vals = args[i]
        replacement = args[i + 1]

        if match_vals is None:
            mask = x_series.is_null()
        else:
            match_list = (
                [match_vals]
                if not hasattr(match_vals, "__iter__")
                or isinstance(match_vals, (str, bytes))
                else list(match_vals)
            )
            mask = x_series.is_in(match_list) | x_series.is_null()

        repl_series = to_series(replacement, length)
        result = pl.select(
            pl.when(mask).then(repl_series).otherwise(result)
        ).to_series()

    if _default is not None:
        default_series = to_series(_default, length)
        result = result.fill_null(default_series)

    if _dtypes is not None:
        result = result.cast(_dtypes)

    return result
