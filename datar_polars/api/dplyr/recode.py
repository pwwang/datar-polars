"""Recode values.

https://github.com/tidyverse/dplyr/blob/master/R/recode.R
"""

from __future__ import annotations

import math
from typing import Any

from datar.core.utils import logger
from datar.apis.dplyr import recode, recode_factor

from ...polars import Series


def _args_to_recodings(*args: Any, **kwargs: Any) -> dict:
    """Convert arguments to a replacement dictionary.

    Arguments can be provided as:
        (1, "a", 2, "b") -> {0: 1, 1: "a", 2: 2, 3: "b"}
    Dict args and kwargs are added directly:
        ({1: "a"}, x="X") -> {1: "a", "x": "X"}
    """
    values = {}
    i = 0
    for i, arg in enumerate(args):
        if isinstance(arg, dict):
            values.update(arg)
        else:
            values[i] = arg

    values.update(kwargs)
    return values


# ── recode ──────────────────────────────────────────────────────────────────
NODEFAULT = object()


@recode.register(object, backend="polars")
def _recode_obj(
    _x: Any,
    *args: Any,
    _default: Any = NODEFAULT,
    _missing: Any = None,
    **kwargs: Any,
) -> Any:
    """Recode a vector (object dispatch)."""
    return recode.dispatch(Series)(
        Series(_x, strict=False),
        *args,
        _default=_default,
        _missing=_missing,
        **kwargs,
    )


@recode.register(Series, backend="polars")
def _recode_series(
    _x: Series,
    *args: Any,
    _default: Any = NODEFAULT,
    _missing: Any = None,
    **kwargs: Any,
) -> Series:
    """Recode a polars Series."""
    values = _args_to_recodings(*args, **kwargs)

    if isinstance(_default, float) and math.isnan(_default):
        _default = None

    if not values and _default is NODEFAULT and _missing is None:
        raise ValueError("No replacements provided.")

    import polars as pl
    # Enum/Categorical columns can't hold arbitrary replacement values; cast to String first
    if _x.dtype in (pl.Categorical, pl.Enum) or isinstance(_x.dtype, pl.Enum):
        _x = _x.cast(pl.String)

    # if _x is a float-like, and any of the values (replacements, default, missing) has non-numeric type
    # we need to convert NAs (nans) to None, otherwise they will be retained as "NaN" strings in the output
    if _x.dtype.is_float():
        if any(
            not pl.Series([v]).dtype.is_numeric()
            for v in list(values.values()) + [_default, _missing]
            if v is not None and v is not NODEFAULT
        ):
            _x = _x.fill_nan(None)

    null_replacement = _missing if _missing is not None else values.pop(None, None)

    x_name = _x.name or "x"
    result = _x.to_frame(x_name)

    # Build a single chained when/then/otherwise expression
    expr = None

    # Missing/null replacement (checked first, before value comparisons)
    if null_replacement is not None:
        expr = pl.when(pl.col(x_name).is_null()).then(pl.lit(null_replacement))

    # Explicit replacements
    for old_val, new_val in values.items():
        cond = pl.col(x_name) == old_val
        if expr is None:
            expr = pl.when(cond).then(pl.lit(new_val))
        else:
            expr = expr.when(cond).then(pl.lit(new_val))

    # Default: values not matching any replacement and not null
    if _default is not NODEFAULT:
        old_keys = [k for k in values.keys() if k is not None]
        if old_keys:
            not_replaced = ~pl.col(x_name).is_in(old_keys) & pl.col(x_name).is_not_null()
        else:
            not_replaced = pl.col(x_name).is_not_null()
        if expr is None:
            expr = pl.when(not_replaced).then(pl.lit(_default))
        else:
            expr = expr.when(not_replaced).then(pl.lit(_default))

    if expr is not None:
        result = result.with_columns(expr.otherwise(pl.col(x_name)).alias(x_name))

    return result.get_column(x_name)


# ── recode_factor ───────────────────────────────────────────────────────────


@recode_factor.register(object, backend="polars")
def _recode_factor_obj(
    _x: Any,
    *args: Any,
    _default: Any = NODEFAULT,
    _missing: Any = None,
    _ordered: bool = False,
    **kwargs: Any,
) -> Any:
    """Recode a factor (object dispatch)."""
    return recode_factor.dispatch(Series)(
        Series(_x, strict=False),
        *args,
        _default=_default,
        _missing=_missing,
        _ordered=_ordered,
        **kwargs,
    )


@recode_factor.register(Series, backend="polars")
def _recode_factor_series(
    _x: Series,
    *args: Any,
    _default: Any = NODEFAULT,
    _missing: Any = None,
    _ordered: bool = False,
    **kwargs: Any,
) -> Series:
    """Recode a factor (polars Series)."""
    import polars as pl

    values = _args_to_recodings(*args, **kwargs)

    if not values:
        raise ValueError("No replacements provided.")

    # recode_factor just recodes and ensures factor-like output
    recoded = recode.dispatch(Series)(
        _x, *args, _default=_default, _missing=_missing, **kwargs
    )

    # Derive the new level order from the original levels/values
    if isinstance(_x.dtype, pl.Enum):
        orig_levels = list(_x.dtype.categories)
    elif isinstance(_x.dtype, pl.Categorical):
        orig_levels = _x.cast(pl.String).drop_nulls().unique(maintain_order=True).to_list()
    else:
        # Keep native types so integer/float keys in `values` match correctly
        orig_levels = list(dict.fromkeys(_x.drop_nulls().to_list()))

    # Map each original level through the replacement dict
    new_levels = []
    seen = set()
    for lvl in orig_levels:
        if lvl in values:
            new_lvl = str(values[lvl])
        elif _default is not NODEFAULT:
            new_lvl = str(_default)
        else:
            new_lvl = str(lvl)
        if new_lvl not in seen:
            seen.add(new_lvl)
            new_levels.append(new_lvl)

    # If _missing is set and the series has nulls or NaNs, add its level too
    has_missing = _x.null_count() > 0 or (
        _x.dtype.is_float() and _x.is_nan().any()
    )
    if _missing is not None and has_missing:
        m = str(_missing)
        if m not in seen:
            new_levels.append(m)

    return recoded.cast(pl.String).cast(pl.Enum(new_levels))
