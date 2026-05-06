"""Subset rows using column values

See source https://github.com/tidyverse/dplyr/blob/master/R/filter.R
"""

from __future__ import annotations

from typing import Any

import polars as pl

from datar.core.utils import logger
from datar.dplyr import filter_

from ...contexts import Context
from ...tibble import (
    Tibble,
    LazyTibble,
    reconstruct_tibble,
    to_lazy,
    to_eager,
)


def _get_gvars(data) -> list:
    """Get grouping variable names from _datar metadata."""
    if hasattr(data, "_datar") and data._datar.get("groups") is not None:
        return list(data._datar["groups"])
    return []


@filter_.register(pl.LazyFrame, context=Context.EVAL, backend="polars")
def _filter_lazy(
    _data: pl.LazyFrame,
    *conditions: Any,
    _preserve: bool = False,
) -> Tibble | LazyTibble:
    """Filter rows of a LazyFrame/Tibble using lazy polars expressions."""
    if _preserve:
        logger.warning("`filter()` doesn't support `_preserve` argument yet.")

    if not conditions:
        return _data.clone()

    _nrow: int | None = None

    def _get_nrow() -> int:
        nonlocal _nrow
        if _nrow is None:
            _nrow = _data.select(pl.len()).collect().item()
        return _nrow

    combined: pl.Expr | None = None
    for cond in conditions:
        expr = _as_bool_expr(cond, _get_nrow)
        if combined is None:
            combined = expr
        else:
            combined = combined & expr

    if combined is None:
        return _data.clone()

    if _is_trivially_false(combined):
        result = _data.clear()
        return reconstruct_tibble(result, _data)

    gvars = _get_gvars(_data)
    if gvars:
        combined = combined.over(gvars)

    out = _data.filter(combined)
    return reconstruct_tibble(out, _data)


@filter_.register(pl.DataFrame, context=Context.EVAL, backend="polars")
def _filter_eager(
    _data: pl.DataFrame,
    *conditions: Any,
    _preserve: bool = False,
) -> Tibble:
    """Filter rows of a DataFrame/Tibble by delegating to lazy."""
    return to_eager(
        _filter_lazy(
            to_lazy(_data), *conditions, _preserve=_preserve
        )
    )


def _as_bool_expr(
    cond: Any, get_nrow: "Callable[[], int] | None" = None
) -> pl.Expr:
    """Normalise a filter condition to a polars Expr."""
    if isinstance(cond, pl.Expr):
        return cond

    if isinstance(cond, pl.Series):
        if get_nrow is not None and len(cond) not in (1, get_nrow()):
            raise ValueError(
                f"`filter()` condition has {len(cond)} rows, "
                f"but data has {get_nrow()} rows."
            )
        return pl.lit(cond)

    if isinstance(cond, (bool, int, float)):
        return pl.lit(bool(cond))

    if hasattr(cond, "__iter__") and not isinstance(cond, str):
        seq = list(cond)
        if len(seq) == 1:
            return pl.lit(bool(seq[0]))
        if get_nrow is not None and len(seq) != get_nrow():
            raise ValueError(
                f"`filter()` condition has {len(seq)} rows, "
                f"but data has {get_nrow()} rows."
            )
        return pl.lit(seq)

    # scalar fallback — wrap in pl.lit
    return pl.lit(bool(cond))


def _is_trivially_false(expr: pl.Expr) -> bool:
    """Detect trivially-false expressions like pl.lit(False).

    For complex expressions we return False (don't know at plan time).
    """
    # Simple heuristic: pl.lit(False) is trivially false
    try:
        meta = expr.meta
        if hasattr(meta, "root_names") and not meta.root_names():
            # Expression with no column references — likely a literal
            return False  # Can't determine without executing
    except Exception:
        pass
    return False
