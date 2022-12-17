"""Utilities for datar_polars"""
from __future__ import annotations

from typing import Any, Sequence
from functools import singledispatch

import numpy as np
from polars import DataFrame, Series, Expr, lit
from pipda import Expression, evaluate_expr
from pipda.context import ContextType
from datar.apis.base import (
    is_integer as _is_integer,
    is_logical as _is_logical,
    intersect as _intersect,
    setdiff as _setdiff,
    union as _union,
    unique as _unique,
)
from datar_numpy.utils import is_scalar  # noqa: F401
from datar_numpy.api import asis as _  # noqa: F401
from datar_numpy.api import sets as _  # noqa: F401, F811


class ExpressionWrapper:
    """A wrapper around an expression to bypass evaluation"""

    __slots__ = ("expr",)

    def __init__(self, expr: Expression) -> None:
        self.expr = expr

    def __str__(self) -> str:
        return str(self.expr)

    def _pipda_eval(self, data: Any, context: ContextType = None) -> Any:
        return evaluate_expr(self.expr, data, context)


def is_null(x: Any) -> bool | Sequence[bool]:
    """Is x a null value?

    Args:
        x: The object to check

    Returns:
        True if x is a null value, False otherwise
    """
    if isinstance(x, float):
        return np.isnan(x)

    if is_scalar(x):
        return x is None

    return [is_null(i) for i in x]


def vars_select(
    all_columns: Sequence[str],
    *columns: int | str,
    raise_nonexists: bool = True,
):
    """Select columns, using indexes or names

    Args:
        all_columns: The column pool to select
        *columns: arguments to select from the pool
        raise_nonexist: Whether raise exception when column not exists
            in the pool

    Returns:
        The selected columns

    Raises:
        KeyError: When the column does not exist in the pool
            and raise_nonexists is True.
    """
    from .collections import Collection
    # convert all to names
    columns = [all_columns[i] if isinstance(i, int) else i for i in columns]

    selected = Collection(*columns, pool=all_columns)  # indexes
    if raise_nonexists and selected.unmatched and selected.unmatched != {None}:
        raise KeyError(f"Columns `{selected.unmatched}` do not exist.")

    return unique(selected).astype(int)


def is_integer(x: Any) -> bool:
    return _is_integer(x, __ast_fallback="normal", __backend="numpy")


def is_logical(x: Any) -> bool:
    return _is_logical(x, __ast_fallback="normal", __backend="numpy")


def unique(x: Any) -> np.ndarray:
    return _unique(x, __ast_fallback="normal", __backend="numpy")


def setdiff(x: Any, y: Any) -> Any:
    return _setdiff(x, y, __ast_fallback="normal", __backend="numpy")


def union(x: Any, y: Any) -> Any:
    return _union(x, y, __ast_fallback="normal", __backend="numpy")


def intersect(x: Any, y: Any) -> Any:
    return _intersect(x, y, __ast_fallback="normal", __backend="numpy")


def to_expr(x: Any, name: str = None) -> Expr | Series:
    """Convert anything to polars Expr or Series so that it can be used in
    polars operations.
    """
    if isinstance(x, (Series, Expr)):
        pass
    elif is_scalar(x):
        x = lit(x)
    else:
        x = Series(x)
    return x if name is None else x.alias(name)


@singledispatch
def name_of(x):
    return str(x)


@name_of.register(Series)
def _name_of_series(x):
    return x.name


@name_of.register(DataFrame)
def _name_of_df(x):
    return None


@name_of.register(Expr)
def _name_of_expr(x):
    return x.meta.output_name()
