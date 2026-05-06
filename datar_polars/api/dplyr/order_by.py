"""Helper functions for ordering window function output.

https://github.com/tidyverse/dplyr/blob/master/R/order-by.R
"""

from __future__ import annotations

from typing import Any, Callable

import polars as pl
from pipda import FunctionCall
from pipda.reference import ReferenceAttr, ReferenceItem

from datar.apis.dplyr import order_by, with_order

from ...polars import Series


def _resolve_arg(arg: Any) -> Any:
    """Resolve ReferenceAttr/ReferenceItem to pl.col(...) Expr."""
    if isinstance(arg, (ReferenceAttr, ReferenceItem)):
        return pl.col(arg._pipda_ref)
    return arg


@order_by.register(object, backend="polars")
def _order_by(order: Any, call: Any) -> Any:
    """Order the data by the given order within a verb.

    This function is designed to be used as an argument to a verb.
    Args:
        order: The ordering values.
        call: A pipda FunctionCall expression.

    Returns:
        Modified FunctionCall with reordered data.
    """
    if not isinstance(call, FunctionCall) or not call._pipda_args:
        raise ValueError(
            "In `order_by()`: `call` must be a registered "
            f"function call with data, not `{type(call).__name__}`."
        )

    # Build an ordering index
    if isinstance(order, pl.Expr):
        order_vals = order
    elif isinstance(order, Series):
        order_vals = order
    else:
        order_vals = pl.Series(order)

    # Get the first argument (the data)
    x = call._pipda_args[0]

    # For Expr or unevaluated pipda references: build a lazily-evaluated
    # polars expression that sorts, applies the function, then unsorts.
    if isinstance(x, (pl.Expr, ReferenceAttr, ReferenceItem)):
        if isinstance(order_vals, pl.Expr):
            return call  # Can't eagerly compute sort indices from Expr

        order_idx = order_vals.arg_sort()
        inv_idx = order_idx.arg_sort()

        # Resolve reference to pl.Expr
        x_expr = _resolve_arg(x)

        # Sort input, substitute into call, evaluate, unsort result
        call._pipda_args = (x_expr.gather(order_idx), *call._pipda_args[1:])
        result = call._pipda_func(
            *(_resolve_arg(a) for a in call._pipda_args),
            **{k: _resolve_arg(v) for k, v in call._pipda_kwargs.items()},
        )

        if isinstance(result, pl.Expr):
            return result.gather(inv_idx)
        return result

    if isinstance(x, Series):
        order_idx = order_vals.arg_sort()
        x = x.gather(order_idx)
        call._pipda_args = (x, *call._pipda_args[1:])
        result = call
        # Reorder result back
        if isinstance(result, Series):
            inv_idx = order_idx.arg_sort()
            return result.gather(inv_idx)
        return result

    # Generic case
    return call


@with_order.register(object, backend="polars")
def _with_order(
    order: Any,
    func: Callable,
    x: Any,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Control argument and result ordering of a window function.

    Args:
        order: The ordering values.
        func: The window function.
        x: The first argument for the function.
        *args: Additional arguments.
        **kwargs: Keyword arguments.

    Returns:
        The ordered result.
    """
    if isinstance(order, (pl.Expr, Series)):
        ord_series = order if isinstance(order, Series) else order
    else:
        ord_series = pl.Series(order)

    # Build sort index
    order_idx = ord_series.arg_sort()

    # Reorder input
    if isinstance(x, Series):
        x_ordered = x.gather(order_idx)
    elif isinstance(x, pl.Expr):
        # Can't easily reorder lazy expressions
        return func(x, *args, **kwargs)
    else:
        import numpy as np

        x_arr = np.asarray(list(x))
        x_ordered = x_arr[order_idx.to_numpy()]

    # Apply function
    result = func(x_ordered, *args, **kwargs)

    # Reorder result back
    inv_idx = order_idx.arg_sort()
    if isinstance(result, Series):
        return result.gather(inv_idx)
    if isinstance(result, pl.Expr):
        return result
    import numpy as np

    result_arr = np.asarray(list(result))
    return result_arr[inv_idx.to_numpy()]
