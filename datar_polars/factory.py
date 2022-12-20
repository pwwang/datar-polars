from typing import Callable

import polars as pl


def func_bootstrap(func: Callable, impl: Callable = None) -> Callable:
    """Bootstrap a registered function to make it work with polars.Expr"""

    if impl is None:
        return lambda fun: func_bootstrap(func, fun)

    @func.register(pl.Expr, backend="polars")
    def _bootstrap_polars(x: pl.Expr, *args, **kwargs) -> pl.Expr:
        return x.map(lambda s: impl(s, *args, **kwargs))

    return impl
