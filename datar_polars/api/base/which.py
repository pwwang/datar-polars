"""Which: find indices of TRUE elements"""
import polars as pl
from pipda import register_func
from datar.apis.base import which, which_max, which_min
from ...contexts import Context

@which.register(pl.Expr, context=Context.EVAL, backend="polars")
def _which_expr(x):
    return pl.when(x).then(pl.int_range(pl.len()) + 1).otherwise(None)

@which.register(object, backend="polars")
def _which_obj(x):
    if isinstance(x, pl.Series):
        if x.dtype != pl.Boolean:
            x = x.cast(pl.Boolean)
        return (x.arg_true() + 1).to_list()
    result = []
    for i, v in enumerate(x):
        if v:
            result.append(i + 1)
    return result

@which_max.register(pl.Expr, context=Context.EVAL, backend="polars")
def _which_max_expr(x):
    return x.arg_max() + 1

@which_max.register(object, backend="polars")
def _which_max_obj(x):
    if isinstance(x, pl.Series):
        return x.arg_max() + 1
    return x.index(max(x)) + 1

@which_min.register(pl.Expr, context=Context.EVAL, backend="polars")
def _which_min_expr(x):
    return x.arg_min() + 1

@which_min.register(object, backend="polars")
def _which_min_obj(x):
    if isinstance(x, pl.Series):
        return x.arg_min() + 1
    return x.index(min(x)) + 1
