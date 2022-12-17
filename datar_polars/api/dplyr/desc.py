"""Provides desc"""
from datar.apis.dplyr import desc
from datar_numpy.utils import make_array

from polars import Categorical, Series, Int64, Expr


@desc.register(object, backend="polars")
def _desc_obj(x):
    try:
        out = -make_array(x)
    except (ValueError, TypeError):
        cat = Series(x).cast(Categorical)
        out = desc.dispatch(Series, backend="polars")(cat)

    return out


@desc.register(Expr, backend="polars")
def _desc_expr(x: Expr):
    return x.arg_sort(reverse=True)


@desc.register(Series, backend="polars")
def _desc_scat(x: Series):
    if x.dtype is not Categorical:
        return desc.dispatch(object, backend="polars")(x.to_numpy())

    return -x.to_physical().cast(Int64)
