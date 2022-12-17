from polars import Series
from datar.apis.base import (  # noqa: F401
    c_,
)
from datar_numpy.api import seq as _  # noqa: F401

from ...collections import Collection


# Define different function so that it has higher priority
@c_.register(Series, backend="polars")
def _c_series(*args):
    return Collection(*args)
