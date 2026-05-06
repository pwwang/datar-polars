import pytest
import polars as pl

from datar.base import (
    arg,
    conj,
    mod,
    re_,
    im,
)
from ..conftest import assert_equal, assert_iterable_equal


def _isscalar(x):
    if isinstance(x, (str, bytes)):
        return True
    try:
        iter(x)
    except TypeError:
        return True
    return False


@pytest.mark.parametrize("fn, x, expected", [
    # scalar inputs
    (arg, 1j, 1.5707963267948966),
    (conj, 1j, -1j),
    (mod, 1j, 1.0),
    (re_, 1j, 0.0),
    (im, 1j, 1.0),
    # list inputs
    (arg, [1j], [1.5707963267948966]),
    (conj, [1j], [-1j]),
    (mod, [1j], [1.0]),
    (re_, [1j], [0.0]),
    (im, [1j], [1.0]),
    # multi-element lists
    (arg, [1j, 2j], [1.5707963267948966, 1.5707963267948966]),
    (conj, [1j, 2j], [-1j, -2j]),
    (mod, [1j, 2j], [1.0, 2.0]),
    (re_, [1j, 2j], [0.0, 0.0]),
    (im, [1j, 2j], [1.0, 2.0]),
    # pl.Series inputs
    (arg, pl.Series([1j]), [1.5707963267948966]),
    (conj, pl.Series([1j]), [-1j]),
    (mod, pl.Series([1j]), [1.0]),
    (re_, pl.Series([1j]), [0.0]),
    (im, pl.Series([1j]), [1.0]),
    # multi-element pl.Series
    (arg, pl.Series([1j, 2j]), [1.5707963267948966, 1.5707963267948966]),
    (conj, pl.Series([1j, 2j]), [-1j, -2j]),
    (mod, pl.Series([1j, 2j]), [1.0, 2.0]),
    (re_, pl.Series([1j, 2j]), [0.0, 0.0]),
    (im, pl.Series([1j, 2j]), [1.0, 2.0]),
])
def test_complex(fn, x, expected):
    if _isscalar(expected):
        assert_equal(fn(x), expected, approx=True)
    else:
        assert_iterable_equal(fn(x), expected, approx=True)
