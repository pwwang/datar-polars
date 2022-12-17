
import pytest
import numpy as np

from datar import options
# from datar.core import plugin  # noqa: F401
options(
    import_names_conflict="silent",
    backends=["numpy", "polars"],
)


# def pytest_addoption(parser):
#     parser.addoption("--modin", action="store_true")


# def pytest_sessionstart(session):
#     set_seed(8888)


SENTINEL = 85258525.85258525


def _isna(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return False
    try:
        return np.isnan(x)
    except (ValueError, TypeError):
        return False


def assert_iterable_equal(x, y, na=SENTINEL, approx=False):
    from datar_polars.utils import is_null

    x = [na if is_null(elt) else elt for elt in x]
    y = [na if is_null(elt) else elt for elt in y]
    if approx is True:
        x = pytest.approx(x)
    elif approx:
        x = pytest.approx(x, rel=approx)
    assert x == y, f"{x} != {y}"


def assert_factor_equal(x, y, na=8525.8525, approx=False):
    xlevs = x.categories
    ylevs = y.categories
    assert_iterable_equal(x, y, na=na, approx=approx)
    assert_iterable_equal(xlevs, ylevs, na=na, approx=approx)


def assert_(x):
    assert x, f"{x} is not True"


def assert_not(x):
    assert not x, f"{x} is not False"


# pytest modifies node for assert
def assert_equal(x, y, approx=False):
    if _isna(x) and _isna(y):
        return
    if approx is True:
        x = pytest.approx(x)
    elif approx:
        x = pytest.approx(x, rel=approx)
    assert x == y, f"{x} != {y}"


def is_installed(pkg):
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False
