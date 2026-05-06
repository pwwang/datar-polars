"""Test configuration and helpers for datar-polars"""

import pytest
import polars as pl

from datar import options

options(backends=["polars"], allow_conflict_names=True)


def pytest_sessionstart(session):
    """Ensure plugin hooks are loaded (bypass simplug cache)."""
    from datar.core import load_plugins  # noqa: F401
    # Force re-execute base_api to clear cached failure from trig.py fix
    from datar_polars.plugin import base_api, dplyr_api
    dplyr_api()
    base_api()


SENTINEL = 85258525.85258525


def assert_df_equal(left, right, approx=False):
    """Assert two polars DataFrames have the same shape, columns, and values."""
    assert left.shape == right.shape, f"Shapes differ: {left.shape} != {right.shape}"
    assert list(left.collect_schema().names()) == list(right.collect_schema().names()), (
        f"Columns differ: {list(left.collect_schema().names())} != {list(right.collect_schema().names())}"
    )
    for col in left.collect_schema().names():
        lv = left.get_column(col).to_list()
        rv = right.get_column(col).to_list()
        if approx:
            assert lv == pytest.approx(rv), f"Column {col!r} differs: {lv} != {rv}"
        else:
            assert lv == rv, f"Column {col!r} differs: {lv} != {rv}"


def assert_equal(x, y, approx=False):
    """Assert scalar equality, handling None/NaN."""
    import numpy as np

    if x is None and y is None:
        return
    if isinstance(x, float) and np.isnan(x) and isinstance(y, float) and np.isnan(y):
        return
    if approx is True:
        assert x == pytest.approx(y), f"{x} != {y}"
    elif approx:
        assert x == pytest.approx(y, rel=approx), f"{x} != {y}"
    else:
        assert x == y, f"{x} != {y}"


def assert_iterable_equal(x, y, na=SENTINEL, approx=False):
    """Assert two iterables have equal values, handling None/NaN."""
    import numpy as np

    x = list(x)
    y = list(y)
    assert len(x) == len(y), f"Lengths differ: {len(x)} != {len(y)}"
    for i, (xv, yv) in enumerate(zip(x, y)):
        if xv is None and yv is None:
            continue
        # Treat None and NaN as equivalent (both represent NA/missing)
        if xv is None and isinstance(yv, float) and np.isnan(yv):
            continue
        if yv is None and isinstance(xv, float) and np.isnan(xv):
            continue
        if isinstance(xv, float) and np.isnan(xv) and isinstance(yv, float) and np.isnan(yv):
            continue
        if approx is True:
            assert xv == pytest.approx(yv), f"[{i}] {xv} != {yv}"
        elif approx:
            assert xv == pytest.approx(yv, rel=approx), f"[{i}] {xv} != {yv}"
        else:
            assert xv == yv, f"[{i}] {xv} != {yv}"


def is_installed(pkg):
    """Check if a package is importable."""
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def pl_data():
    """Create a set of common polars test data."""
    from collections import namedtuple

    from datar_polars.tibble import as_tibble

    out = namedtuple(
        "pl_data",
        "scalar df gf tibble",
    )
    out.scalar = 1
    out.df = as_tibble(pl.DataFrame({"x": [1, 2, 2, 3]}))
    out.gf = as_tibble(pl.DataFrame({"x": [1, 2, 2, 3], "g": [1, 1, 2, 2]}))
    out.tibble = out.df
    return out
