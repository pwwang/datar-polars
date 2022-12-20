import pytest

import polars as pl
from datar_polars.factor import Factor


def test_factor():
    ft = Factor([1, 2, 3, 4, 5])
    assert isinstance(ft, Factor)
    assert ft.dtype == pl.Categorical
    ft2 = ft.copy()
    assert isinstance(ft2, Factor)
