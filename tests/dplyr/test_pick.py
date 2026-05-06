import pytest
import polars as pl

from datar import f
from datar.dplyr import pick, mutate, summarise, starts_with
from datar.tibble import tibble


def test_pick_columns_from_data():
    df = tibble(x1=1, y=2, x2=3, z=4)
    out = df >> mutate(sel=pick(f.z, starts_with("x")))
    collected = out.collect()
    sel_cols = list(collected["sel"].columns)
    assert "z" in sel_cols
    assert "x1" in sel_cols


def test_must_supply_one_selector():
    df = tibble(x=[2, 3, 4])
    with pytest.raises(ValueError):
        df >> mutate(y=pick())
