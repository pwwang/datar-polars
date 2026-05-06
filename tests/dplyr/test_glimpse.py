import pytest
import polars as pl

from datar.dplyr import glimpse
from datar.tibble import tibble


def test_glimpse_returns_object():
    df = tibble(x=list(range(10)), y=[str(i) for i in range(10)])
    g = glimpse(df)
    out = str(g)
    assert len(out) > 0


def test_glimpse_html():
    df = tibble(x=list(range(20)), y=[str(i) for i in range(20)])
    g = glimpse(df, 100)
    out = g._repr_html_()
    assert "<table>" in out
