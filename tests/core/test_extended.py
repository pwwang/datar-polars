import pytest  # noqa: F401

import polars as pl
from polars.testing import assert_frame_equal
from datar_polars.extended import Grouper, GFGrouper, DFGrouper, RFGrouper
from datar_polars.tibble import Tibble, TibbleGrouped, TibbleRowwise

from ..conftest import assert_iterable_equal


def test_gfgrouper():
    df = pl.DataFrame({"x": [2, 2, 3], "y": [3, 2, 1]})
    grouper = GFGrouper(df, ["x"])
    with pytest.raises(ValueError):
        GFGrouper(grouper.gf, ["y"])

    assert grouper.df is df
    assert grouper.gf.by == ["x"]
    assert grouper.vars == ["x"]
    assert_frame_equal(
        grouper.data,
        pl.DataFrame({"x": [2, 3], "_rows": [[0, 1], [2]]}).with_column(
            pl.col("_rows").cast(pl.List(pl.UInt32))
        ),
    )
    assert_iterable_equal(grouper.rows, [[0, 1], [2]])
    assert_iterable_equal(grouper.indices, [0, 0, 1])
    assert_iterable_equal(grouper.size, [2, 1])
    assert grouper.n == 2

    gf = df.groupby("x", maintain_order=True)
    grouper = GFGrouper(gf)
    assert grouper.gf is gf
    assert grouper.df.frame_equal(df)
    assert grouper.gf.by == ["x"]
    assert grouper.vars == ["x"]
    assert_frame_equal(
        grouper.data,
        pl.DataFrame({"x": [2, 3], "_rows": [[0, 1], [2]]}).with_column(
            pl.col("_rows").cast(pl.List(pl.UInt32))
        ),
    )
    assert_frame_equal(grouper.keys, pl.DataFrame({"x": [2, 3]}))
    assert_iterable_equal(grouper.rows, [[0, 1], [2]])
    assert_iterable_equal(grouper.indices, [0, 0, 1])
    assert_iterable_equal(grouper.size, [2, 1])
    assert grouper.n == 2


def test_dfgrouper():
    df = pl.DataFrame({"x": [2, 2, 3], "y": [3, 2, 1]})
    grouper = DFGrouper(df)
    assert grouper.df is df
    assert grouper.gf is None
    assert grouper.vars == []
    assert_frame_equal(
        grouper.data,
        pl.DataFrame({"_rows": [[0, 1, 2]]}),
    )
    assert_frame_equal(grouper.keys, pl.DataFrame())
    assert_iterable_equal(grouper.rows, [[0, 1, 2]])
    assert_iterable_equal(grouper.indices, [0, 0, 0])
    assert_iterable_equal(grouper.size, [3])
    assert grouper.n == 1

    with pytest.raises(ValueError):
        DFGrouper(df, ["x"])


def test_rfgrouper():
    df = pl.DataFrame({"x": [2, 2, 3], "y": [3, 2, 1]})
    grouper = RFGrouper(df)
    assert grouper.df is df
    assert grouper.gf is None
    assert grouper.vars == []
    assert_frame_equal(
        grouper.data,
        pl.DataFrame({"_rows": [[0], [1], [2]]}),
    )
    assert_frame_equal(grouper.keys, pl.DataFrame())
    assert_iterable_equal(grouper.rows, [[0], [1], [2]])
    assert_iterable_equal(grouper.indices, [0, 1, 2])
    assert_iterable_equal(grouper.size, [1, 1, 1])
    assert grouper.n == 3

    grouper = RFGrouper(df, ["x"])
    assert grouper.df is df
    assert grouper.gf is None
    assert grouper.vars == ["x"]
    assert_frame_equal(
        grouper.data,
        pl.DataFrame({"x": [2, 2, 3], "_rows": [[0], [1], [2]]}),
    )
    assert_frame_equal(grouper.keys, pl.DataFrame({"x": [2, 2, 3]}))
    assert_iterable_equal(grouper.rows, [[0], [1], [2]])
    assert_iterable_equal(grouper.indices, [0, 1, 2])
    assert_iterable_equal(grouper.size, [1, 1, 1])
    assert grouper.n == 3


def test_dataframe_namespace():
    df = pl.DataFrame({"x": [2, 2, 3], "y": [4, 5, 6]})
    assert df.datar.grouper is None
    assert df.datar.meta == {}

    df.datar.meta = {"a": 1}
    assert df.datar.meta == {"a": 1}

    gdf = df.datar.group_by("x")
    assert isinstance(gdf, TibbleGrouped)
    assert_iterable_equal(gdf.datar.grouper.vars, ["x"])
    assert isinstance(gdf.datar.grouper, Grouper)
    assert gdf.datar.grouper.df.frame_equal(gdf)
    assert gdf.datar.grouper.vars == ["x"]
    # df not changed
    assert not isinstance(df, TibbleGrouped)
    assert df.datar.grouper is None
    assert df.datar.meta == {"a": 1}

    ugdf = df.datar.ungroup()
    assert isinstance(ugdf, Tibble)
    assert ugdf.datar.grouper is None
    # df not changed
    assert not isinstance(df, TibbleGrouped)
    assert df.datar.grouper is None
    assert df.datar.meta == {"a": 1}

    rdf = df.datar.rowwise()
    assert isinstance(rdf, TibbleRowwise)
    assert rdf.datar.grouper.df is df
    assert rdf.datar.grouper.gf is None
    # df not changed
    assert not isinstance(df, TibbleGrouped)
    assert df.datar.grouper is None
    assert df.datar.meta == {"a": 1}

    urdf = rdf.datar.ungroup()
    assert isinstance(urdf, Tibble)
    assert urdf.datar.grouper is None
    # df not changed
    assert not isinstance(df, TibbleGrouped)
    assert df.datar.grouper is None
    assert df.datar.meta == {"a": 1}

    rdf2 = rdf.datar.rowwise("x")
    assert rdf2.datar.grouper.vars == ["x"]
    assert rdf2.datar.grouper.gf is None
    assert rdf2.datar.grouper.df is rdf
    # df not changed
    assert not isinstance(df, TibbleGrouped)
    assert df.datar.grouper is None
    assert df.datar.meta == {"a": 1}

    urdf2 = rdf2.datar.ungroup()
    assert isinstance(urdf2, Tibble)
    assert urdf2.datar.grouper is None
    # df not changed
    assert not isinstance(df, TibbleGrouped)
    assert df.datar.grouper is None
    assert df.datar.meta == {"a": 1}


def test_series_namespace():
    s = pl.Series("x", [2, 2, 3])
    assert s.datar.grouper is None

    s.datar.grouper = GFGrouper(pl.DataFrame({"x": [2, 2, 3]}), ["x"])
    assert s.datar.grouper is not None

    s.datar.ungroup()
    assert s.datar.grouper is None
