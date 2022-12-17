import pytest
from datar.core.names import NameNonUniqueError
from polars import datatypes as DT, DataFrame, Series, col
from polars.testing import assert_frame_equal
from polars.exceptions import NotFoundError, ShapeError
from datar_polars.collections import Collection
from datar_polars.tibble import (
    Tibble,
    TibbleGrouped,
    TibbleRowwise,
)

from ..conftest import assert_iterable_equal


def test_tibble():
    df = Tibble({"a": [1]})
    assert df.shape == (1, 1)


def test_tibble_from_pairs():
    with pytest.raises(NameNonUniqueError):
        df = Tibble.from_pairs(
            ["a", "a"],
            [1, 1],
        )
    with pytest.raises(ValueError):
        df = Tibble.from_pairs(
            ["a", "a"],
            [
                1,
            ],
        )

    df = Tibble.from_pairs(
        ["a", "a"],
        [1, [0, 1, 2]],
        _name_repair="universal",
    )
    # assert_frame_equal(df, DataFrame({"a__0": [1, 1, 1], "a__1": [0, 1, 2]}))
    assert_iterable_equal(df["a__0"], [1, 1, 1])
    assert_iterable_equal(df["a__1"], [0, 1, 2])

    df = Tibble.from_pairs(["a"], [1.1])
    assert df["a"].dtype == DT.Float64

    df = Tibble.from_pairs(["a"], [1.1], _dtypes={"a": int})
    assert df["a"].dtype == DT.Int64

    df = Tibble.from_pairs(["a"], [1.1], _dtypes=int)
    assert df["a"].dtype == DT.Int64

    with pytest.raises(NotFoundError):
        df["b"]

    with pytest.raises(ValueError):
        df = Tibble.from_pairs(["a"], [1.1], _name_repair="minimal")

    df = Tibble.from_pairs(["a"], [Collection(1, 2, 3)])
    assert_iterable_equal(df["a"], [1, 2, 3])


def test_tibble_from_args():
    df = Tibble.from_args(a=1)
    assert df.shape == (1, 1)
    assert_iterable_equal(df["a"], [1])

    df = Tibble.from_args(a=1, b=None)
    assert df.shape == (1, 1)
    assert_iterable_equal(df["a"], [1])

    df = Tibble.from_args()
    assert df.shape == (0, 0)

    df = Tibble.from_args(a=Series([1, 2, 3]))
    assert df.shape == (3, 1)

    df = Tibble.from_args(a=Series([1.1, 2.2, 3.3]), _dtypes=int)
    assert_iterable_equal(df["a"], [1, 2, 3])

    df2 = Tibble.from_args(df)
    assert df2.frame_equal(df)

    df2 = Tibble.from_args(DataFrame({"a": [1, 2, 3]}))
    assert_frame_equal(df2, df)

    df2 = Tibble.from_args(x=DataFrame({"a": [1, 2, 3]}))
    assert_frame_equal(df2["x"], df)

    df2 = Tibble.from_args(x=df)
    assert_frame_equal(df2["x"], df)

    df2 = Tibble.from_args(1, df)
    assert_iterable_equal(df2["a"], [1, 2, 3])

    df2 = Tibble.from_args(1, y=df)
    assert_frame_equal(df2["y"], df)

    df2 = Tibble.from_args(1, x=Tibble())
    assert_iterable_equal(df2.columns, ["1", "x"])

    df2 = Tibble.from_args("1", "2", _dtypes=int)
    assert_frame_equal(df2, Tibble.from_args(1, 2, _dtypes=int))

    df2 = Tibble.from_args(x=1, y=[1, 2, 3], _dtypes={"y": str})
    assert_frame_equal(df2, Tibble.from_args(x=1, y=["1", "2", "3"]))

    with pytest.raises(ShapeError):
        Tibble.from_args(x=[2, 3], y=[1, 2, 3])


def test_tibble_setitem():
    df = Tibble.from_args(a=[1, 2, 3])

    df["b"] = 1
    assert_iterable_equal(df["b"], [1, 1, 1])

    df["c"] = [4, 2, 3]
    assert_iterable_equal(df["c"], [4, 2, 3])

    with pytest.raises(ValueError):
        df[1] = 10


def test_groupby_to_tibble_grouped():
    df = Tibble.from_args(a=[2, 2, 3], b=[4, 5, 6])
    gf = df.datar.group_by("a")
    assert isinstance(gf, TibbleGrouped)
    assert gf.datar.grouper.df is df

    assert "grouped: (a), n=2" in repr(gf)
    assert "grouped: (a), n=2" in gf._repr_html_()


def test_rowwise_to_tibble_rowwise():
    df = Tibble.from_args(a=[2, 2, 3], b=[4, 5, 6])
    rf = df.datar.rowwise()
    assert isinstance(rf, TibbleRowwise)
    assert rf.datar.grouper.df is df

    assert "rowwise: ()" in repr(rf)
    assert "rowwise: ()" in rf._repr_html_()


def test_tibble_copy():
    df = Tibble.from_args(a=[1, 2, 3])
    df.datar.meta["a"] = 1

    df2 = df.copy()
    assert df2.datar.meta["a"] == 1


def test_tibble_select():
    df = Tibble.from_args(a=[1, 2, 3], b=[4, 5, 6])
    df.datar.meta["a"] = 1
    df2 = df.select("a")
    assert_iterable_equal(df2.columns, ["a"])
    assert_iterable_equal(df2["a"], [1, 2, 3])
    assert df2.datar.meta["a"] == 1


def test_tibble_filter():
    df = Tibble.from_args(a=[1, 2, 3])
    df.datar.meta["a"] = 1
    df2 = df.filter(col("a") > 1)
    assert_iterable_equal(df2["a"], [2, 3])
    assert df2.datar.meta["a"] == 1


def test_tibble_drop():
    df = Tibble.from_args(a=[1, 2, 3], b=[4, 5, 6])
    df.datar.meta["a"] = 1
    df2 = df.drop("a")
    assert_iterable_equal(df2.columns, ["b"])
    assert df2.datar.meta["a"] == 1
