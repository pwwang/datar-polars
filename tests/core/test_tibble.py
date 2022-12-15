import pytest
from datar.core.names import NameNonUniqueError
from polars import datatypes as DT
from polars.exceptions import NotFoundError
from datar_polars.tibble import Tibble

from ..conftest import assert_iterable_equal


def test_tibble():
    df = Tibble({"a": [1]}, meta={"x": 2})
    assert df._datar["x"] == 2
    assert_iterable_equal(df["a"], [1])


def test_tibble_from_pairs():
    with pytest.raises(NameNonUniqueError):
        df = Tibble.from_pairs(
            ["a", "a"],
            [1, 1],
        )
    with pytest.raises(ValueError):
        df = Tibble.from_pairs(
            ["a", "a"],
            [1, ],
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


def test_tibble_from_args():
    df = Tibble.from_args(a=1)
    assert df.shape == (1, 1)
    assert_iterable_equal(df["a"], [1])


def test_tibble_setitem():
    df = Tibble.from_args(a=[1, 2, 3])

    df['b'] = 1
    assert_iterable_equal(df['b'], [1, 1, 1])

    df['c'] = [4, 2, 3]
    assert_iterable_equal(df['c'], [4, 2, 3])
