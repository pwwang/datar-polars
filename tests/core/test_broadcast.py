import pytest

# from datar.base import factor, c
import polars as pl
from polars.testing import assert_frame_equal
from datar.tibble import tibble
from datar_polars.extended import GFGrouper
from datar_polars.tibble import TibbleGrouped, TibbleRowwise, SeriesGrouped
from datar_polars.broadcast import (
    _broadcast_base,
    # broadcast2,
    broadcast_to,
    # add_to_tibble,
    # init_tibble_from,
    # _get_index_grouper,
)

from ..conftest import assert_factor_equal, assert_iterable_equal


def test_broadcast_base_scalar():
    df = tibble(a=1)
    out = _broadcast_base(1, df)
    assert out is df


def test_broadcast_base_array_groupby():
    df = tibble(a=[]).datar.group_by("a")
    with pytest.raises(ValueError, match=r"`\[1, 2\]` must be size"):
        _broadcast_base([1, 2], df)

    # all size-1 groups
    df = tibble(a=[1, 2]).datar.group_by("a")
    out = _broadcast_base([1, 2], df)
    assert_iterable_equal(out['a'], [1, 1, 2, 2])

    df = tibble(a=[1, 2, 1, 2]).datar.group_by("a")
    with pytest.raises(ValueError, match=r"Cannot recycle `x` with size"):
        _broadcast_base([1, 2, 3], df, "x")

    df = tibble(a=[2, 2, 1, 2]).datar.group_by("a")
    with pytest.raises(ValueError, match=r"Cannot recycle `\[1, 2\]`"):
        _broadcast_base([1, 2], df)

    df.datar.meta["broadcasted"] = True
    out = _broadcast_base([1, 2, 3], df)
    assert out is df

    df = tibble(a=[2, 2, 1, 2]).datar.group_by("a")
    out = _broadcast_base([1, 2, 3], df)
    assert_iterable_equal(out['a'], [2, 2, 1, 1, 1, 2])
    assert out.datar.grouper.n == 2
    assert_iterable_equal(out.datar.grouper.rows[0], [0, 1, 5])
    assert_iterable_equal(out.datar.grouper.rows[1], [2, 3, 4])

    df = tibble(a=[1, 2, 2, 3, 3, 3]).datar.group_by("a")
    with pytest.raises(
        ValueError, match=r"Cannot recycle `x` with size 2 to 1"
    ):
        _broadcast_base([1, 2], df, "x")

    df = tibble(a=[1, 2, 1, 2]).datar.group_by("a")
    out = _broadcast_base([1, 2], df)
    assert out is df

    # TibbleGrouped
    df = tibble(a=[1, 2, 1]).datar.group_by("a")
    out = _broadcast_base([1, 2], df)
    assert out.datar.meta["broadcasted"] is True
    assert_iterable_equal(out["a"], [1, 2, 2, 1])

    df = tibble(a=[1, 2, 1]).datar.rowwise("a")
    with pytest.raises(ValueError, match=r"must be size 1"):
        _broadcast_base([1, 2], df)
    with pytest.raises(ValueError, match=r"must be size 1"):
        _broadcast_base([1, 2], df['a'])


def test_broadcast_base_array_ndframe():
    df = tibble(a=[1, 2, 3, 4])
    # df.index = [0, 1, 1, 1]
    df.datar.meta["broadcasted"] = True

    out = _broadcast_base([1, 2], df)
    assert out is df

    df = tibble(a=[1, 2, 3])
    base = _broadcast_base([1, 2, 3], df)
    assert base is df

    df = tibble(a=1)
    base = _broadcast_base([1, 2, 3], df)
    assert_iterable_equal(base['a'], [1] * 3)

    df = tibble(a=[0, 1, 2])
    with pytest.raises(ValueError, match=r"`x` must be size \[1 3\], not 2\."):
        _broadcast_base([1, 2], df, "x")


def test_broadcast_base_groupby_groupby():
    # incompatible grouper
    df = tibble(a=[1, 2, 3]).datar.group_by("a")
    value = tibble(a=[1, 2, 2]).datar.group_by("a")
    with pytest.raises(ValueError, match=r"`x` has an incompatible grouper"):
        _broadcast_base(value, df, "x")

    # base doesn't broadcast when all size-1 groups
    df = tibble(a=[1, 2, 3]).datar.group_by("a")
    base = _broadcast_base(df, df)
    assert base is df

    # group names differ
    value1 = tibble(b=[1, 2, 3]).datar.group_by("b")
    with pytest.raises(ValueError, match=r"`x` has an incompatible grouper"):
        _broadcast_base(value1, df, "x")

    # index not unique, already broadcasted
    df = tibble(a=[2, 1, 1]).datar.group_by("a")
    df.datar.meta["broadcasted"] = True
    base = _broadcast_base(value, df)
    assert base is df

    # size-1 group gets broadcasted
    df = tibble(a=[2, 1, 2]).datar.group_by("a")
    value = tibble(a=[1, 2, 1]).datar.group_by("a")
    base = _broadcast_base(value, df)
    assert_iterable_equal(base['a'], [2, 1, 1, 2])

    # TibbleGrouped
    df = tibble(a=[1, 2, 2]).datar.group_by("a")
    base = _broadcast_base(value, df)
    assert_iterable_equal(base["a"], [1, 1, 2, 2])

    # rowwise
    df = tibble(a=[1, 2, 2]).datar.rowwise()
    base = df['a']
    broadcasted = _broadcast_base(df['a'], base)
    assert broadcasted is base

    base = _broadcast_base(df['a'], df)
    assert base is df

    value = tibble(a=[1, 2, 3]).datar.group_by("a")
    with pytest.raises(ValueError):
        _broadcast_base(value, df)


def test_broadcast_base_groupby_ndframe():
    df = tibble(a=[1, 2, 2, 3, 3, 3])
    with pytest.raises(ValueError, match="Can't recycle"):
        _broadcast_base(df.datar.group_by("a"), df)

    # only when group sizes are len(value) or [1, len(value)]
    value = tibble(a=[2, 1, 2, 1]).datar.group_by("a")
    df = tibble(a=3)
    base = _broadcast_base(value, df)
    assert_iterable_equal(base["a"], [3, 3, 3, 3])

    df = tibble(a=[3, 4])
    base = _broadcast_base(value, df)
    assert_iterable_equal(base["a"], [3, 3, 4, 4])

    # TibbleGrouped
    value = tibble(a=[2, 1, 2, 1]).datar.group_by("a")
    base = _broadcast_base(value, df)
    assert_iterable_equal(base["a"], [3, 3, 4, 4])


def test_broadcast_base_ndframe_groupby():
    df = tibble(a=1, b=2).datar.group_by("a")
    gp = GFGrouper(df, ["b"])
    value = pl.Series("b", [1]).datar.as_agg(gp)
    with pytest.raises(
        ValueError, match="`b` is an incompatible aggregated result"
    ):
        _broadcast_base(value, df)

    df = tibble(a=[2, 1, 2, 1]).datar.group_by("a")
    value = pl.Series([3, 4]).datar.as_agg(df.datar.grouper)
    base = _broadcast_base(value, df)
    assert base is df

    df = tibble(a=[1, 2]).datar.group_by("a")
    df.datar.meta["broadcasted"] = True
    base = _broadcast_base(value, df)
    assert base is df

    # Broadcast size-1 groups in base
    df = tibble(a=[1, 2]).datar.group_by("a")
    value = pl.Series([3, 4, 4]).datar.as_agg(df.datar.grouper, [1, 2])
    base = _broadcast_base(value, df)
    assert_iterable_equal(base['a'], [1, 2, 2])

    # Rowwise
    df = tibble(a=[1, 2]).datar.rowwise()
    value = tibble(a=1)
    base = _broadcast_base(value, df)
    assert base is df

    base = df['a']
    broadcasted = _broadcast_base(value, base)
    assert broadcasted is base

    value = tibble(a=[1, 2, 3])
    with pytest.raises(ValueError):
        _broadcast_base(value, df)


def test_broadcast_base_ndframe_ndframe():
    df = tibble(a=[1, 2, 3])
    df.datar.agg_index = tibble(a=[0, 1, 1])
    value = tibble(a=[1, 2, 3])
    base = _broadcast_base(value, df)
    assert base is df

    df = tibble(a=[1, 2, 3])
    value = tibble(a=[1, 2, 3])
    value.datar.agg_index = tibble(a=[1, 2, 3])
    base = _broadcast_base(value, df)
    assert_iterable_equal(base['a'], [1, 2, 3])


# broadcast_to
def test_broadcast_to_scalar():
    value = broadcast_to(1, 2)
    assert value == 1


# def test_broadcast_to_factor():
#     x = factor(list("abc"))
#     base = Series([1, 2, 3])
#     out = broadcast_to(x, base.index)
#     assert_factor_equal(out.values, x)

#     # empty
#     x = factor([], levels=list("abc"))
#     base = Series([], dtype=object).groupby([])
#     out = broadcast_to(x, get_obj(base).index, base.grouper)
#     assert_factor_equal(out.values, x)

#     # grouped
#     x = factor(["a", "b"], levels=list("abc"))
#     base = Series([1, 2, 3, 4], index=[4, 5, 6, 7]).groupby([1, 1, 2, 2])
#     out = broadcast_to(x, get_obj(base).index, base.grouper)
#     assert_iterable_equal(out.index, [4, 5, 6, 7])
#     assert_iterable_equal(out, ["a", "b"] * 2)


def test_broadcast_to_arrays_ndframe():
    with pytest.raises(ValueError, match=r"Can't broadcast a 0-size object"):
        broadcast_to([], 3)

    with pytest.raises(ValueError, match=r"Can't broadcast a 3-size"):
        broadcast_to([1, 2, 3], 2)

    value = broadcast_to([1, 2], 2)
    assert_iterable_equal(value, [1, 2])

    df = tibble(a=[1, 1, 3, 3]).datar.group_by("a")
    value = broadcast_to([1, 2], df.datar.grouper)
    assert isinstance(value, SeriesGrouped)
    assert_iterable_equal(value, [1, 2, 1, 2])

    df = tibble(a=[1, 1, 3]).datar.group_by("a")
    with pytest.raises(ValueError, match=r"Can't broadcast a 2-size"):
        broadcast_to([1, 2], df.datar.grouper)


def test_broadcast_to_arrays_groupby():
    df = tibble(x=[]).datar.group_by("x")
    value = broadcast_to([], df.datar.grouper)
    assert len(value) == 0

    df = tibble(x=[2, 1, 2, 1])
    df = df.datar.group_by("x")
    value = broadcast_to(["a", "b"], df.datar.grouper)
    assert_iterable_equal(value, ["a", "a", "b", "b"])


def test_broadcast_to_ndframe_ndframe():
    value = pl.Series([1, 2, 3])
    out = broadcast_to(value, 3)
    assert_iterable_equal(out, [1, 2, 3])

    out = broadcast_to(value.to_frame(name="x"), 3)
    assert_iterable_equal(out['x'], [1, 2, 3])


def test_broadcast_to_ndframe_groupby():
    df = tibble(x=[1, 2, 2, 1, 1, 2, 3, 3, 3]).datar.group_by("x")
    value = pl.Series([8, 9, 10])
    out = broadcast_to(value, df.datar.grouper)
    assert_iterable_equal(out, [8, 8, 9, 9, 10, 10, 8, 9, 10])

    out = broadcast_to(value.to_frame(name="x"), df.datar.grouper)
    assert_iterable_equal(out['x'], [8, 8, 9, 9, 10, 10, 8, 9, 10])


def test_broadcast_to_agg_groupby():
    df = tibble(
        x=[1, 2, 2, 1, 1, 2, 3, 3, 3],
        y=[1, 2, 3, 4, 5, 6, 7, 8, 9],
    ).datar.group_by("x")
    value = df['y'].sum()
    out = broadcast_to(value, df.datar.grouper)
    assert isinstance(out, SeriesGrouped)
    assert_iterable_equal(out, [10, 11, 11, 10, 10, 11, 24, 24, 24])

    out = broadcast_to(value.to_frame(name="y"), df.datar.grouper)
    assert_iterable_equal(out['y'], [10, 11, 11, 10, 10, 11, 24, 24, 24])


def test_broadcast_to_groupby_ndframe():
    df = tibble(x=[1, 2, 2, 1, 1, 2], y=[1, 2, 3, 4, 5, 6]).datar.group_by("x")
    out = broadcast_to(df['x'], df.datar.grouper)
    assert_iterable_equal(out, df['x'])

    out = broadcast_to(df, df.datar.grouper)
    assert_frame_equal(out, df)

    nn = df['y'].sum()
    nn = broadcast_to(nn, df.datar.grouper)
    assert isinstance(nn, SeriesGrouped)
    assert_iterable_equal(nn, [10, 11, 11, 10, 10, 11])


# def test_broadcast2():
#     # types: scalar/arrays, DattaFrame/Series, GroupBy, TibbleGrouped
#     # scalar/arrays <-> other
#     left, right, grouper, is_rowwise = broadcast2(
#         1,
#         Series(1),
#     )
#     assert left == 1
#     assert_iterable_equal(right, [1])
#     assert grouper is None
#     assert not is_rowwise

#     # not happening in practice, since 1 + 1 will be calculated directory
#     left, right, grouper, is_rowwise = broadcast2(
#         1,
#         2,
#     )
#     assert left == 1
#     assert right == 2
#     assert grouper is None
#     assert not is_rowwise

#     left, right, grouper, is_rowwise = broadcast2(
#         Series([1, 2, 3, 4]).groupby([1, 2, 1, 2]),
#         [7, 8],
#     )
#     assert_iterable_equal(left, [1, 2, 3, 4])
#     assert_iterable_equal(right, [7, 7, 8, 8])
#     assert_iterable_equal(grouper.group_info[0], [0, 1, 0, 1])
#     assert not is_rowwise

#     left, right, grouper, is_rowwise = broadcast2(
#         tibble(x=[1, 2, 3, 4]).rowwise(),
#         7,
#     )

#     assert_iterable_equal(left.x, [1, 2, 3, 4])
#     assert right == 7
#     assert_iterable_equal(grouper.group_info[0], [0, 1, 2, 3])
#     assert is_rowwise


# # init_tibble_from
# def test_init_tibble_from_scalarorarrays():
#     x = init_tibble_from(1, "a")
#     assert_frame_equal(x, tibble(a=1))

#     x = init_tibble_from([1, 2], "a")
#     assert_frame_equal(x, tibble(a=[1, 2]))


# def test_init_tibble_from_series():
#     x = Series(1)
#     df = init_tibble_from(x, "x")
#     assert_frame_equal(df, tibble(x=1))


# def test_init_tibble_from_sgb():
#     x = tibble(a=[1, 2, 3]).groupby("a").a
#     df = init_tibble_from(x, "a")
#     assert isinstance(df, TibbleGrouped)
#     assert_iterable_equal(get_obj(df.a), get_obj(x))

#     # rowwise
#     x = tibble(a=[1, 2, 3]).rowwise().a
#     df = init_tibble_from(x, "a")
#     assert isinstance(df, TibbleRowwise)
#     assert_iterable_equal(get_obj(df.a), get_obj(x))


# def test_init_tibble_from_df():
#     x = tibble(a=[1, 2, 3])
#     df = init_tibble_from(x, None)
#     assert_frame_equal(x, df)

#     # TibbleGrouped
#     x = tibble(a=[1, 2, 3]).group_by("a")
#     df = init_tibble_from(x, "df")
#     assert isinstance(df, TibbleGrouped)
#     assert_iterable_equal(df.columns, ["df$a"])


# # add_to_tibble
# def test_add_to_tibble():
#     df = tibble(a=[1, 2])
#     tbl = add_to_tibble(df, None, None)
#     assert tbl is df

#     tbl = add_to_tibble(None, None, df)
#     assert_frame_equal(tbl, df)

#     df = df.group_by("a")
#     tbl = add_to_tibble(df, "b", [3, 4], broadcast_tbl=True)
#     assert isinstance(tbl, TibbleGrouped)
#     assert get_obj(tbl.b).tolist() == [3, 4, 3, 4]

#     value = tibble(b=[3, 4])
#     df = tibble(a=[1, 2])
#     tbl = add_to_tibble(df, None, value)
#     assert_frame_equal(tbl, tibble(a=[1, 2], b=[3, 4]))

#     # allow dup names
#     df = tibble(a=[1, 2])
#     value = tibble(a=[3, 4])
#     tbl = add_to_tibble(df, None, value, allow_dup_names=True)
#     assert_iterable_equal(tbl.columns, ["a", "a"])


# def test_catindex():
#     df = tibble(g=[1, 1, 2, 2]).group_by("g")
#     x = Series(
#         [5, 6],
#         index=Index(Categorical([1, 2], categories=[1, 2, 3]), name="g"),
#     )
#     out = broadcast_to(x, df.index, df.g.grouper)
#     assert_iterable_equal(out, [5, 5, 6, 6])

#     df = tibble(g=[1, 2]).group_by("g")
#     x = Series(
#         [5, 5, 6, 6],
#         index=Index(Categorical([1, 1, 2, 2], categories=[1, 2, 3]), name="g"),
#     )
#     base = _broadcast_base(x, df)
#     assert_iterable_equal(get_obj(base.g), [1, 1, 2, 2])


# def test_recycle_scalar_composed_base():
#     base = tibble(x=1)
#     value = Series([1, 2])
#     out = _broadcast_base(value, base)
#     assert_iterable_equal(out.x, [1, 1])


# def test_recycle_len1_list_gets_scalar():
#     assert broadcast_to([1], None, None) == 1


# def test_recycle_same_index_ndframe():
#     index = Index([1, 2])
#     s1 = Series([1, 2], index=index)
#     s2 = Series([3, 4], index=index)
#     out = broadcast_to(s2, s1.index)
#     assert out is s2


# def test_scalar_composed_value_gets_recycled():
#     s = Series(1)
#     out = broadcast_to(s, index=Index(range(3)))
#     assert_iterable_equal(out, [1, 1, 1])


# def test_empty_frame_gets_recycled():
#     df = DataFrame()
#     out = broadcast_to(df, index=Index(range(3)))
#     assert out.index.equals(Index(range(3)))


# def test_incompatible_index():
#     s1 = Series([1, 2], index=[0, 1])
#     s2 = Series([3, 4], index=[1, 2]).groupby([1, 2])
#     with pytest.raises(ValueError):
#         broadcast_to(s1, get_obj(s2).index)

#     with pytest.raises(ValueError):
#         broadcast_to(s1, get_obj(s2).index, s2.grouper)


# def test_nongrouped_value_with_equal_index_gets_recycled():
#     s1 = Series([1, 2], index=[3, 4]).groupby([1, 2])
#     s2 = Series([3, 4], index=[3, 4])
#     out = broadcast_to(s2, get_obj(s1).index, s1.grouper)
#     assert out is s2


# def test_get_index_grouper():
#     out = _get_index_grouper(1)
#     assert out == (None, None)
