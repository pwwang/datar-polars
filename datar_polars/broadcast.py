"""All broadcasting (value recycling) rules

Should be only used with backend pandas or modin

There are basically 4 places where broadcasting is needed.
1. in `mutate()`
2. in `summarise()`
3. in `tibble()`
4. and arithemetic operators

The base to be used to broadcast other values should be either:
- a Series object (e.g. `f.a` when data is a DataFrame object)
- a DataFrame/Tibble object
    e.g. `f[f.a:]` when data is a DataFrame/Tibble object
- a SeriesGrouped object (e.g. `f.a` when data is a TibbleGrouped object)
- a TibbleGrouped object (e.g. `f[f.a:]` when data is a TibbleGrouped object)

In `summarise()`, `tibble()` and the operands for arithemetic operators, the
base should also be broadcasted. For example:

>>> tibble(x=[1, 2]) >> group_by(f.x) >> summarise(x=f.x + [1, 2])
>>> # tibble(x=[2, 3, 3, 4])

In the above example, `f.x` is broadcasted into `[1, 1, 2, 2]` based on the
right operand `[1, 2]`
"""
from __future__ import annotations

import time
from functools import singledispatch
from typing import TYPE_CHECKING, Any, List, Tuple, Union

import numpy as np
import polars as pl

from .tibble import Tibble, TibbleGrouped, TibbleRowwise, SeriesGrouped
from .utils import name_of, is_scalar


def _regroup(
    x: TibbleGrouped,
    new_sizes: np.ndarray,
) -> TibbleGrouped:
    """Regroup grouped object if some groups could be broadcasted"""
    grouper = x.datar.grouper
    base_sizes = grouper.size
    base_sizes1 = base_sizes == 1
    repeats = np.ones(x.shape[0], dtype=int)
    # If any group is 1-sized, try to broadcast it
    if base_sizes1.any():
        # better way?
        idx_to_broadcast = np.concatenate(
            [grouper.group_indices[i] for i in np.flatnonzero(base_sizes1)]
        )
        if isinstance(new_sizes, int):
            # broadcast all 1-sized groups to given size
            repeats[idx_to_broadcast] = new_sizes
        else:
            # broadcast each 1-sized group to the corresponding size
            repeats[idx_to_broadcast] = new_sizes[base_sizes1]

    if (repeats == 1).all():
        return x

    indices = np.arange(x.shape[0]).repeat(repeats)
    return x[indices, :].datar.group_by(grouper.vars, sort=grouper.sort)


def _agg_result_compatible(index: DataFrame | Series, grouper: Grouper) -> bool:
    """Check index of an aggregated result is compatible with a grouper"""
    if index.columns != grouper.vars:
        return False

    if isinstance(index, Series):
        index = index.to_frame()

    if not index.frame_equal(grouper.keys):
        return False

    # also check the size
    # if isinstance(index, CategoricalIndex):
    #     index = index.remove_unused_categories()

    size1 = (
        index
        .select(pl.concat_list(pl.all()))[:, 0]
        .value_counts()[:, -1]
        .to_numpy()
    )
    size2 = grouper.size
    return (
        (size1.values == 1)
        | (size2.values == 1)
        | (size1.values == size2.values)
    ).all()


def _realign_indexes(value: TibbleGrouped, grouper: Grouper) -> np.ndarray:
    """Realign indexes of a value to a grouper"""
    v_new_indices = []
    g_indices = []
    for i, v_ind in enumerate(value.datar.grouper.rows):
        g_ind = grouper.rows[i]
        if v_ind.size == 1 and g_ind.size > 1:
            v_new_indices.extend(v_ind.repeat(g_ind.size))
        else:
            v_new_indices.extend(v_ind)
        g_indices.extend(g_ind)

    value = value[v_new_indices, :]
    sorted_indices = np.argsort(g_indices)
    return value[sorted_indices, :]


@singledispatch
def _broadcast_base(
    value,
    base: BroadcastingBaseType,
    name: str = None,
) -> BroadcastingBaseType:
    """Broadcast the base dataframe when value has more elements

    Args:
        value: The value
        base: The base data frame

    Returns:
        A tuple of the transformed value and base
    """
    # plain arrays, scalars, np.array(True)
    if is_scalar(value) or len(value) == 1:
        return base

    name = name or name_of(value) or str(value)

    if isinstance(base, TibbleGrouped):
        sizes = base.grouper.size
        usizes = sizes.unique()

        # Broadcast each group into size len(value)
        # usizes should be only 1 number, or [1, len(value)]
        if usizes.size == 0:
            raise ValueError(f"`{name}` must be size [0 1], not {len(value)}.")

        if usizes.size == 1:
            if usizes[0] == 1:
                return _regroup(base, len(value))

            if usizes[0] != len(value):
                raise ValueError(
                    f"Cannot recycle `{name}` with size "
                    f"{len(value)} to {usizes[0]}."
                )
            return base

        if usizes.size == 2:
            if set(usizes) != set([1, len(value)]):
                size_tip = usizes[usizes != len(value)][0]
                raise ValueError(
                    f"Cannot recycle `{name}` with size "
                    f"{len(value)} to {size_tip}."
                )

            # if not get_obj(base).index.is_unique:
            #     # already broadcasted
            #     return base

            # broadcast size=1 groups and regroup
            return _regroup(base, len(value))

        size_tip = usizes[usizes != len(value)][0]
        raise ValueError(
            f"Cannot recycle `{name}` with size {len(value)} to {size_tip}."
        )

    if isinstance(base, TibbleRowwise):
        # len(value) > 1
        raise ValueError(f"`{name}` must be size 1, not {len(value)}.")

    # DF/Series
    # if not base.index.is_unique:
    #     return base

    # The length should be [1, len(value)]
    if base.shape[0] == len(value):
        return base

    if base.shape[0] == 1:
        return base[[0] * len(value), :]

    raise ValueError(
        f"`{name}` must be size [1 {base.shape[0]}], not {len(value)}."
    )


@_broadcast_base.register(SeriesGrouped)
@_broadcast_base.register(TibbleGrouped)
def _(
    value: TibbleGrouped | SeriesGrouped,
    base: BroadcastingBaseType,
    name: str = None,
) -> Tuple[Any, BroadcastingBaseType]:
    """Broadcast grouped object when value is a grouped object"""
    name = name or name_of(value) or str(value)

    if isinstance(base, TibbleGrouped):
        if not value.datar.grouper.compatible_with(base.datar.grouper):
            raise ValueError(f"`{name}` has an incompatible grouper.")

        if (value.grouper.size() == 1).all():
            # Don't modify base when values are 1-size groups
            # Leave it to broadcast_to() to broadcast to values
            # No need to broadcast the base
            return base

        # check if base has already broadcasted
        # if not get_obj(base).index.is_unique:
        #     return base

        # Broadcast size-1 groups in base
        return _regroup(base, value.grouper.size())

    if isinstance(base, TibbleRowwise):
        if not value.grouper.compatible_with(base.datar.grouper):
            raise ValueError(f"`{name}` has an incompatible grouper.")
        # Don't broadcast rowwise
        return base

    # base is ungrouped
    # DataFrame/Series

    # df >> group_by(f.a) >> mutate(new_col=tibble(x=1, y=f.a))
    #                                              ^^^^^^^^^^
    val_sizes = value.grouper.size

    if base.shape[0] == 1 or (val_sizes == base.shape[0]).all():
        if base.shape[0] == 1:
            repeats = value.shape[0]
        else:
            repeats = val_sizes

        return base[np.repeat(range(base.shape[0]), repeats), :].datar.group_by(
            value.grouper.vas,
            sort=value.grouper.sort,
        )

    # Otherwise
    raise ValueError(f"Can't recycle a grouped object `{name}` to ungrouped.")


@_broadcast_base.register(pl.Series)
@_broadcast_base.register(pl.DataFrame)
def _(
    value: pl.DataFrame | pl.Series,
    base: BroadcastingBaseType,
    name: str = None,
) -> Tuple[Any, BroadcastingBaseType]:
    """Broadcast a DataFrame/Series object to a grouped object

    This is mostly a case when trying to broadcast an aggregated object to
    the original object. For example: `gf >> mutate(f.x / sum(f.x))`

    But `sum(f.x)` could return a Series object that has more than 1 elements
    for a group. Then we need to broadcast `f.x` to match the result.
    """
    if isinstance(base, GroupBy):
        if getattr(base, "is_rowwise", False):
            return base

        # Now the index of value works more like grouping data
        if not _agg_result_compatible(value.index, base.grouper):
            name = name or name_of(value) or str(value)
            raise ValueError(f"`{name}` is an incompatible aggregated result.")

        if isinstance(value.index, CategoricalIndex):
            val_sizes = value.index.remove_unused_categories().value_counts(
                sort=False,
            )
        else:
            val_sizes = value.index.value_counts(sort=False)

        if (val_sizes == 1).all():
            # Don't modify base when values are 1-size groups
            # Leave it to broadcast_to() to broadcast to values
            # No need to broadcast the base
            return base

        if not base.index.is_unique:
            return base

        # Broadcast size-1 groups in base
        return _regroup(base, val_sizes)

    if isinstance(base, TibbleRowwise):
        if value.shape[0] != 1 and not base.index.equals(value.index):
            raise ValueError(f"`{name}` must be size 1, not {value.shape[0]}.")
        # Don't broadcast rowwise
        return base

    if isinstance(base, TibbleGrouped):
        grouped_old = base._datar["grouped"]
        grouped_new = _broadcast_base(value, grouped_old, name)
        if grouped_new is not grouped_old:
            base = TibbleGrouped.from_groupby(grouped_new, deep=False)
        return base

    # base: DataFrame/Series
    if not base.index.is_unique:
        return base

    if (
        base.index.size == 1
        and base.index[0] == 0
        and not base.index.equals(value.index)
    ):
        base = base.reindex([0] * value.index.size)
        base.index = value.index
        return base

    return base


@singledispatch
def broadcast_to(
    value,
    index: Index,
    grouper: Grouper = None,
) -> Series:
    """Broastcast value to expected dimension, the result is a series with
    the given index

    Before calling this function, make sure that the index is already
    broadcasted based on the value. This means that index is always the wider
    then the value if it has one. Also the size for each group is larger then
    the length of the value if value doesn't have an index.

    Examples:
        >>> # 1. broadcast scalars/arrays(no index)
        >>> broadcast_to(1, Index(range(3)))
        >>> # Series: [1,1,1], index: range(3)
        >>> broadcast_to([1, 2], Index([0, 0, 1, 1]))
        >>> # ValueError
        >>>
        >>> # grouper: size()        # [2, 2]
        >>> # grouper: result_index  # ['a', 'b']
        >>> broadcast_to([1, 2], Index([0, 0, 1, 1]), grouper)
        >>> # Series: [1, 2, 1, 2], index: [0, 0, 1, 1]
        >>>
        >>> # 2. broadcast a Series
        >>> broadcast(Series([1, 2]), index=[0, 0, 1, 1])
        >>> # Series: [1, 1, 2, 2], index: [0, 0, 1, 1]
        >>> broadcast(
        >>>     Series([1, 2], index=['a', 'b']),
        >>>     index=[4, 4, 5, 5],
        >>>     grouper,
        >>> )
        >>> # Series: [1, 1, 2, 2], index: [4, 4, 5, 5]

    Args:
        value: Value to be broadcasted
        index: The index to broadcasted to
        grouper: The grouper of the original value

    Returns:
        The series with the given index
    """
    if is_scalar(value):
        return value

    if len(value) == 1 and is_scalar(value[0]):
        return value[0]

    if not grouper:
        if len(value) == 0:
            return Series(value, index=index, dtype=object)

        # Series will raise the length problem
        return Series(value, index=index)

    gsizes = grouper.size()
    if gsizes.size == 0:
        return Series(value, index=index, dtype=object)

    # broadcast value to each group
    # length of each group is checked in _broadcast_base
    # A better way to distribute the value to each group?
    # TODO: NA in grouper_index?
    # See also https://github.com/pandas-dev/pandas/issues/35202
    idx = np.concatenate(
        [grouper.groups[gdata] for gdata in grouper.result_index]
    )
    # make np.tile([[3, 4]], 2) to be [[3, 4], [3, 4]],
    # instead of [[3, 4, 3, 4]]
    repeats = (grouper.ngroups,) + (1,) * (np.ndim(value) - 1)
    # Can't do reindex for this case:
    # >>> df = tibble(a=[1, 1, 2, 2, 3, 3], _index=[1, 1, 2, 2, 3, 3])
    # >>> df["b"] = [4, 5]  # error
    out = Series(np.tile(value, repeats).tolist(), index=idx)
    if out.index.is_unique:
        return out.reindex(index)

    # recode index
    index_dict = out.index.to_frame().groupby(out.index).grouper.indices
    new_idx = Index(range(out.size))
    new_index = np.ones(out.size, dtype=int)

    for ix, rowids in index_dict.items():
        np.place(new_index, index == ix, rowids)

    out.index = new_idx
    out = out.reindex(new_index)
    new_idx = np.ones(out.size, dtype=int)
    for ix, rowids in index_dict.items():
        new_idx[np.isin(new_index, rowids)] = ix

    out.index = new_idx
    return out


@broadcast_to.register(Categorical)
def _(
    value: Categorical,
    index: Index,
    grouper: Grouper = None,
) -> Series:
    """Broadcast categorical data"""
    if not grouper:
        if value.size == 0:
            return Series(value, index=index)
        if value.size == 1:
            return Series(value.repeat(index.size), index=index)
        # Series will raise the length problem
        return Series(value, index=index)

    gsizes = grouper.size()
    if gsizes.size == 0:
        return Series(value, index=index)

    # broadcast value to each group
    # length of each group is checked in _broadcast_base
    # A better way to distribute the value to each group?
    idx = np.concatenate(
        [grouper.groups[gdata] for gdata in grouper.result_index]
    )
    # make np.tile([[3, 4]], 2) to be [[3, 4], [3, 4]],
    # instead of [[3, 4, 3, 4]]
    repeats = grouper.ngroups
    value = Categorical(np.tile(value, repeats), categories=value.categories)
    return Series(value, index=idx).reindex(index)


@broadcast_to.register(NDFrame)
def _(
    value: NDFrame,
    index: Index,
    grouper: Grouper = None,
) -> Union[Tibble, Series]:
    """Broadcast series/dataframe"""
    if value.index is index:
        # if it is the same index
        # e.g. transform results
        return value

    if not grouper:
        # recycle row-1 series/frame
        if value.index.size == 1 and value.index[0] == 0:
            value = value.reindex([0] * index.size)
            value.index = index

        # empty frame get recycled
        if isinstance(value, DataFrame) and value.index.size == 0:
            value.index = index

        # if not value.index.equals(index):
        if not value.index.equals(index) and frozenset(
            value.index
        ) != frozenset(index):
            raise ValueError("Value has incompatible index.")

        if isinstance(value, Series):
            return Series(value, name=value.name, index=index)

        return Tibble(value, index=index)

    # now target is grouped and the value's index is overlapping with the
    # grouper's index
    # This is typically an aggregated result to the orignal structure
    # For example:  f.x.mean() / f.x
    if _agg_result_compatible(value.index, grouper):

        if isinstance(value, Series):
            out = Series(
                value,
                index=grouper.result_index.take(grouper.group_info[0]),
                name=value.name,
                copy=False,
            )
        else:  # DataFrame
            out = Tibble(
                value,
                index=grouper.result_index.take(grouper.group_info[0]),
                copy=False,
            )

        out.index = index
        return out

    if value.index.equals(index):
        return value

    raise ValueError("Incompatible value to recycle.")


@broadcast_to.register(GroupBy)
def _(
    value: GroupBy,
    index: Index,
    grouper: Grouper = None,
) -> Union[Series, Tibble]:
    """Broadcast pandas grouped object"""
    if not grouper:
        raise ValueError(
            "Can't broadcast grouped object to a non-grouped object."
        )

    # Compatibility has been checked in _broadcast_base
    if isinstance(value, SeriesGrouped):
        if np.array_equal(grouper.group_info[0], value.grouper.group_info[0]):
            return Series(
                get_obj(value).values, index=index, name=get_obj(value).name
            )

        # broadcast size-one groups and
        # realign the index
        revalue = _realign_indexes(value, grouper)
        return Series(revalue, index=index, name=get_obj(value).name)

    if np.array_equal(grouper.group_info[0], value.grouper.group_info[0]):
        return Tibble(
            get_obj(value).values, index=index, columns=get_obj(value).columns
        )

    # realign the index
    revalue = _realign_indexes(value, grouper)
    return Tibble(revalue, index=index, columns=get_obj(value).columns)


@broadcast_to.register(TibbleGrouped)
def _(
    value: TibbleGrouped,
    index: Index,
    grouper: Grouper = None,
) -> Tibble:
    """Broadcast TibbleGrouped object"""
    return broadcast_to(
        value._datar["grouped"],
        index=index,
        grouper=grouper,
    )


@singledispatch
def _get_index_grouper(value) -> Tuple[Index, Grouper]:
    return None, None


@_get_index_grouper.register(TibbleGrouped)
def _(value):
    return value.index, value._datar["grouped"].grouper


@_get_index_grouper.register(NDFrame)
def _(value):
    return value.index, None


@_get_index_grouper.register(GroupBy)
def _(value):
    return get_obj(value).index, value.grouper


@singledispatch
def _type_priority(value) -> int:
    return -1


@_type_priority.register(GroupBy)
def _(value):
    return 10


@_type_priority.register(NDFrame)
def _(value):
    return 5


@_type_priority.register(TibbleGrouped)
def _(value):
    return 10


@singledispatch
def _ungroup(value):
    return value


@_ungroup.register(GroupBy)
def _(value):
    return get_obj(value)


@_ungroup.register(TibbleGrouped)
def _(value):
    return value._datar["grouped"].obj


@singledispatch
def broadcast2(left, right) -> Tuple[Any, Any, Grouper, bool]:
    """Broadcast 2 values for operators"""
    left_pri = _type_priority(left)
    right_pri = _type_priority(right)
    if left_pri == right_pri == -1:
        return left, right, None, False

    if left_pri > right_pri:
        left = _broadcast_base(right, left)
        index, grouper = _get_index_grouper(left)
        is_rowwise = isinstance(left, TibbleRowwise) or getattr(
            left, "is_rowwise", False
        )
        right = broadcast_to(right, index, grouper)
    else:
        right = _broadcast_base(left, right)
        index, grouper = _get_index_grouper(right)
        is_rowwise = isinstance(right, TibbleRowwise) or getattr(
            right, "is_rowwise", False
        )
        left = broadcast_to(left, index, grouper)

    return _ungroup(left), _ungroup(right), grouper, is_rowwise


@singledispatch
def init_tibble_from(value, name: str) -> Tibble:
    """Initialize a tibble from a value"""
    if is_scalar(value):
        return Tibble({name: [value]})

    return Tibble({name: value})


@init_tibble_from.register(Series)
def _(value: Series, name: str) -> Tibble:
    # Deprecate warning, None will be used as series name in the future
    # So use 0 as default here
    name = name or value.name or 0
    return Tibble(value.to_frame(name=name), copy=False)


@init_tibble_from.register(SeriesGrouped)
def _(value: SeriesGrouped, name: str) -> TibbleGrouped:

    return value.to_frame(name).datar.group_by(value.datar.grouper.vars)


@init_tibble_from.register(DataFrame)
@init_tibble_from.register(DataFrameGroupBy)
def _(value: DataFrame | DataFrame, name: str) -> Tibble:
    from .api.tibble.tibble import as_tibble

    result = as_tibble(value, __ast_fallback="normal", __backend="polars")

    if name:
        result = result.copy()
        result.columns = [f"{name}${col}" for col in result.columns]
    return result


def add_to_tibble(
    tbl: Tibble,
    name: str,
    value: Any,
    broadcast_tbl: bool = False,
) -> "Tibble":
    """Add data to tibble"""
    if value is None:
        return tbl

    if tbl is None:
        return init_tibble_from(value, name)

    if broadcast_tbl:
        tbl = _broadcast_base(value, tbl, name)

    if not name and isinstance(value, DataFrame):
        for col in value.columns:
            tbl = add_to_tibble(tbl, col, value[col])

        return tbl

    tbl[name] = value
    return tbl