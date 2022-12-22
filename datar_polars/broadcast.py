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
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import numpy as np
import polars as pl
from datar_polars.extended import Grouper

from .tibble import (
    Aggregated,
    Tibble,
    TibbleGrouped,
    TibbleRowwise,
    TibbleAgg,
    SeriesGrouped,
    SeriesRowwise,
    SeriesAgg,
)
from .factor import Factor
from .utils import name_of, is_scalar, unique


# def _realign_indexes(value: TibbleGrouped, grouper: Grouper) -> np.ndarray:
#     """Realign indexes of a value to a grouper"""
#     v_new_indices = []
#     g_indices = []
#     for i, v_ind in enumerate(value.datar.grouper.rows):
#         g_ind = grouper.rows[i]
#         if v_ind.size == 1 and g_ind.size > 1:
#             v_new_indices.extend(v_ind.repeat(g_ind.size))
#         else:
#             v_new_indices.extend(v_ind)
#         g_indices.extend(g_ind)

#     value = value[v_new_indices, :]
#     sorted_indices = np.argsort(g_indices)
#     return value[sorted_indices, :]


@singledispatch
def _broadcast_base(
    value,
    base: pl.DataFrame | pl.Series,
    name: str = None,
) -> pl.DataFrame | pl.Series:
    """Broadcast the base dataframe when value has more elements

    Args:
        value: The value
        base: The base data frame

    Returns:
        A tuple of the transformed value and base
    """
    if base.datar.meta.get("broadcasted"):
        return base

    # plain arrays, scalars, np.array(True)
    if is_scalar(value) or len(value) == 1:
        return base

    name = name or name_of(value) or str(value)

    if isinstance(base, TibbleGrouped):
        sizes = base.datar.grouper.size
        usizes = unique(sizes)

        # Broadcast each group into size len(value)
        # usizes should be only 1 number, or [1, len(value)]
        if usizes.size == 0:
            raise ValueError(f"`{name}` must be size [0 1], not {len(value)}.")

        if usizes.size == 1:
            if usizes[0] == 1:
                return base.datar.regroup(len(value))

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

            # broadcast size=1 groups and regroup
            return base.datar.regroup(len(value))

        size_tip = usizes[usizes != len(value)][0]
        raise ValueError(
            f"Cannot recycle `{name}` with size {len(value)} to {size_tip}."
        )

    if isinstance(base, (SeriesRowwise, TibbleRowwise)):
        # len(value) > 1
        raise ValueError(f"`{name}` must be size 1, not {len(value)}.")

    # The length should be [1, len(value)]
    if base.shape[0] == len(value):
        return base

    if base.shape[0] == 1:
        out = base[[0] * len(value), :]
        out.datar.meta["broadcasted"] = True
        return out

    raise ValueError(
        f"`{name}` must be size [1 {base.shape[0]}], not {len(value)}."
    )


@_broadcast_base.register(SeriesRowwise)
@_broadcast_base.register(SeriesGrouped)
@_broadcast_base.register(TibbleGrouped)
def _(
    value: TibbleGrouped | SeriesGrouped,
    base: pl.DataFrame | pl.Series,
    name: str = None,
) -> pl.DataFrame | pl.Series:
    """Broadcast grouped object when value is a grouped object"""
    # check if base has already broadcasted
    if base.datar.meta.get("broadcasted"):
        return base

    name = name or name_of(value) or str(value)

    if isinstance(base, TibbleGrouped):
        if not value.datar.grouper.compatible_with(base.datar.grouper):
            raise ValueError(f"`{name}` has an incompatible grouper.")

        if (value.datar.grouper.size == 1).all():
            # Don't modify base when values are 1-size groups
            # Leave it to broadcast_to() to broadcast to values
            # No need to broadcast the base
            return base


        # Broadcast size-1 groups in base
        return base.datar.regroup(value.datar.grouper)

    if isinstance(base, (SeriesRowwise, TibbleRowwise)):
        if not value.datar.grouper.compatible_with(base.datar.grouper):
            raise ValueError(f"`{name}` has an incompatible grouper.")
        # Don't broadcast rowwise
        return base

    # base is ungrouped
    # DataFrame/Series

    # df >> group_by(f.a) >> mutate(new_col=tibble(x=1, y=f.a))
    #                                              ^^^^^^^^^^
    val_sizes = value.datar.grouper.size

    if base.shape[0] == 1 or (val_sizes == base.shape[0]).all():
        if base.shape[0] == 1:
            repeats = value.shape[0]
        else:
            repeats = val_sizes

        out = base[np.repeat(range(base.shape[0]), repeats), :].datar.group_by(
            *value.datar.grouper.vars,
            sort=value.datar.grouper.sort,
        )
        out.datar.meta["broadcasted"] = True
        return out

    # Otherwise
    raise ValueError(f"Can't recycle a grouped object `{name}` to ungrouped.")


@_broadcast_base.register(Aggregated)
def _(
    value: Aggregated,
    base: pl.DataFrame | pl.Series,
    name: str = None,
) -> pl.DataFrame | pl.Series:
    """Broadcast a DataFrame/Series object to a grouped object

    This is mostly a case when trying to broadcast an aggregated object to
    the original object. For example: `gf >> mutate(f.x / sum(f.x))`

    But `sum(f.x)` could return a Series object that has more than 1 elements
    for a group. Then we need to broadcast `f.x` to match the result.
    """
    # check if base has already broadcasted
    if base.datar.meta.get("broadcasted"):
        return base

    if isinstance(base, TibbleGrouped):
        # Now the index of value works more like grouping data
        if not base.datar.grouper.compatible_with_agg_keys(
            value.datar.agg_keys
        ):
            name = name or name_of(value) or str(value)
            raise ValueError(f"`{name}` is an incompatible aggregated result.")

        # Broadcast size-1 groups in base
        return base.datar.regroup(value.datar.agg_keys)

    if isinstance(base, (SeriesRowwise, TibbleRowwise)):
        if value.shape[0] != 1 and base.shape[0] != 1:
            raise ValueError(f"`{name}` must be size 1, not {value.shape[0]}.")
        # Don't broadcast rowwise
        return base

    if base.shape[0] == 1 and value.shape[0] > 1:
        base = base[[0] * value.shape[0], :]
        base.datar.meta["broadcasted"] = True

    return base


@singledispatch
def broadcast_to(
    value,
    size: int | Grouper,
) -> pl.Series:
    """Broastcast value to expected dimension, the result is a series with
    the given index

    Before calling this function, make sure that the index is already
    broadcasted based on the value. This means that index is always the wider
    then the value if it has one. Also the size for each group is larger then
    the length of the value if value doesn't have an index.

    Args:
        value: Value to be broadcasted
        size: The size the value should be broadcasted to
            - int: broadcast each element to this size
            - Grouper: broadcast the value to each group

    Returns:
        The series with the given index
    """
    # Scalar values don't need to be broadcasted
    # They can be added directly to a tibble
    if is_scalar(value):
        return value

    if len(value) == 1 and is_scalar(value[0]):
        return value[0]

    if isinstance(size, int):
        if len(value) == 0:
            return pl.Series(value)

        if len(value) != size:
            raise ValueError(
                f"Can't broadcast a {len(value)}-size object to {size}."
            )

        return pl.Series(value)

    gsizes = size.size
    # if gsizes.size == 0:
    #     # Do whatever, polars will raise ShapeError
    #     return pl.Series(value)

    if np.unique(gsizes).size != 1 or gsizes[0] != len(value):
        raise ValueError(
            f"Can't broadcast a {len(value)}-size object to {gsizes}."
        )

    return SeriesGrouped(np.tile(value, size.n), grouper=size)


# @broadcast_to.register(Factor)
# def _(
#     value: Factor,
#     index: Index,
#     grouper: Grouper = None,
# ) -> Series:
#     """Broadcast categorical data"""
#     if not grouper:
#         if value.size == 0:
#             return Series(value, index=index)
#         if value.size == 1:
#             return Series(value.repeat(index.size), index=index)
#         # Series will raise the length problem
#         return Series(value, index=index)

#     gsizes = grouper.size()
#     if gsizes.size == 0:
#         return Series(value, index=index)

#     # broadcast value to each group
#     # length of each group is checked in _broadcast_base
#     # A better way to distribute the value to each group?
#     idx = np.concatenate(
#         [grouper.groups[gdata] for gdata in grouper.result_index]
#     )
#     # make np.tile([[3, 4]], 2) to be [[3, 4], [3, 4]],
#     # instead of [[3, 4, 3, 4]]
#     repeats = grouper.ngroups
#     value = Categorical(np.tile(value, repeats), categories=value.categories)
#     return Series(value, index=idx).reindex(index)


# @broadcast_to.register(NDFrame)
# def _(
#     value: NDFrame,
#     index: Index,
#     grouper: Grouper = None,
# ) -> Union[Tibble, Series]:
#     """Broadcast series/dataframe"""
#     if value.index is index:
#         # if it is the same index
#         # e.g. transform results
#         return value

#     if not grouper:
#         # recycle row-1 series/frame
#         if value.index.size == 1 and value.index[0] == 0:
#             value = value.reindex([0] * index.size)
#             value.index = index

#         # empty frame get recycled
#         if isinstance(value, DataFrame) and value.index.size == 0:
#             value.index = index

#         # if not value.index.equals(index):
#         if not value.index.equals(index) and frozenset(
#             value.index
#         ) != frozenset(index):
#             raise ValueError("Value has incompatible index.")

#         if isinstance(value, Series):
#             return Series(value, name=value.name, index=index)

#         return Tibble(value, index=index)

#     # now target is grouped and the value's index is overlapping with the
#     # grouper's index
#     # This is typically an aggregated result to the orignal structure
#     # For example:  f.x.mean() / f.x
#     if _agg_result_compatible(value.index, grouper):

#         if isinstance(value, Series):
#             out = Series(
#                 value,
#                 index=grouper.result_index.take(grouper.group_info[0]),
#                 name=value.name,
#                 copy=False,
#             )
#         else:  # DataFrame
#             out = Tibble(
#                 value,
#                 index=grouper.result_index.take(grouper.group_info[0]),
#                 copy=False,
#             )

#         out.index = index
#         return out

#     if value.index.equals(index):
#         return value

#     raise ValueError("Incompatible value to recycle.")


# @broadcast_to.register(GroupBy)
# def _(
#     value: GroupBy,
#     index: Index,
#     grouper: Grouper = None,
# ) -> Union[Series, Tibble]:
#     """Broadcast pandas grouped object"""
#     if not grouper:
#         raise ValueError(
#             "Can't broadcast grouped object to a non-grouped object."
#         )

#     # Compatibility has been checked in _broadcast_base
#     if isinstance(value, SeriesGrouped):
#         if np.array_equal(grouper.group_info[0], value.grouper.group_info[0]):
#             return Series(
#                 get_obj(value).values, index=index, name=get_obj(value).name
#             )

#         # broadcast size-one groups and
#         # realign the index
#         revalue = _realign_indexes(value, grouper)
#         return Series(revalue, index=index, name=get_obj(value).name)

#     if np.array_equal(grouper.group_info[0], value.grouper.group_info[0]):
#         return Tibble(
#             get_obj(value).values, index=index, columns=get_obj(value).columns
#         )

#     # realign the index
#     revalue = _realign_indexes(value, grouper)
#     return Tibble(revalue, index=index, columns=get_obj(value).columns)


# @broadcast_to.register(TibbleGrouped)
# def _(
#     value: TibbleGrouped,
#     index: Index,
#     grouper: Grouper = None,
# ) -> Tibble:
#     """Broadcast TibbleGrouped object"""
#     return broadcast_to(
#         value._datar["grouped"],
#         index=index,
#         grouper=grouper,
#     )


# @singledispatch
# def _get_index_grouper(value) -> Tuple[Index, Grouper]:
#     return None, None


# @_get_index_grouper.register(TibbleGrouped)
# def _(value):
#     return value.index, value._datar["grouped"].grouper


# @_get_index_grouper.register(NDFrame)
# def _(value):
#     return value.index, None


# @_get_index_grouper.register(GroupBy)
# def _(value):
#     return get_obj(value).index, value.grouper


# @singledispatch
# def _type_priority(value) -> int:
#     return -1


# @_type_priority.register(GroupBy)
# def _(value):
#     return 10


# @_type_priority.register(NDFrame)
# def _(value):
#     return 5


# @_type_priority.register(TibbleGrouped)
# def _(value):
#     return 10


# @singledispatch
# def _ungroup(value):
#     return value


# @_ungroup.register(GroupBy)
# def _(value):
#     return get_obj(value)


# @_ungroup.register(TibbleGrouped)
# def _(value):
#     return value._datar["grouped"].obj


# @singledispatch
# def broadcast2(left, right) -> Tuple[Any, Any, Grouper, bool]:
#     """Broadcast 2 values for operators"""
#     left_pri = _type_priority(left)
#     right_pri = _type_priority(right)
#     if left_pri == right_pri == -1:
#         return left, right, None, False

#     if left_pri > right_pri:
#         left = _broadcast_base(right, left)
#         index, grouper = _get_index_grouper(left)
#         is_rowwise = isinstance(left, TibbleRowwise) or getattr(
#             left, "is_rowwise", False
#         )
#         right = broadcast_to(right, index, grouper)
#     else:
#         right = _broadcast_base(left, right)
#         index, grouper = _get_index_grouper(right)
#         is_rowwise = isinstance(right, TibbleRowwise) or getattr(
#             right, "is_rowwise", False
#         )
#         left = broadcast_to(left, index, grouper)

#     return _ungroup(left), _ungroup(right), grouper, is_rowwise


# @singledispatch
# def init_tibble_from(value, name: str) -> Tibble:
#     """Initialize a tibble from a value"""
#     if is_scalar(value):
#         return Tibble({name: [value]})

#     return Tibble({name: value})


# @init_tibble_from.register(Series)
# def _(value: Series, name: str) -> Tibble:
#     # Deprecate warning, None will be used as series name in the future
#     # So use 0 as default here
#     name = name or value.name or 0
#     return Tibble(value.to_frame(name=name), copy=False)


# @init_tibble_from.register(SeriesGrouped)
# def _(value: SeriesGrouped, name: str) -> TibbleGrouped:

#     return value.to_frame(name).datar.group_by(value.datar.grouper.vars)


# @init_tibble_from.register(DataFrame)
# @init_tibble_from.register(DataFrameGroupBy)
# def _(value: DataFrame | DataFrame, name: str) -> Tibble:
#     from .api.tibble.tibble import as_tibble

#     result = as_tibble(value, __ast_fallback="normal", __backend="polars")

#     if name:
#         result = result.copy()
#         result.columns = [f"{name}${col}" for col in result.columns]
#     return result


# def add_to_tibble(
#     tbl: Tibble,
#     name: str,
#     value: Any,
#     broadcast_tbl: bool = False,
# ) -> "Tibble":
#     """Add data to tibble"""
#     if value is None:
#         return tbl

#     if tbl is None:
#         return init_tibble_from(value, name)

#     if broadcast_tbl:
#         tbl = _broadcast_base(value, tbl, name)

#     if not name and isinstance(value, DataFrame):
#         for col in value.columns:
#             tbl = add_to_tibble(tbl, col, value[col])

#         return tbl

#     tbl[name] = value
#     return tbl
