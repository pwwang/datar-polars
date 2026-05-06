"""Tibble API implementations for the polars backend.

Provides: tibble, tibble_, tibble_row, tribble, as_tibble, enframe, deframe,
add_row, add_column, has_rownames, remove_rownames, rownames_to_column,
rowid_to_column, column_to_rownames.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import polars as pl

from datar.core.names import repair_names
from datar.apis.tibble import (
    tibble,
    tibble_,
    tibble_row,
    tribble,
    as_tibble,
    enframe,
    deframe,
    add_row,
    add_column,
    has_rownames,
    remove_rownames,
    rownames_to_column,
    rowid_to_column,
    column_to_rownames,
)
from pipda import ReferenceAttr, ReferenceItem

from ..contexts import Context
from ..tibble import Tibble, LazyTibble, reconstruct_tibble, as_tibble as _as_tibble
from ..common import is_scalar, setdiff, union
from ..utils import name_of, vars_select, replace_na_with_none


# ═══════════════════════════════════════════════════════════════════════════════
# tibble() — construct a Tibble
# ═══════════════════════════════════════════════════════════════════════════════


@tibble.register(backend="polars")
def _tibble(
    *args,
    _name_repair: str | Callable = "check_unique",
    _rows: Optional[int] = None,
    _dtypes=None,
    _drop_index: bool = False,
    _index=None,
    **kwargs,
) -> Tibble:
    """Construct a Tibble from args and kwargs."""
    if not args and not kwargs:
        if _rows is None:
            _rows = 0
        return Tibble(pl.DataFrame([{} for _ in range(_rows)]))

    _DEFERRED = object()  # Sentinel to reserve position in data dict
    data = {}
    deferred = {}
    dtype_map = {}  # Preserve original pl.Series dtypes (Enum/Categorical)

    # Collect all (name, val) pairs to repair names before insertion
    all_items = []
    for val in args:
        all_items.append((name_of(val), val, False))  # positional
    for key, val in kwargs.items():
        all_items.append((key, val, True))  # named kwarg

    # Apply name repair to avoid data-dict key collisions.
    all_names = [name for name, _, _ in all_items]
    arg_names = all_names[:len(args)]  # Original arg names for check_unique
    _repair_is_rename_list = (
        not isinstance(_name_repair, str)
        and not callable(_name_repair)
        and len(_name_repair) != len(all_names)
    )
    if all_names and _name_repair == "check_unique":
        repair_names(all_names, repair="check_unique")
    elif all_names and _name_repair != "minimal" and not _repair_is_rename_list:
        # "unique", "universal", "rename list" (same length), or callable:
        # repair before insertion to avoid dict-key collisions from
        # duplicate arg names. Skip only when _name_repair is a list
        # whose length differs from all_items — it targets final column
        # names (e.g. a DataFrame arg expands to multiple columns).
        repaired = repair_names(all_names, repair=_name_repair)
        for i, new_name in enumerate(repaired):
            all_items[i] = (new_name, all_items[i][1], all_items[i][2])

    for key, val, is_kwarg in all_items:
        if isinstance(val, pl.Series):
            data[key] = val.to_list()
            if isinstance(val.dtype, (pl.Enum, pl.Categorical)):
                dtype_map[key] = val.dtype
        elif isinstance(val, pl.DataFrame):
            if is_kwarg:
                # Named kwarg: store as struct column for unpack()
                data[key] = val.to_struct(key).to_list()
            else:
                # Positional arg: expand columns into the outer frame
                for col in val.columns:
                    data[col] = val.get_column(col).to_list()
        elif isinstance(val, dict):
            data.update(val)
        elif is_scalar(val):
            data[key] = [val]
        elif hasattr(val, "_pipda_eval"):
            data[key] = _DEFERRED
            deferred[key] = val
        elif hasattr(val, "__len__") and not isinstance(val, (str, bytes)):
            try:
                data[key] = list(val)
            except TypeError:
                data[key] = _DEFERRED
                deferred[key] = val
        else:
            data[key] = [val]

    # Evaluate deferred expressions against built data
    has_actual = any(v is not _DEFERRED for v in data.values())
    if deferred and has_actual:
        # Build a temporary DataFrame with known columns
        actual_data = {k: v for k, v in data.items() if v is not _DEFERRED}
        max_len = max((len(v) for v in actual_data.values()), default=1)
        for key in list(actual_data.keys()):
            if len(actual_data[key]) == 1 and max_len > 1:
                actual_data[key] = actual_data[key] * max_len
        for key, val in actual_data.items():
            data[key] = val
        tmp_df = Tibble(pl.DataFrame(actual_data))
        from pipda import evaluate_expr
        for key, expr in deferred.items():
            result = evaluate_expr(expr, tmp_df, Context.EVAL)
            if isinstance(result, pl.Series):
                data[key] = result.to_list()
                if isinstance(result.dtype, (pl.Enum, pl.Categorical)):
                    dtype_map[key] = result.dtype
            elif isinstance(result, pl.Expr):
                # Evaluate against tmp_df
                data[key] = tmp_df.select(result.alias(key)).get_column(key).to_list()
            elif hasattr(result, "__iter__") and not isinstance(result, (str, bytes)):
                data[key] = list(result)
            else:
                data[key] = [result]
    elif deferred:
        # No data yet — can't evaluate expressions
        for key, expr in deferred.items():
            data[key] = [expr]  # Store raw expr

    if data:
        # Recycle scalars to match the longest column length
        max_len = max(
            (len(v) for v in data.values()),
            default=0,
        )
        if max_len > 1:
            for key, val in list(data.items()):
                if len(val) == 1 and max_len > 1:
                    data[key] = val * max_len
        try:
            # Handle mixed scalar/list columns: wrap scalars in lists
            # and use Object dtype to preserve element types.
            series_list = []
            for key, val in data.items():
                val = replace_na_with_none(val)
                if isinstance(val, (list, tuple)):
                    has_lists = any(
                        isinstance(v, (list, tuple)) for v in val
                    )
                    all_lists = all(
                        isinstance(v, (list, tuple)) for v in val
                    )
                    if has_lists and not all_lists:
                        wrapped = [
                            [v] if not isinstance(v, (list, tuple))
                            else list(v)
                            for v in val
                        ]
                        series_list.append(
                            pl.Series(key, wrapped, dtype=pl.Object)
                        )
                    else:
                        s = pl.Series(key, val, strict=False)
                        if key in dtype_map:
                            s = s.cast(dtype_map[key], strict=False)
                        series_list.append(s)
                else:
                    s = pl.Series(key, val, strict=False)
                    if key in dtype_map:
                        s = s.cast(dtype_map[key], strict=False)
                    series_list.append(s)
            df = pl.DataFrame(series_list)
        except pl.exceptions.ShapeError as e:
            raise ValueError(str(e)) from e
    else:
        df = pl.DataFrame([{} for _ in range(_rows or 0)])

    # Apply name repair (handles renaming for non-duplicate cases)
    cols = df.columns
    new_cols = repair_names(list(cols), repair=_name_repair)
    if list(new_cols) != list(cols):
        df = df.rename(dict(zip(cols, new_cols)))

    result = Tibble(df)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# tibble_() — pipeable tibble constructor
# ═══════════════════════════════════════════════════════════════════════════════


@tibble_.register(object, backend="polars")
def _tibble_(
    *args,
    _name_repair: str | Callable = "check_unique",
    _rows: Optional[int] = None,
    _dtypes=None,
    _drop_index: bool = False,
    _index=None,
    **kwargs,
) -> Tibble:
    """Pipeable version of tibble()."""
    return _tibble(
        *args,
        _name_repair=_name_repair,
        _rows=_rows,
        _dtypes=_dtypes,
        _drop_index=_drop_index,
        _index=_index,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# tribble() — row-by-row tibble constructor
# ═══════════════════════════════════════════════════════════════════════════════


@tribble.register(backend="polars")
def _tribble(
    *dummies,
    _name_repair: str | Callable = "minimal",
    _dtypes=None,
) -> Tibble:
    """Create a Tibble using an easier to read row-by-row layout."""
    columns = []
    data = []
    for i, dummy in enumerate(dummies):
        if (
            isinstance(dummy, (ReferenceAttr, ReferenceItem))
            and dummy._pipda_level == 1
        ):
            columns.append(dummy._pipda_ref)
        elif not columns:
            raise ValueError(
                "Must specify at least one column using the `f.<name>` syntax."
            )
        else:
            ncols = len(columns)
            if not data:
                data = [[] for _ in range(ncols)]
            data[i % ncols].append(dummy)

    if not data:
        data = [[] for _ in range(len(columns))]

    if len(data[-1]) != len(data[0]):
        raise ValueError(
            "Data must be rectangular. "
            f"{sum(len(dat) for dat in data)} cells is not an integer "
            f"multiple of {len(columns)} columns."
        )

    pdf = {col: replace_na_with_none(vals) for col, vals in zip(columns, data)}
    df = pl.DataFrame(pdf, strict=False)

    # Apply name repair
    cols = df.columns
    new_cols = repair_names(list(cols), repair=_name_repair)
    if list(new_cols) != list(cols):
        df = df.rename(dict(zip(cols, new_cols)))

    return Tibble(df)


# ═══════════════════════════════════════════════════════════════════════════════
# tibble_row() — single-row tibble
# ═══════════════════════════════════════════════════════════════════════════════


@tibble_row.register(backend="polars")
def _tibble_row(
    *args,
    _name_repair: str | Callable = "check_unique",
    _dtypes=None,
    **kwargs,
) -> Tibble:
    """Construct a Tibble guaranteed to have exactly one row."""
    if not args and not kwargs:
        df = Tibble(pl.DataFrame([{}]))
    else:
        df = _tibble(*args, _name_repair=_name_repair, _dtypes=_dtypes, **kwargs)

    nrows = df.collect().shape[0]
    if nrows > 1:
        raise ValueError("All arguments must be size one, use `[]` to wrap.")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# as_tibble() — convert to Tibble
# ═══════════════════════════════════════════════════════════════════════════════


@as_tibble.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _as_tibble_tbl(df: Tibble) -> Tibble:
    """Tibble input: return as-is."""
    return df


@as_tibble.register(dict, context=Context.EVAL, backend="polars")
def _as_tibble_dict(df: dict) -> Tibble:
    """Convert dict to Tibble."""
    return _as_tibble(df)


@as_tibble.register(pl.DataFrame, context=Context.EVAL, backend="polars")
def _as_tibble_pldf(df: pl.DataFrame) -> Tibble:
    """Convert pl.DataFrame to Tibble."""
    return _as_tibble(df)


@as_tibble.register(pl.LazyFrame, context=Context.EVAL, backend="polars")
def _as_tibble_pllf(df: pl.LazyFrame) -> Tibble:
    """Convert pl.LazyFrame to Tibble."""
    return _as_tibble(df)


@as_tibble.register(object, context=Context.EVAL, backend="polars")
def _as_tibble_obj(df: Any) -> Tibble:
    """Convert any object to Tibble."""
    return _as_tibble(df)


# ═══════════════════════════════════════════════════════════════════════════════
# enframe() — convert named vector/list/dict to Tibble
# ═══════════════════════════════════════════════════════════════════════════════


@enframe.register(object, backend="polars")
def _enframe(x, name="name", value="value") -> Tibble:
    """Convert a mapping or list to a one- or two-column Tibble."""
    if not value:
        raise ValueError("`value` can't be empty.")

    if x is None:
        x = []

    if isinstance(x, dict):
        pass  # dicts are handled below, don't wrap in list
    elif is_scalar(x):
        x = [x]

    x_shape = getattr(x, "shape", ())
    x_dim = len(x_shape)
    if x_dim > 1:
        raise ValueError(
            f"`x` must not have more than one dimension, got {x_dim}."
        )

    if isinstance(x, pl.Series):
        x = x.to_list()

    if not name and isinstance(x, dict):
        x = list(x.values())
    elif name:
        if not isinstance(x, dict):
            names = list(range(len(x)))
            values = x
        else:
            names = list(x.keys())
            values = list(x.values())
        x = [list(pair) for pair in zip(names, values)]

    if name:
        names_list = [pair[0] for pair in x]
        values_list = [pair[1] for pair in x]
        has_lists = any(isinstance(v, (list, tuple)) for v in values_list)
        all_lists = all(isinstance(v, (list, tuple)) for v in values_list)
        if has_lists and not all_lists:
            df = pl.DataFrame({
                name: pl.Series(name, names_list),
                value: pl.Series(value, values_list, dtype=pl.Object),
            })
        else:
            df = pl.DataFrame(x, schema=[name, value], orient="row")
        return Tibble(df)
    else:
        return Tibble(pl.DataFrame(x, schema=[value]))


# ═══════════════════════════════════════════════════════════════════════════════
# deframe() — convert Tibble to dict/list
# ═══════════════════════════════════════════════════════════════════════════════


@deframe.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _deframe_tibble(x: Tibble) -> Any:
    """Convert a two-column Tibble to a dict, or one-column to a list."""
    return _deframe_any(x)


@deframe.register(pl.DataFrame, context=Context.EVAL, backend="polars")
def _deframe_pldf(x: pl.DataFrame) -> Any:
    """Convert a DataFrame to a dict or list."""
    return _deframe_any(x)


@deframe.register(object, context=Context.EVAL, backend="polars")
def _deframe_any(x: Any) -> Any:
    """Convert a data frame to dict (2 cols) or list (1 col)."""
    if isinstance(x, Tibble):
        x = x.collect()
    if isinstance(x, pl.DataFrame):
        if x.shape[1] == 1:
            return x.get_column(x.columns[0]).to_list()
        if x.shape[1] != 2:
            import logging
            logging.warning(
                "`x` must be a one- or two-column data frame in `deframe()`."
            )
        return dict(zip(
            x.get_column(x.columns[0]).to_list(),
            x.get_column(x.columns[1]).to_list(),
        ))
    return x


# ═══════════════════════════════════════════════════════════════════════════════
# add_row() — add rows to a Tibble
# ═══════════════════════════════════════════════════════════════════════════════


def _pos_from_before_after(before, after, length):
    """Get insertion position from before/after."""
    if before is not None and after is not None:
        raise ValueError("Can't specify both `_before` and `_after`.")
    if before is None and after is None:
        return length
    if after is not None:
        return after + 1
    return before


@add_row.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _add_row(
    _data: Tibble,
    *args,
    _before=None,
    _after=None,
    **kwargs,
) -> Tibble:
    """Add one or more rows to a Tibble."""
    if not args and not kwargs:
        # Empty new row
        new_data = {col: [None] for col in _data.collect_schema().names()}
        new_df = pl.DataFrame(new_data)
    else:
        new_tbl = _tibble(*args, **kwargs)
        new_df = new_tbl.collect()

    extra_vars = setdiff(new_df.columns, _data.collect_schema().names())
    if extra_vars:
        raise ValueError(f"New rows can't add columns: {extra_vars}")

    existing = _data.collect()
    pos = _pos_from_before_after(_before, _after, existing.shape[0])

    # Collect parts
    part1 = existing.slice(0, pos)
    part2 = existing.slice(pos, existing.shape[0] - pos)

    result = pl.concat([part1, new_df, part2], how="diagonal_relaxed")
    return reconstruct_tibble(Tibble(result), _data)


# ═══════════════════════════════════════════════════════════════════════════════
# add_column() — add columns to a Tibble
# ═══════════════════════════════════════════════════════════════════════════════


def _check_names_before_after(pos, names):
    """Get position by given index or name."""
    if not isinstance(pos, str):
        return pos
    try:
        return names.index(pos)
    except ValueError:
        raise KeyError(f"Column `{pos}` does not exist.") from None


def _pos_from_before_after_names(before, after, names) -> int:
    """Get position from before/after using column names."""
    if before is not None:
        before = _check_names_before_after(before, names)
    if after is not None:
        after = _check_names_before_after(after, names)
    return _pos_from_before_after(before, after, len(names))


@add_column.register(
    (Tibble, LazyTibble),
    context=Context.EVAL,
    kw_context={"_before": Context.SELECT, "_after": Context.SELECT},
    backend="polars",
)
def _add_column(
    _data: Tibble,
    *args,
    _before=None,
    _after=None,
    _name_repair="check_unique",
    _dtypes=None,
    **kwargs,
) -> Tibble:
    """Add one or more columns to a Tibble."""
    new_tbl = _tibble(*args, _name_repair="minimal", _dtypes=_dtypes, **kwargs)

    if new_tbl.collect().shape[1] == 0:
        return reconstruct_tibble(_data)

    nrows = _data.collect().shape[0]
    new_nrows = new_tbl.collect().shape[0]

    # Broadcast if needed
    if new_nrows == 1 and nrows > 1:
        new_df = new_tbl.collect()
        # Repeat the single row to match existing data height
        collected = pl.concat([new_df] * nrows)
    elif new_nrows == nrows:
        collected = new_tbl.collect()
    else:
        raise ValueError(
            f"Cannot add {new_nrows} rows of new columns to "
            f"{nrows} rows of existing data."
        )

    existing = _data.collect()
    existing_cols = existing.columns
    pos = _pos_from_before_after_names(
        _before, _after, existing_cols
    )

    # Insert at position
    part1 = existing[:, :pos]
    part2 = existing[:, pos:]

    # result = pl.concat([part1, collected, part2], how="horizontal")

    # Apply name repair
    # cols = result.collect_schema().names()
    cols = existing_cols[:pos] + collected.columns + existing_cols[pos:]
    new_cols = repair_names(list(cols), repair=_name_repair)
    if list(new_cols) != list(cols):
        part1.columns = new_cols[:pos]
        collected.columns = new_cols[pos:pos+collected.shape[1]]
        part2.columns = new_cols[pos+collected.shape[1]:]

    result = pl.concat([part1, collected, part2], how="horizontal")
    return reconstruct_tibble(Tibble(result), _data)


# ═══════════════════════════════════════════════════════════════════════════════
# has_rownames() — check if Tibble has row names
# ═══════════════════════════════════════════════════════════════════════════════


@has_rownames.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _has_rownames(_data: Tibble) -> bool:
    """Check if a Tibble has row names.

    In polars, row names are stored in _datar metadata.
    """
    rownames = _data._datar.get("rownames", None)
    return rownames is not None


# ═══════════════════════════════════════════════════════════════════════════════
# remove_rownames() — remove row names
# ═══════════════════════════════════════════════════════════════════════════════


@remove_rownames.register((Tibble, LazyTibble), context=Context.EVAL, backend="polars")
def _remove_rownames(_data: Tibble) -> Tibble:
    """Remove row names from a Tibble."""
    result = reconstruct_tibble(_data)
    result._datar["rownames"] = None
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# rownames_to_column() — add row names as a column
# ═══════════════════════════════════════════════════════════════════════════════


@rownames_to_column.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _rownames_to_column(_data: Tibble, var="rowname") -> Tibble:
    """Add row names as a column."""
    if var in _data.collect_schema().names():
        raise ValueError(f"Column name `{var}` must not be duplicated.")

    rownames = _data._datar.get("rownames", None)
    if rownames is not None:
        # Use stored rownames
        existing = _data.collect()
        rn_col = pl.Series(var, rownames)
        result_df = existing.with_columns(rn_col)
        # Reorder to put rowname first
        cols = [var] + [c for c in existing.columns]
        result_df = result_df.select(cols)
    else:
        # Use integer row numbers
        nrows = _data.collect().shape[0]
        rn_col = pl.Series(var, list(range(nrows)))
        existing = _data.collect()
        result_df = existing.with_columns(rn_col)
        cols = [var] + [c for c in existing.columns]
        result_df = result_df.select(cols)

    result = Tibble(result_df)
    result._datar = dict(_data._datar) if hasattr(_data, "_datar") else {}
    result._datar["rownames"] = None
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# rowid_to_column() — add row IDs as a column
# ═══════════════════════════════════════════════════════════════════════════════


@rowid_to_column.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _rowid_to_column(_data: Tibble, var="rowid") -> Tibble:
    """Add sequential row IDs as a column."""
    if var in _data.collect_schema().names():
        raise ValueError(f"Column name `{var}` must not be duplicated.")

    nrows = _data.collect().shape[0]
    existing = _data.collect()
    result_df = existing.with_columns(pl.Series(var, list(range(nrows))))
    cols = [var] + [c for c in existing.columns]
    result_df = result_df.select(cols)

    result = Tibble(result_df)
    result._datar = dict(_data._datar) if hasattr(_data, "_datar") else {}
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# column_to_rownames() — convert a column to row names
# ═══════════════════════════════════════════════════════════════════════════════


@column_to_rownames.register((Tibble, LazyTibble), context=Context.SELECT, backend="polars")
def _column_to_rownames(_data: Tibble, var="rowname") -> Tibble:
    """Convert a column to row names."""
    if _has_rownames(_data):
        raise ValueError("`_data` must be a data frame without row names.")

    existing = _data.collect()
    if var not in existing.columns:
        raise KeyError(f"Column `{var}` does not exist.")

    rownames = existing.get_column(var).to_list()
    rownames = [str(rn) for rn in rownames]

    # Remove the column
    remaining_cols = [c for c in existing.columns if c != var]
    result_df = existing.select(remaining_cols)

    result = Tibble(result_df)
    result._datar = dict(_data._datar) if hasattr(_data, "_datar") else {}
    result._datar["rownames"] = rownames
    return result
