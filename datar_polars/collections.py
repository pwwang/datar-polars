"""Provide Collection class to mimic `c` from `r-base`"""

from __future__ import annotations

from typing import Any, Iterable

import polars as pl
from pipda import ReferenceAttr, ReferenceItem

from .common import is_scalar, is_null, is_integer

UNMATCHED = object()


class Inverted(object):
    """Wrapper to mark a column set as inverted (negatively selected)."""

    def __init__(self, elems):
        self.elems = elems

    def __repr__(self):
        return f"Inverted({self.elems})"


class Negated(object):
    """Wrapper to mark a column set as negated."""

    def __init__(self, elems):
        self.elems = elems

    def __repr__(self):
        return f"Negated({self.elems})"


class Intersect(object):
    """Wrapper representing intersection of two column sets."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"Intersect({self.left}, {self.right})"


class Collection(list):
    """Mimic the c function in R

    All elements will be flattened. Tracks which elements are not found in the pool.

    Args:
        *args: The elements
        pool: The pool used to match elements against
    """

    def __init__(self, *args: Any, pool: int | Iterable = None) -> None:
        self.elems = args
        self.pool = pool
        self.unmatched: set = set()
        self.error = None
        self.expand(pool=pool)

    def __invert__(self) -> Inverted:
        return Inverted(*self.elems, pool=self.pool)

    def __neg__(self) -> Negated:
        return Negated(*self.elems, pool=self.pool)

    def expand(self, pool: int | Iterable = None) -> Collection:
        """Expand the elements of this collection"""
        if pool is not None:
            self.pool = pool
        else:
            pool = self.pool

        self.unmatched.clear()
        self.error = None

        if pool is not None:
            elems = [
                elem
                for elem in self.elems
                if not is_scalar(elem) or not is_null(elem)
            ]
        else:
            elems = self.elems

        if not elems:
            list.__init__(self, [])
            return self

        expanded = []
        expanded_append = expanded.append
        expanded_extend = expanded.extend
        for elem in elems:
            if isinstance(elem, Collection):
                expanded_extend(elem.expand(pool))
                self.unmatched.update(elem.unmatched)
            elif isinstance(elem, (list, tuple)) and len(elem) == 0:
                # Empty list/tuple — no columns selected (e.g., where()
                # returning no matches).  Skip it.
                pass
            elif isinstance(elem, slice):
                # Expand slice against pool
                if pool is not None:
                    step = elem.step if elem.step is not None else 1
                    pool_len = pool if is_integer(pool) else len(list(pool))
                    # Numeric bounds: use slice.indices() which handles
                    # clamping, negative indices, and reverse ranges properly
                    if isinstance(elem.start, (int, type(None))) and isinstance(
                        elem.stop, (int, type(None))
                    ):
                        start, stop, step = slice(
                            elem.start, elem.stop, elem.step
                        ).indices(pool_len)
                        if not is_integer(pool):
                            # list pool uses inclusive range
                            if step > 0 and start <= stop:
                                stop += 1
                            elif step < 0 and start >= stop:
                                stop -= 1
                        expanded_extend(range(start, stop, step))
                    else:
                        if elem.start is None:
                            start_idx = pool_len - 1 if step < 0 else 0
                        else:
                            start_idx = self._index_from_pool(elem.start)
                        if elem.stop is None:
                            if step < 0:
                                stop_idx = -1
                            elif is_integer(pool):
                                stop_idx = pool_len
                            else:
                                stop_idx = pool_len - 1
                        else:
                            stop_idx = self._index_from_pool(elem.stop)
                        if (
                            start_idx is not UNMATCHED
                            and stop_idx is not UNMATCHED
                        ):
                            if is_integer(pool):
                                expanded_extend(range(start_idx, stop_idx, step))
                            else:
                                expanded_extend(
                                    range(start_idx, stop_idx + (1 if step > 0 else -1), step)
                                )
                        else:
                            self.unmatched.add(elem)
                elif (
                    isinstance(elem.start, (str, ReferenceAttr, ReferenceItem))
                    or isinstance(elem.stop, (str, ReferenceAttr, ReferenceItem))
                ):
                    # Slice with symbolic bounds needs pool to resolve
                    expanded_append(elem)
                else:
                    # Expand numeric slice literally (no pool needed)
                    start = elem.start if elem.start is not None else 0
                    stop = elem.stop if elem.stop is not None else 0
                    step = elem.step
                    inclusive = step in (1, -1) or step is None

                    if step is None:
                        step = 1 if stop >= start else -1

                    if inclusive:
                        stop += step

                    expanded_extend(range(start, stop, step))
            elif isinstance(elem, pl.Expr):
                elem = self._index_from_pool(elem)
                if elem is not UNMATCHED:
                    expanded_append(elem)
            elif isinstance(elem, Iterable) and not isinstance(elem, (str, bytes)):
                # Expand non-string iterables (lists, generators, etc.)
                # Must come BEFORE is_scalar — generators don't have
                # __len__ and would be misclassified as scalars.
                try:
                    exp = Collection(*elem, pool=pool)
                    self.unmatched.update(exp.unmatched)
                    expanded_extend(exp)
                except TypeError:
                    expanded_append(elem)
            elif is_scalar(elem):
                elem = self._index_from_pool(elem)
                if elem is not UNMATCHED:
                    expanded_append(elem)
            else:
                expanded_append(elem)
        list.__init__(self, expanded)
        return self

    def _index_from_pool(self, elem: Any) -> Any:
        """Try to pull the index of the element from the pool"""
        if self.pool is None:
            return elem

        if isinstance(self.pool, int):
            # pool is a length
            if isinstance(elem, int):
                # Negative indices: count from end (R-style)
                if elem < 0:
                    elem = self.pool + elem
                if 0 <= elem < self.pool:
                    return elem
            self.unmatched.add(elem)
            return UNMATCHED

        # Resolve element to a column name string before matching
        key = elem
        if isinstance(elem, (ReferenceAttr, ReferenceItem)):
            key = elem._pipda_ref
        elif isinstance(elem, pl.Expr) and hasattr(elem, "meta"):
            key = elem.meta.output_name()

        # pool is an iterable (e.g. list of column names)
        if isinstance(key, int):
            pool_len = len(list(self.pool))
            # Negative indices: count from end (R-style)
            if key < 0:
                key = pool_len + key
            if 0 <= key < pool_len:
                return key
            self.unmatched.add(elem)
            return UNMATCHED
        try:
            return list(self.pool).index(key)
        except ValueError:
            self.unmatched.add(elem)
            return UNMATCHED

    def __repr__(self) -> str:
        return f"Collection({self.elems})"

    def __str__(self) -> str:
        return list.__repr__(self)


class Negated(Collection):
    """Negated collection, representing collections by `-c(...)` or `-f[...]`"""

    def __repr__(self) -> str:
        return f"Negated({self.elems})"

    def expand(self, pool: int | Iterable = None) -> "Collection":
        """Expand the object"""
        super().expand(pool)

        if self.pool is not None:
            pool_range = (
                range(self.pool) if isinstance(self.pool, int)
                else range(len(self.pool))
            )
            list.__init__(self, [elem for elem in pool_range if elem not in self])
        else:
            list.__init__(self, [-elem for elem in self])
        return self


class Inverted(Collection):
    """Inverted collection, tries to exclude some elements"""

    def __init__(self, *args: Any, pool: int | Iterable = None) -> None:
        super().__init__(*args, pool=pool)

    def __repr__(self) -> str:
        return f"Inverted({self.elems})"

    def expand(self, pool: int | Iterable = None) -> "Collection":
        """Expand the object"""
        if pool is not None:
            self.pool = pool
        if self.pool is None:
            # Can't compute complement without a pool; just resolve elements
            super().expand(pool=None)
            return self
        super().expand(self.pool)
        pool_range = (
            range(self.pool) if is_integer(self.pool) else range(len(self.pool))
        )
        list.__init__(self, [elem for elem in pool_range if elem not in self])
        return self


class Intersect(Collection):
    """Intersect of two collections, designed for `&` operator"""

    def __init__(self, *args: Any, pool: int | Iterable = None) -> None:
        if len(args) != 2:
            raise ValueError("Intersect can only accept two collections.")
        self.elems = args
        self.pool = pool
        self.unmatched = set()
        self.error = None
        # don't expand on init

    def __repr__(self) -> str:
        return f"Intersect({self.elems})"

    def expand(self, pool: int | Iterable = None) -> "Collection":
        """Expand the object"""
        left = Collection(self.elems[0], pool=pool)
        right = frozenset(Collection(self.elems[1], pool=pool))
        list.__init__(self, [elem for elem in left if elem in right])
        return self
