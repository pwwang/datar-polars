"""Provides specific contexts for the polars datar backend"""

from enum import Enum

import polars as pl

from pipda.context import (
    ContextBase,
    ContextEval as ContextEvalPipda,
    ContextPending,
    ContextSelect,
)

from .tibble import Tibble, LazyTibble


def _is_real_attr(parent, ref: str) -> bool:
    """Check if `ref` is a real Python attribute defined on the parent's type,
    as opposed to a column name accessed via __getattr__/__getitem__."""
    # Check the class and its bases for a descriptor (property) or method
    for cls in type(parent).__mro__:
        if ref in cls.__dict__:
            return True
    return False


class ContextEval(ContextEvalPipda):
    """Evaluation context for polars DataFrames.

    Returns lazy polars Expressions (pl.col) for column access instead of
    materialized Series, enabling full lazy evaluation across all verbs.
    """

    def _save_used_ref(self, parent, ref, level) -> None:
        """Increment the counters for used references."""
        if (
            not isinstance(parent, (Tibble, LazyTibble))
            or not isinstance(ref, str)
            or level != 1
        ):
            return
        if not hasattr(parent, "_datar") or "used_refs" not in parent._datar:
            return

        parent._datar["used_refs"].add(ref)

    def getitem(self, parent, ref, level):
        """Interpret f[ref] — return lazy Expr for Tibble/LazyTibble parents."""
        self._save_used_ref(parent, ref, level)

        if isinstance(parent, (Tibble, LazyTibble)):
            if isinstance(ref, str) and ref in parent.collect_schema().names():
                return pl.col(ref)
            return super().getitem(parent, ref, level)

        return super().getitem(parent, ref, level)

    def getattr(self, parent, ref, level):
        """Evaluate f.column_name — return lazy Expr for Tibble/LazyTibble parents."""
        if isinstance(parent, dict):
            return self.getitem(parent, ref, level)

        if isinstance(parent, (Tibble, LazyTibble)):
            self._save_used_ref(parent, ref, level)
            # If ref is a real Python attribute but NOT a column name,
            # return the real attribute. This lets .shape, .columns,
            # .dtypes etc. work on materialized DataFrames.
            # If ref is a column name (even if it clashes with a real
            # attribute like .height), prefer pl.col(ref).
            try:
                if _is_real_attr(parent, ref) and ref not in parent.collect_schema().names():
                    return getattr(parent, ref)
            except Exception:
                pass
            return pl.col(ref)

        self._save_used_ref(parent, ref, level)
        return super().getattr(parent, ref, level)

    @property
    def ref(self) -> ContextBase:
        """Defines how `item` in `f[item]` is evaluated."""
        return Context.SELECT  # type: ignore[return-value]


class Context(Enum):
    """Context enumerator for types of contexts."""

    PENDING = ContextPending()
    SELECT = ContextSelect()
    EVAL = ContextEval()
