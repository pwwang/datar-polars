"""Provides specific contexts for datar"""
from enum import Enum

from polars import DataFrame, col
from polars.internals.dataframe.groupby import GroupBy
from pipda.context import (
    ContextBase,
    ContextEval as ContextEvalPipda,
    ContextPending,
    ContextSelect,
)
from .tibble import TibbleRowwise


class ContextEvalExpr(ContextEvalPipda):
    """Evaluation context to evaluate pipda expressions into polars expressions
    """

    def _save_used_ref(self, parent, ref, level) -> None:
        """Increments the counters for used references"""
        if level != 1 or "used_refs" not in parent._datar:
            return

        parent._datar["used_refs"].add(ref)

    def getitem(self, parent, ref, level):
        """Interpret f[ref]"""
        if isinstance(ref, str):
            if isinstance(parent, (GroupBy, DataFrame)):
                self._save_used_ref(parent, ref, level)
                return col(ref)
            if isinstance(parent, TibbleRowwise):
                out = col(ref)
                out.rowwise = True
                return out
        return super().getitem(parent, ref, level)

    def getattr(self, parent, ref, level):
        """Evaluate f.a"""
        if isinstance(ref, str):
            if isinstance(parent, (GroupBy, DataFrame)):
                self._save_used_ref(parent, ref, level)
                return col(ref)
            if isinstance(parent, TibbleRowwise):
                out = col(ref)
                out.rowwise = True
                return out

        if isinstance(parent, dict):
            return super().getitem(parent, ref, level)

        return super().getattr(parent, ref, level)

    @property
    def ref(self) -> ContextBase:
        """Defines how `item` in `f[item]` is evaluated.

        This function should return a `ContextBase` object."""
        return Context.SELECT


class ContextEvalData(ContextEvalPipda):
    """Evaluation context to evaluate pipda expressions into real data"""

    def _save_used_ref(self, parent, ref, level) -> None:
        """Increments the counters for used references"""
        if level != 1 or "used_refs" not in parent._datar:
            return

        parent._datar["used_refs"].add(ref)

    def getitem(self, parent, ref, level):
        """Interpret f[ref]"""
        if isinstance(ref, str):
            if isinstance(parent, DataFrame):
                self._save_used_ref(parent, ref, level)
                return parent[ref]

        return super().getitem(parent, ref, level)

    def getattr(self, parent, ref, level):
        """Evaluate f.a"""
        if isinstance(ref, str):
            if isinstance(parent, DataFrame):
                self._save_used_ref(parent, ref, level)
                return parent[ref]

        if isinstance(parent, dict):
            return super().getitem(parent, ref, level)

        return super().getattr(parent, ref, level)

    @property
    def ref(self) -> ContextBase:
        """Defines how `item` in `f[item]` is evaluated.

        This function should return a `ContextBase` object."""
        return Context.SELECT


class Context(Enum):
    """Context enumerator for types of contexts"""

    PENDING = ContextPending()
    SELECT = ContextSelect()
    EVAL_EXPR = ContextEvalExpr()
    EVAL = ContextEvalData()
