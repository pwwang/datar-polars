"""Operators for the polars backend

Dispatches operator strings (invert, neg, and_, or_, add, sub, …) to
polars-native operations.  Polars Series / Expr already support all standard
Python operators, so arithmetic and logic dispatch is thin.
"""

from __future__ import annotations

import operator as _operator
from collections.abc import Sequence
from typing import Any

from .collections import Collection, Inverted, Negated, Intersect


def _invert(operand: Any) -> Any:
    """~x  —  bitwise NOT or column-set inversion"""
    if isinstance(operand, (slice, Sequence)):
        return Inverted(operand)
    return _operator.invert(operand)


def _neg(operand: Any) -> Any:
    """-x  —  negation or column-set negation"""
    if isinstance(operand, (slice, Sequence)):
        return Negated(operand)
    return _operator.neg(operand)


def _pos(operand: Any) -> Any:
    """+x"""
    return _operator.pos(operand)


def _and_(left: Any, right: Any) -> Any:
    """x & y  —  logical AND or column-set intersection"""
    if isinstance(left, Sequence) or isinstance(right, Sequence):
        return Intersect(left, right)
    return left & right


def _or_(left: Any, right: Any) -> Any:
    """x | y  —  logical OR or column-set union"""
    if isinstance(left, Sequence) or isinstance(right, Sequence):
        return Collection(left, right)
    return left | right


_OP_MAP: dict[str, tuple] = {
    "invert": (_invert, True),
    "neg": (_neg, True),
    "pos": (_pos, True),
    "and_": (_and_, False),
    "or_": (_or_, False),
    "rand_": (lambda x, y: _and_(y, x), False),
    "ror_": (lambda x, y: _or_(y, x), False),
}


def operate(op: str, x: Any, y: Any = None) -> Any:
    """Dispatch an operator string to the appropriate implementation.

    Args:
        op: Operator name (``"add"``, ``"sub"``, ``"invert"``, ``"and_"``,
            etc., matching pipda's OPERATORS registry).
        x: Left operand.
        y: Right operand (``None`` for unary operators).

    Returns:
        Result of applying the operation.
    """
    if op in _OP_MAP:
        func, is_unary = _OP_MAP[op]
        if is_unary:
            return func(x)
        return func(x, y)

    # All remaining operators (add, sub, mul, truediv, floordiv, mod, pow,
    # lshift, rshift, eq, ne, lt, le, gt, ge, etc.) map directly to Python's
    # operator module and are dispatched straight to polars Series.
    op_func = getattr(_operator, op, None)
    if op_func is not None:
        if y is not None:
            return op_func(x, y)
        return op_func(x)

    # Handle reflected (right-hand) operators: radd, rsub, rfloordiv, etc.
    # These arise from expressions like `2 // f.col` → rfloordiv(col, 2).
    # The semantics are: rfloordiv(x, y) == floordiv(y, x)
    if op.startswith("r") and len(op) > 1:
        fwd_func = getattr(_operator, op[1:], None)
        if fwd_func is not None:
            return fwd_func(y, x)

    raise ValueError(f"Unknown operator: {op}")
