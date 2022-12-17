import operator
from collections.abc import Sequence
from typing import Any, Callable

from pipda.operator import OPERATORS

from .collections import Collection, Inverted, Negated, Intersect


def _binop(
    opfunc: Callable,
    left: Any,
    right: Any,
    boolean: bool = False,
) -> Any:
    """Binary operator function

    Args:
        opfunc: The operator function
        left: The left operand
        right: The right operand
        boolean: Whether the result is boolean

    Returns:
        The result of the operation
    """
    return opfunc(left, right)


def _arithmetize1(op: str, operand: Any) -> Any:
    """Operator for single operand"""
    op_func = getattr(operator, op)
    return op_func(operand)


def _arithmetize2(op: str, left: Any, right: Any) -> Any:
    """Operator for paired operands"""
    if OPERATORS[op][1]:
        op_func = getattr(operator, op[1:])
        return _binop(op_func, right, left)

    op_func = getattr(operator, op)
    return _binop(op_func, left, right)


def invert(operand: Any) -> Any:
    """Interpretation for ~x"""
    if isinstance(operand, (slice, Sequence)):
        return Inverted(operand)

    return _arithmetize1("invert", operand)


def neg(operand: Any) -> Any:
    """Interpretation for -x"""
    if isinstance(operand, (slice, Sequence)):
        return Negated(operand)

    return _arithmetize1("neg", operand)


def pos(operand: Any) -> Any:
    """Interpretation for +x"""
    return _arithmetize1("pos", operand)


def and_(left: Any, right: Any) -> Any:
    """Interpretation for x & y"""
    if isinstance(left, Sequence) or isinstance(right, Sequence):
        # induce an intersect with Collection
        return Intersect(left, right)

    return _binop(operator.and_, left, right, boolean=True)


def or_(left: Any, right: Any) -> Any:
    """Interpretation for x | y"""
    if isinstance(left, Sequence) or isinstance(right, Sequence):
        # or union?
        return Collection(left, right)

    return _binop(operator.or_, left, right, boolean=True)


def operate(op: str, x: Any, y: Any = None) -> Any:
    """Operator function

    Args:
        op: The operator name
        x: The left operand
        y: The right operand

    Returns:
        The result of the operation
    """
    if op == "invert":
        return invert(x)

    if op == "neg":
        return neg(x)

    if op == "pos":
        return pos(x)

    if op == "and_":
        return and_(x, y)

    if op == "or_":
        return or_(x, y)

    if op == "rand_":
        return and_(y, x)

    if op == "ror_":
        return or_(y, x)

    return _arithmetize2(op, x, y)
