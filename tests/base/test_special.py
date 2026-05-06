import pytest
import polars as pl

from datar.base import (
    beta,
    lgamma,
    digamma,
    trigamma,
    choose,
    factorial,
    gamma,
    lfactorial,
    lchoose,
    lbeta,
    psigamma,
)
from ..conftest import assert_equal, assert_iterable_equal


def test_beta():
    assert_equal(beta(1, 2), 0.5, approx=True)
    assert_iterable_equal(
        beta([1, 2], [2, 3]), [0.5, 0.08333333333333333], approx=True
    )
    assert_iterable_equal(
        beta(pl.Series([1, 2]), pl.Series([2, 3])),
        [0.5, 0.08333333333333333],
        approx=True,
    )


def test_lgamma():
    assert_equal(lgamma(1), 0, approx=True)
    assert_iterable_equal(lgamma([1, 2]), [0, 0], approx=True)
    assert_iterable_equal(lgamma(pl.Series([1, 2])), [0, 0], approx=True)


def test_digamma():
    assert_equal(digamma(1), -0.5772156649015329, approx=True)
    assert_iterable_equal(
        digamma([1, 2]),
        [-0.5772156649015329, 0.42278433509846714],
        approx=True,
    )
    assert_iterable_equal(
        digamma(pl.Series([1, 2])),
        [-0.5772156649015329, 0.42278433509846714],
        approx=True,
    )


def test_trigamma():
    assert_equal(trigamma(1), 1.6449340668482266, approx=True)
    assert_iterable_equal(
        trigamma([1, 2]), [1.6449340668482266, 0.6449340668482266], approx=True
    )
    assert_iterable_equal(
        trigamma(pl.Series([1, 2])),
        [1.6449340668482266, 0.6449340668482266],
        approx=True,
    )


def test_choose():
    assert_equal(choose(2, 1), 2, approx=True)
    assert_iterable_equal(choose([2, 4], [1, 2]), [2, 6], approx=True)


def test_factorial():
    assert_equal(factorial(1), 1)
    assert_iterable_equal(factorial([1, 4]), [1, 24])


def test_gamma():
    assert_equal(gamma(1), 1)
    assert_iterable_equal(gamma([1, 2]), [1, 1])
    assert_iterable_equal(gamma(pl.Series([1, 2])), [1, 1])


def test_lfactorial():
    import math

    assert_equal(lfactorial(1), 0)
    assert_iterable_equal(lfactorial([1, 2]), [0, math.log(2)])
    assert_iterable_equal(lfactorial(pl.Series([1, 2])), [0, math.log(2)])


def test_lchoose():
    import math

    assert_equal(lchoose(2, 1), math.log(2), approx=True)
    assert_iterable_equal(
        lchoose([2, 4], [1, 2]), [math.log(2), math.log(6)], approx=True
    )


def test_lbeta():
    import math

    assert_equal(lbeta(1, 2), -math.log(2), approx=True)
    assert_iterable_equal(
        lbeta([1, 2], [2, 3]), [-math.log(2), -2.4849066497880004], approx=True
    )


def test_psigamma():
    assert_equal(psigamma(1, 1), 1.6449340668482266, approx=True)
    assert_iterable_equal(
        psigamma([1, 2], 1),
        [1.6449340668482266, 0.6449340668482266],
        approx=True,
    )
