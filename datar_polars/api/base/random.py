"""Random number generation for the polars backend.

Implements: set_seed, rnorm, runif, rpois, rbinom, rcauchy, rchisq, rexp.
Uses numpy.random internally (polars lacks distribution-specific RNG).
Returns pl.Series for vector results.
"""

from __future__ import annotations

import random as _random
from typing import Any

import polars as pl

from datar.apis.base import (
    set_seed,
    rnorm,
    runif,
    rpois,
    rbinom,
    rcauchy,
    rchisq,
    rexp,
)

from ...contexts import Context


def _resolve_n(n: Any) -> int:
    """Resolve n to an integer size."""
    if isinstance(n, pl.Expr):
        # Can't resolve Expr statically — use a sensible default
        return 1
    if isinstance(n, pl.Series):
        if len(n) == 0:
            return 0
        return max(n.to_list())
    try:
        return max(n) if hasattr(n, "__iter__") else int(n)
    except Exception:
        return int(n)


# ---- set_seed ----------------------------------------------------------


@set_seed.register(object, backend="polars")
def _set_seed(seed: int):
    import numpy as np

    _random.seed(seed)
    np.random.seed(seed)


# ---- rnorm -------------------------------------------------------------


@rnorm.register(object, context=Context.EVAL, backend="polars")
def _rnorm(n: Any, mean: float = 0, sd: float = 1) -> pl.Series:
    import numpy as np

    size = _resolve_n(n)
    return pl.Series(np.random.normal(mean, sd, size))


# ---- runif -------------------------------------------------------------


@runif.register(object, context=Context.EVAL, backend="polars")
def _runif(n: Any, min_val: float = 0, max_val: float = 1) -> pl.Series:
    import numpy as np

    size = _resolve_n(n)
    return pl.Series(np.random.uniform(min_val, max_val, size))


# ---- rpois -------------------------------------------------------------


@rpois.register(object, context=Context.EVAL, backend="polars")
def _rpois(n: Any, lambda_: float) -> pl.Series:
    import numpy as np

    size = _resolve_n(n)
    return pl.Series(np.random.poisson(lambda_, size))


# ---- rbinom ------------------------------------------------------------


@rbinom.register(object, context=Context.EVAL, backend="polars")
def _rbinom(n: Any, size_param: float, prob: float) -> pl.Series:
    import numpy as np

    n_size = _resolve_n(n)
    return pl.Series(np.random.binomial(size_param, prob, n_size))


# ---- rcauchy -----------------------------------------------------------


@rcauchy.register(object, context=Context.EVAL, backend="polars")
def _rcauchy(
    n: Any, location: float = 0, scale: float = 1
) -> pl.Series:
    import numpy as np

    size = _resolve_n(n)
    # Don't involve the Index from Series in arithmetic
    scale = getattr(scale, "values", scale)
    location = getattr(location, "values", location)
    return pl.Series(np.random.standard_cauchy(size) * scale + location)


# ---- rchisq ------------------------------------------------------------


@rchisq.register(object, context=Context.EVAL, backend="polars")
def _rchisq(n: Any, df: float) -> pl.Series:
    import numpy as np

    size = _resolve_n(n)
    return pl.Series(np.random.chisquare(df, size))


# ---- rexp --------------------------------------------------------------


@rexp.register(object, context=Context.EVAL, backend="polars")
def _rexp(n: Any, rate: float = 1) -> pl.Series:
    import numpy as np

    size = _resolve_n(n)
    return pl.Series(np.random.exponential(1 / rate, size))
