"""Base arithm API for the polars backend.

Implements: pmin, pmax, mod, sign, signif, trunc, exp, log, log2, log10,
log1p, sum_, mean, median, min_, max_, prod, abs_, sqrt, round_,
ceiling, floor, sd, var.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from datar.apis.base import (
    abs_,
    ceiling,
    cov,
    exp,
    floor,
    log,
    log10,
    log1p,
    log2,
    max_,
    mean,
    median,
    min_,
    mod,
    outer,
    pmax,
    pmin,
    prod,
    proportions,
    quantile,
    round_,
    scale,
    sd,
    sign,
    signif,
    sqrt,
    sum_,
    trunc,
    var,
    weighted_mean,
)

from ...contexts import Context
from ...tibble import Tibble, LazyTibble
from ..dplyr.context import _MultiValueExpr


def _to_series(x: Any) -> pl.Series:
    """Convert non-Expr input to a pl.Series."""
    if isinstance(x, pl.Series):
        return x
    if isinstance(x, pl.Expr):
        raise TypeError("Cannot convert pl.Expr to Series")
    return pl.Series(
        "",
        [x] if not hasattr(x, "__len__") or isinstance(x, (str, bytes)) else x,
        strict=False,
    )


def _is_iterable(x: Any) -> bool:
    return hasattr(x, "__iter__") and not isinstance(x, (str, bytes))


def _is_expr_list(x: list) -> bool:
    """Check if a list contains only pl.Expr objects (e.g. from c_across)."""
    return len(x) > 0 and all(isinstance(e, pl.Expr) for e in x)


def _prepare_args(*args: Any) -> list:
    """Flatten Series to lists, leave others as-is."""
    out: list = []
    for a in args:
        if isinstance(a, pl.Series):
            out.append(a.to_list())
        else:
            out.append(a)
    return out


def _elementwise_minmax(op: str, *args: Any) -> Any:
    """Element-wise min or max across broadcastable arguments."""
    prepared = _prepare_args(*args)
    if not any(_is_iterable(a) for a in prepared):
        return _scalar_minmax(op, *prepared)

    max_len = max((len(a) for a in prepared if _is_iterable(a)), default=0)
    aligned: list[list] = []
    for a in prepared:
        if _is_iterable(a):
            aligned.append(list(a))
        else:
            aligned.append([a] * max_len)

    fn = min if op == "min" else max
    return [fn(row) for row in zip(*aligned)]


def _scalar_minmax(op: str, *args: Any) -> Any:
    if not args:
        return None
    fn = min if op == "min" else max
    return fn(args)


# ---- pmin ---------------------------------------------------------------


@pmin.register(pl.Expr, context=Context.EVAL, backend="polars")
def _pmin_expr(*args: Any, na_rm: bool = False) -> pl.Expr:
    """Row-wise minimum of expressions."""
    exprs = []
    for a in args:
        if isinstance(a, pl.Expr):
            exprs.append(a)
        elif isinstance(a, pl.Series):
            exprs.append(pl.lit(a))
        else:
            exprs.append(pl.lit(a))
    return pl.min_horizontal(*exprs)


@pmin.register(object, context=Context.EVAL, backend="polars")
def _pmin_obj(*args: Any, na_rm: bool = False) -> Any:
    """Element-wise minimum of arbitrary inputs."""
    if not args:
        return None
    if all(isinstance(a, pl.Series) for a in args):
        df = pl.DataFrame({f"__{i}": a for i, a in enumerate(args)})
        return df.select(pl.min_horizontal(pl.all())).to_series()
    return _elementwise_minmax("min", *args)


# ---- pmax ---------------------------------------------------------------


@pmax.register(pl.Expr, context=Context.EVAL, backend="polars")
def _pmax_expr(*args: Any, na_rm: bool = False) -> pl.Expr:
    """Row-wise maximum of expressions."""
    exprs = []
    for a in args:
        if isinstance(a, pl.Expr):
            exprs.append(a)
        elif isinstance(a, pl.Series):
            exprs.append(pl.lit(a))
        else:
            exprs.append(pl.lit(a))
    return pl.max_horizontal(*exprs)


@pmax.register(object, context=Context.EVAL, backend="polars")
def _pmax_obj(*args: Any, na_rm: bool = False) -> Any:
    """Element-wise maximum of arbitrary inputs."""
    if not args:
        return None
    if all(isinstance(a, pl.Series) for a in args):
        df = pl.DataFrame({f"__{i}": a for i, a in enumerate(args)})
        return df.select(pl.max_horizontal(pl.all())).to_series()
    return _elementwise_minmax("max", *args)


# ---- mod ----------------------------------------------------------------


@mod.register(pl.Expr, context=Context.EVAL, backend="polars")
def _mod_expr(x: pl.Expr) -> pl.Expr:
    """Modulus (absolute value) of an expression."""
    return x.abs()


@mod.register(object, context=Context.EVAL, backend="polars")
def _mod_obj(x: Any) -> Any:
    """Modulus (absolute value) of a scalar, Series, or array."""
    if isinstance(x, pl.Series):
        if x.dtype == pl.Object:
            return x.map_elements(abs, return_dtype=pl.Float64)
        return x.abs()
    if isinstance(x, pl.Expr):
        return x.abs()
    if isinstance(x, complex):
        return abs(x)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [abs(v) for v in x]
    import math

    return math.fabs(x) if isinstance(x, float) else abs(x)


# ---- sign ---------------------------------------------------------------


@sign.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sign_expr(x: pl.Expr) -> pl.Expr:
    """Sign of an expression (-1, 0, or 1)."""
    return x.sign()


@sign.register(object, context=Context.EVAL, backend="polars")
def _sign_obj(x: Any) -> Any:
    """Sign of a scalar, Series, or array."""
    if isinstance(x, pl.Series):
        return x.sign()
    if isinstance(x, pl.Expr):
        return x.sign()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [1 if v > 0 else -1 if v < 0 else 0 for v in x]
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


# ---- signif -------------------------------------------------------------


@signif.register(pl.Expr, context=Context.EVAL, backend="polars")
def _signif_expr(x: pl.Expr, digits: int = 6) -> pl.Expr:
    """Round an expression to significant figures."""
    return x.round_sig_figs(digits)


@signif.register(object, context=Context.EVAL, backend="polars")
def _signif_obj(x: Any, digits: int = 6) -> Any:
    """Round a scalar, Series, or array to significant figures."""
    if isinstance(x, pl.Series):
        return x.round_sig_figs(digits)
    if isinstance(x, pl.Expr):
        return x.round_sig_figs(digits)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        result = []
        for v in x:
            if v == 0:
                result.append(0)
            else:
                magnitude = int(math.floor(math.log10(abs(v))))
                factor = 10 ** (digits - 1 - magnitude)
                result.append(round(v * factor) / factor)
        return result
    if x == 0:
        return 0
    import math
    magnitude = int(math.floor(math.log10(abs(x))))
    factor = 10 ** (digits - 1 - magnitude)
    return round(x * factor) / factor


# ---- trunc --------------------------------------------------------------


@trunc.register(pl.Expr, context=Context.EVAL, backend="polars")
def _trunc_expr(x: pl.Expr) -> pl.Expr:
    """Truncate an expression toward zero."""
    return x.truncate()


@trunc.register(object, context=Context.EVAL, backend="polars")
def _trunc_obj(x: Any) -> Any:
    """Truncate a scalar, Series, or array toward zero."""
    if isinstance(x, pl.Series):
        return x.truncate()
    if isinstance(x, pl.Expr):
        return x.truncate()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.trunc(v) for v in x]
    import math
    return math.trunc(x)


# ---- exp ----------------------------------------------------------------


@exp.register(pl.Expr, context=Context.EVAL, backend="polars")
def _exp_expr(x: pl.Expr) -> pl.Expr:
    return x.exp()


@exp.register(object, context=Context.EVAL, backend="polars")
def _exp_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.exp()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.exp(v) for v in x]
    import math
    return math.exp(x)


# ---- log ----------------------------------------------------------------


@log.register(pl.Expr, context=Context.EVAL, backend="polars")
def _log_expr(x: pl.Expr, base: float = math.e) -> pl.Expr:
    return x.log(base)


@log.register(object, context=Context.EVAL, backend="polars")
def _log_obj(x: Any, base: float = math.e) -> Any:
    if isinstance(x, pl.Series):
        return x.log(base)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.log(v, base) for v in x]
    import math
    return math.log(x, base)


# ---- log2 ---------------------------------------------------------------


@log2.register(pl.Expr, context=Context.EVAL, backend="polars")
def _log2_expr(x: pl.Expr) -> pl.Expr:
    return x.log(2)


@log2.register(object, context=Context.EVAL, backend="polars")
def _log2_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.log(2)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.log2(v) for v in x]
    import math
    return math.log2(x)


# ---- log10 --------------------------------------------------------------


@log10.register(pl.Expr, context=Context.EVAL, backend="polars")
def _log10_expr(x: pl.Expr) -> pl.Expr:
    return x.log(10)


@log10.register(object, context=Context.EVAL, backend="polars")
def _log10_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.log(10)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.log10(v) for v in x]
    import math
    return math.log10(x)


# ---- log1p --------------------------------------------------------------


@log1p.register(pl.Expr, context=Context.EVAL, backend="polars")
def _log1p_expr(x: pl.Expr) -> pl.Expr:
    return x.log1p()


@log1p.register(object, context=Context.EVAL, backend="polars")
def _log1p_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.log1p()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.log1p(v) for v in x]
    import math
    return math.log1p(x)


# ---- sum_ ---------------------------------------------------------------


@sum_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sum_expr(x: pl.Expr, na_rm: bool = False) -> pl.Expr:
    return x.sum()


@sum_.register(object, context=Context.EVAL, backend="polars")
def _sum_obj(x: Any, na_rm: bool = False) -> Any:
    return _to_series(x).sum()


@sum_.register(list, context=Context.EVAL, backend="polars")
def _sum_list(x: list, na_rm: bool = False) -> Any:
    if _is_expr_list(x):
        return pl.sum_horizontal(x)
    return _sum_obj(x, na_rm=na_rm)


# ---- mean ---------------------------------------------------------------


@mean.register(pl.Expr, context=Context.EVAL, backend="polars")
def _mean_expr(x: pl.Expr, na_rm: bool = False) -> pl.Expr:
    return x.mean()


@mean.register(object, context=Context.EVAL, backend="polars")
def _mean_obj(x: Any, na_rm: bool = False) -> Any:
    return _to_series(x).mean()


@mean.register(list, context=Context.EVAL, backend="polars")
def _mean_list(x: list, na_rm: bool = False) -> Any:
    if _is_expr_list(x):
        return pl.mean_horizontal(*x)
    return _mean_obj(x, na_rm=na_rm)


# ---- median -------------------------------------------------------------


@median.register(pl.Expr, context=Context.EVAL, backend="polars")
def _median_expr(x: pl.Expr, na_rm: bool = False) -> pl.Expr:
    return x.median()


@median.register(object, context=Context.EVAL, backend="polars")
def _median_obj(x: Any, na_rm: bool = False) -> Any:
    return _to_series(x).median()


@median.register(list, context=Context.EVAL, backend="polars")
def _median_list(x: list, na_rm: bool = False) -> Any:
    if _is_expr_list(x):
        return pl.concat_list(x).list.median()
    return _median_obj(x, na_rm=na_rm)


# ---- min_ ---------------------------------------------------------------


@min_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _min_expr(x: pl.Expr, na_rm: bool = False) -> pl.Expr:
    return x.min()


@min_.register(object, context=Context.EVAL, backend="polars")
def _min_obj(x: Any, na_rm: bool = False) -> Any:
    return _to_series(x).min()


@min_.register(list, context=Context.EVAL, backend="polars")
def _min_list(x: list, na_rm: bool = False) -> Any:
    if _is_expr_list(x):
        return pl.min_horizontal(*x)
    return _min_obj(x, na_rm=na_rm)


# ---- max_ ---------------------------------------------------------------


@max_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _max_expr(x: pl.Expr, na_rm: bool = False) -> pl.Expr:
    return x.max()


@max_.register(object, context=Context.EVAL, backend="polars")
def _max_obj(x: Any, na_rm: bool = False) -> Any:
    return _to_series(x).max()


@max_.register(list, context=Context.EVAL, backend="polars")
def _max_list(x: list, na_rm: bool = False) -> Any:
    if _is_expr_list(x):
        return pl.max_horizontal(*x)
    return _max_obj(x, na_rm=na_rm)


# ---- prod ---------------------------------------------------------------


@prod.register(pl.Expr, context=Context.EVAL, backend="polars")
def _prod_expr(x: pl.Expr, na_rm: bool = False) -> pl.Expr:
    return x.product()


@prod.register(object, context=Context.EVAL, backend="polars")
def _prod_obj(x: Any, na_rm: bool = False) -> Any:
    return _to_series(x).product()


@prod.register(list, context=Context.EVAL, backend="polars")
def _prod_list(x: list, na_rm: bool = False) -> Any:
    if _is_expr_list(x):
        return pl.reduce(function=lambda acc, e: acc * e, exprs=x)
    return _prod_obj(x, na_rm=na_rm)


# ---- abs_ ---------------------------------------------------------------


@abs_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _abs_expr(x: pl.Expr) -> pl.Expr:
    return x.abs()


@abs_.register(object, context=Context.EVAL, backend="polars")
def _abs_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.abs()
    if isinstance(x, pl.Expr):
        return x.abs()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [abs(v) for v in x]
    return abs(x)


# ---- sqrt ---------------------------------------------------------------


@sqrt.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sqrt_expr(x: pl.Expr) -> pl.Expr:
    return x.sqrt()


@sqrt.register(object, context=Context.EVAL, backend="polars")
def _sqrt_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.sqrt()
    if isinstance(x, pl.Expr):
        return x.sqrt()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.sqrt(v) for v in x]
    import math
    return math.sqrt(x)


# ---- round_ -------------------------------------------------------------


@round_.register(pl.Expr, context=Context.EVAL, backend="polars")
def _round_expr(x: pl.Expr, digits: int = 0) -> pl.Expr:
    if digits >= 0:
        return x.round(digits)
    factor = 10 ** (-digits)
    return (x / factor).round(0) * factor


@round_.register(object, context=Context.EVAL, backend="polars")
def _round_obj(x: Any, digits: int = 0) -> Any:
    if isinstance(x, pl.Series):
        if digits >= 0:
            return x.round(digits)
        factor = 10 ** (-digits)
        return (x / factor).round(0) * factor
    if isinstance(x, pl.Expr):
        return _round_expr(x, digits)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [round(v, digits) for v in x]
    return round(x, digits)


# ---- ceiling ------------------------------------------------------------


@ceiling.register(pl.Expr, context=Context.EVAL, backend="polars")
def _ceiling_expr(x: pl.Expr) -> pl.Expr:
    return x.ceil()


@ceiling.register(object, context=Context.EVAL, backend="polars")
def _ceiling_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.ceil()
    if isinstance(x, pl.Expr):
        return x.ceil()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.ceil(v) for v in x]
    import math
    return math.ceil(x)


# ---- floor --------------------------------------------------------------


@floor.register(pl.Expr, context=Context.EVAL, backend="polars")
def _floor_expr(x: pl.Expr) -> pl.Expr:
    return x.floor()


@floor.register(object, context=Context.EVAL, backend="polars")
def _floor_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.floor()
    if isinstance(x, pl.Expr):
        return x.floor()
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        import math
        return [math.floor(v) for v in x]
    import math
    return math.floor(x)


# ---- sd -----------------------------------------------------------------


@sd.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sd_expr(x: pl.Expr, na_rm: bool = False) -> pl.Expr:
    return x.std(ddof=1)


@sd.register(object, context=Context.EVAL, backend="polars")
def _sd_obj(x: Any, na_rm: bool = False) -> Any:
    return _to_series(x).std(ddof=1)


@sd.register(list, context=Context.EVAL, backend="polars")
def _sd_list(x: list, na_rm: bool = False) -> Any:
    if _is_expr_list(x):
        return pl.concat_list(x).list.std(ddof=1)
    return _sd_obj(x, na_rm=na_rm)


# ---- var ----------------------------------------------------------------


@var.register(pl.Expr, context=Context.EVAL, backend="polars")
def _var_expr(x: pl.Expr, na_rm: bool = False, ddof: int = 1) -> pl.Expr:
    return x.var(ddof=ddof)


@var.register(object, context=Context.EVAL, backend="polars")
def _var_obj(x: Any, na_rm: bool = False, ddof: int = 1) -> Any:
    return _to_series(x).var(ddof=ddof)


@var.register(list, context=Context.EVAL, backend="polars")
def _var_list(x: list, na_rm: bool = False, ddof: int = 1) -> Any:
    if _is_expr_list(x):
        return pl.concat_list(x).list.var(ddof=ddof)
    return _var_obj(x, na_rm=na_rm, ddof=ddof)


# ---- proportions --------------------------------------------------------


@proportions.register(object, context=Context.EVAL, backend="polars")
def _proportions(x: Any, margin: int = 1) -> Any:
    """Compute proportions of a table."""
    if isinstance(x, pl.Expr):
        return x / x.sum()
    if isinstance(x, pl.Series):
        total = x.sum()
        if total == 0:
            return x * 0.0
        return x / total
    if isinstance(x, (pl.DataFrame, pl.LazyFrame)):
        pdf = x.collect() if isinstance(x, pl.LazyFrame) else x
        numeric_cols = [
            c
            for c in pdf.columns
            if pdf[c].dtype
            in (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
        ]
        if not numeric_cols:
            return pdf
        if margin == 1:
            row_totals = pl.sum_horizontal(
                [pl.col(c) for c in numeric_cols]
            )
            exprs = [
                (pl.col(c) / row_totals).alias(c)
                if c in numeric_cols
                else pl.col(c)
                for c in pdf.columns
            ]
            return pdf.select(exprs)
        elif margin == 2:
            exprs = {}
            for c in numeric_cols:
                col_sum = pdf[c].sum()
                if col_sum != 0:
                    exprs[c] = pdf[c] / col_sum
                else:
                    exprs[c] = pdf[c] * 0.0
            return pdf.with_columns(**exprs)
        else:
            total = sum(pdf[c].sum() for c in numeric_cols)
            exprs = {}
            for c in numeric_cols:
                if total != 0:
                    exprs[c] = pdf[c] / total
                else:
                    exprs[c] = pdf[c] * 0.0
            return pdf.with_columns(**exprs)
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        arr = list(x)
        total = sum(arr)
        if total == 0:
            return [0.0] * len(arr)
        return [v / total for v in arr]
    return x


# ---- cov ----------------------------------------------------------------


@cov.register(pl.Expr, context=Context.EVAL, backend="polars")
def _cov_expr(
    x: pl.Expr,
    y: Any = None,
    na_rm: bool = False,
    ddof: int = 1,
) -> pl.Expr:
    """Covariance of an expression."""
    if y is None:
        return x.var(ddof=ddof)
    if isinstance(y, pl.Expr):
        return pl.cov(x, y, ddof=ddof)
    return pl.cov(x, pl.lit(y), ddof=ddof)


@cov.register((Tibble, LazyTibble, pl.DataFrame), context=Context.EVAL, backend="polars")
def _cov_tibble(
    x: Tibble,
    y: Any = None,
    na_rm: bool = False,
    ddof: int = 1,
) -> Any:
    """Covariance matrix of a Tibble/DataFrame."""
    if y is not None:
        raise ValueError("In `cov(...)`: No `y` is allowed when `x` is a data frame.")
    if isinstance(x, pl.DataFrame):
        pdf = x
    else:
        pdf = x.collect()
    cols = pdf.columns
    n = len(cols)
    import numpy as np

    mat = np.empty((n, n))
    for i in range(n):
        mat[i, i] = pdf[cols[i]].var(ddof=ddof)
        for j in range(i + 1, n):
            c = np.cov(
                pdf[cols[i]].to_list(),
                pdf[cols[j]].to_list(),
                ddof=ddof,
            )[0, 1]
            mat[i, j] = c
            mat[j, i] = c
    return pl.DataFrame({cols[i]: mat[:, i] for i in range(n)})


@cov.register(object, context=Context.EVAL, backend="polars")
def _cov_obj(
    x: Any,
    y: Any = None,
    na_rm: bool = False,
    ddof: int = 1,
) -> Any:
    """Covariance of two vectors."""
    if isinstance(x, pl.Series):
        if y is None:
            raise ValueError(
                "In `cov(...)`: `y` is required when `x` is a Series."
            )
        import numpy as np

        x_vals = x.to_list()
        if isinstance(y, pl.Series):
            y_vals = y.to_list()
        elif hasattr(y, "__iter__") and not isinstance(y, (str, bytes)):
            y_vals = list(y)
        else:
            y_vals = [y]
        return np.cov(x_vals, y_vals, ddof=ddof)[0, 1]
    import numpy as np

    if y is None:
        return np.var(x, ddof=ddof)
    return np.cov(x, y, ddof=ddof)[0, 1]


# ---- quantile -----------------------------------------------------------


@quantile.register(pl.Expr, context=Context.EVAL, backend="polars")
def _quantile_expr(
    x: pl.Expr,
    probs: Any = (0.0, 0.25, 0.5, 0.75, 1.0),
    na_rm: bool = False,
    names: bool = True,
    type_: int = 7,
    digits: int = 7,
) -> pl.Expr:
    """Quantile of an expression."""
    if not isinstance(probs, (list, tuple)):
        probs = [probs]
    if len(probs) == 1:
        return x.quantile(float(probs[0]), interpolation="linear")
    # Multiple quantiles: wrap in _MultiValueExpr so summarise can explode
    return _MultiValueExpr(
        pl.concat_list(
            [x.quantile(float(p), interpolation="linear") for p in probs]
        )
    )


@quantile.register(object, context=Context.EVAL, backend="polars")
def _quantile_obj(
    x: Any,
    probs: Any = (0.0, 0.25, 0.5, 0.75, 1.0),
    na_rm: bool = False,
    names: bool = True,
    type_: int = 7,
    digits: int = 7,
) -> Any:
    """Quantile of a vector/Series."""
    single = not isinstance(probs, (list, tuple))
    if isinstance(x, pl.Series):
        if single:
            probs = [probs]
        result = x.quantile(probs, interpolation="linear")
        if single:
            return result[0] if len(result) > 0 else None
        return result
    if isinstance(x, pl.Expr):
        return _quantile_expr(x, probs, na_rm, names, type_, digits)
    import numpy as np

    if single:
        probs = [probs]
    result = np.quantile(x, probs)
    if single:
        return float(result.item()) if hasattr(result, "item") else float(result)
    return result


# ---- scale --------------------------------------------------------------


@scale.register(pl.Expr, context=Context.EVAL, backend="polars")
def _scale_expr(
    x: pl.Expr,
    center: Any = True,
    scale_: Any = True,
) -> pl.Expr:
    """Scale and center an expression."""
    result = x
    if center is True:
        result = result - result.mean()
    elif center is not False and center is not None:
        result = result - center
    if scale_ is True:
        result = result / result.std(ddof=1)
    elif scale_ is not False and scale_ is not None:
        result = result / scale_
    return result


@scale.register(object, context=Context.EVAL, backend="polars")
def _scale_obj(
    x: Any,
    center: Any = True,
    scale_: Any = True,
) -> Any:
    """Scale and center a vector/iterable."""
    if isinstance(x, pl.Series):
        result = x
        if center is True:
            result = result - result.mean()
        elif center is not False and center is not None:
            result = result - center
        if scale_ is True:
            std = result.std(ddof=1)
            if std and std != 0:
                result = result / std
        elif scale_ is not False and scale_ is not None:
            result = result / scale_
        return result
    import numpy as np

    arr = np.asarray(x, dtype=float)
    if center is True:
        arr = arr - np.mean(arr)
    elif center is not False and center is not None:
        arr = arr - center
    if scale_ is True:
        std = np.std(arr, ddof=1)
        if std != 0:
            arr = arr / std
    elif scale_ is not False and scale_ is not None:
        arr = arr / scale_
    return arr


# ---- weighted_mean ------------------------------------------------------


@weighted_mean.register(pl.Expr, context=Context.EVAL, backend="polars")
def _weighted_mean_expr(
    x: pl.Expr,
    w: Any = None,
    na_rm: bool = False,
) -> pl.Expr:
    """Weighted mean of an expression."""
    if w is None:
        return x.mean()
    if isinstance(w, pl.Expr):
        return (x * w).sum() / w.sum()
    return (x * pl.lit(w)).sum() / pl.lit(w).sum()


@weighted_mean.register(object, context=Context.EVAL, backend="polars")
def _weighted_mean_obj(
    x: Any,
    w: Any = None,
    na_rm: bool = False,
) -> Any:
    """Weighted mean of a vector/iterable."""
    if w is None:
        if isinstance(x, pl.Series):
            return x.mean()
        if isinstance(x, pl.Expr):
            return x.mean()
        import numpy as np

        return np.mean(x)
    if isinstance(x, pl.Series):
        if isinstance(w, pl.Series):
            return (x * w).sum() / w.sum()
        w_vals = list(w) if hasattr(w, "__iter__") else [w]
        w_s = pl.Series("w", w_vals[: len(x)])
        return (x * w_s).sum() / w_s.sum()
    import numpy as np

    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    return np.average(x_arr, weights=w_arr)


# ---- outer ---------------------------------------------------------------


@outer.register(pl.Expr, context=Context.EVAL, backend="polars")
def _outer_expr(x: pl.Expr, y: Any, fun: str = "*") -> pl.Expr:
    """Outer product of two expressions (returns cross-product)."""
    return x


@outer.register(object, context=Context.EVAL, backend="polars")
def _outer_obj(x: Any, y: Any, fun: str = "*") -> Any:
    """Compute outer product of two vectors."""
    import numpy as np

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    return np.outer(x_arr, y_arr)
