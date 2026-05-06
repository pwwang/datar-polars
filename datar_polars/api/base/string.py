"""String functions for the polars backend.

Implements: paste, paste0, toupper, tolower, nchar, nzchar, grep, grepl,
gsub, sub, chartr, endswith, startswith, sprintf, strsplit, strtoi,
substr, substring, trimws.
"""

from __future__ import annotations

import re as _re
from typing import Any

import polars as pl

from datar.apis.base import (
    paste,
    paste0,
    tolower,
    toupper,
    nchar,
    nzchar,
    grep,
    grepl,
    gsub,
    sub,
    chartr,
    endswith,
    startswith,
    sprintf,
    strsplit,
    strtoi,
    substr,
    substring,
    trimws,
)

from ...contexts import Context


# ---- toupper ------------------------------------------------------------


@toupper.register(pl.Expr, context=Context.EVAL, backend="polars")
def _toupper_expr(x: pl.Expr) -> pl.Expr:
    return x.str.to_uppercase()


@toupper.register(object, context=Context.EVAL, backend="polars")
def _toupper_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.str.to_uppercase()
    return str(x).upper()


# ---- tolower ------------------------------------------------------------


@tolower.register(pl.Expr, context=Context.EVAL, backend="polars")
def _tolower_expr(x: pl.Expr) -> pl.Expr:
    return x.str.to_lowercase()


@tolower.register(object, context=Context.EVAL, backend="polars")
def _tolower_obj(x: Any) -> Any:
    if isinstance(x, pl.Series):
        return x.str.to_lowercase()
    return str(x).lower()


# ---- nchar --------------------------------------------------------------


@nchar.register(pl.Expr, context=Context.EVAL, backend="polars")
def _nchar_expr(
    x: pl.Expr,
    type_: str = "width",
    allow_na: bool = True,
    keep_na: bool = False,
    _na_len: int = 2,
) -> pl.Expr:
    return x.str.replace_all("\0", "").str.len_chars()


@nchar.register(object, context=Context.EVAL, backend="polars")
def _nchar_obj(
    x: Any,
    type_: str = "width",
    allow_na: bool = True,
    keep_na: bool = False,
    _na_len: int = 2,
) -> Any:
    if isinstance(x, pl.Series):
        return x.str.replace_all("\0", "").str.len_chars()
    s = str(x)
    if s.endswith("\0"):
        raise ValueError("invalid zero-byte character")
    return len(s.replace("\0", ""))


# ---- nzchar -------------------------------------------------------------


@nzchar.register(pl.Expr, context=Context.EVAL, backend="polars")
def _nzchar_expr(x: pl.Expr, keep_na: bool = False) -> pl.Expr:
    return x.str.len_chars() > 0


@nzchar.register(object, context=Context.EVAL, backend="polars")
def _nzchar_obj(x: Any, keep_na: bool = False) -> Any:
    if isinstance(x, (list, tuple)):
        result = _nzchar_obj(pl.Series(x), keep_na)
        return result.to_list() if isinstance(result, pl.Series) else result
    if isinstance(x, pl.Series):
        return x.str.len_chars() > 0
    return len(str(x)) > 0


# ---- paste --------------------------------------------------------------


@paste.register(object, context=Context.EVAL, backend="polars")
def _paste(*args: Any, sep: str = " ", collapse: Any = None) -> Any:
    """Concatenate strings with separator."""
    has_expr = any(isinstance(a, pl.Expr) for a in args)

    if not has_expr:
        # Standalone mode — handle in Python and return plain list/scalar
        return _paste_plain(args, sep, collapse)

    str_args = []
    for arg in args:
        if isinstance(arg, pl.Expr):
            str_args.append(arg.cast(pl.Utf8))
        elif isinstance(arg, pl.Series):
            str_args.append(pl.lit(arg).cast(pl.Utf8))
        elif isinstance(arg, (list, tuple)):
            str_args.append(pl.lit(pl.Series(arg)).cast(pl.Utf8))
        elif arg is None:
            str_args.append(pl.lit("NA"))
        else:
            str_args.append(pl.lit(str(arg)))

    result = pl.concat_str(str_args, separator=sep) if str_args else pl.lit("")
    if collapse is not None and isinstance(collapse, str):
        result = result.str.concat(collapse)
    return result


def _paste_plain(
    args: tuple, sep: str, collapse: Any | None
) -> list[str] | str:
    """Pure-Python paste for standalone (non-Expr) calls."""
    # Normalize each arg to a list of strings
    normalized: list[list[str]] = []
    max_len = 0
    all_scalar = True
    for arg in args:
        if isinstance(arg, (list, tuple)):
            items = [str(x) for x in arg]
            max_len = max(max_len, len(items))
            normalized.append(items)
            all_scalar = False
        elif isinstance(arg, pl.Series):
            items = [str(x) for x in arg.to_list()]
            max_len = max(max_len, len(items))
            normalized.append(items)
            all_scalar = False
        else:
            normalized.append([str(arg)])
            max_len = max(max_len, 1)

    # Broadcast scalars to max_len
    for i, items in enumerate(normalized):
        if len(items) == 1 and max_len > 1:
            normalized[i] = items * max_len

    result = [sep.join(parts) for parts in zip(*normalized)]
    if collapse is not None:
        return str(collapse).join(result)
    if all_scalar:
        return result[0]
    return result


# ---- paste0 -------------------------------------------------------------


@paste0.register(object, context=Context.EVAL, backend="polars")
def _paste0(*args: Any, collapse: Any = None) -> Any:
    """Concatenate strings without separator."""
    return paste(
        *args, sep="", collapse=collapse,
        __ast_fallback="normal", __backend="polars",
    )


# ---- chartr -------------------------------------------------------------


@chartr.register(pl.Expr, context=Context.EVAL, backend="polars")
def _chartr_expr(old: Any, new: Any, x: pl.Expr) -> pl.Expr:
    """Translate characters in string (like R's chartr)."""
    _trans = str.maketrans(str(old), str(new))
    return x.map_elements(
        lambda s: str(s).translate(_trans), return_dtype=pl.Utf8,
    )


@chartr.register(object, context=Context.EVAL, backend="polars")
def _chartr_obj(old: Any, new: Any, x: Any) -> Any:
    """Translate characters (scalar, Series, or Expr via object dispatch)."""
    _trans = str.maketrans(str(old), str(new))
    if isinstance(x, (pl.Series, pl.Expr)):
        return x.map_elements(
            lambda s: str(s).translate(_trans), return_dtype=pl.Utf8,
        )
    return str(x).translate(_trans)


# ---- grep ---------------------------------------------------------------


@grep.register(pl.Expr, context=Context.EVAL, backend="polars")
def _grep_expr(
    pattern: Any,
    x: pl.Expr,
    ignore_case: bool = False,
    value: bool = False,
    fixed: bool = False,
    invert: bool = False,
) -> pl.Expr:
    """Indices where pattern matches (1-based)."""
    matches = x.str.contains(pattern, literal=fixed)
    if ignore_case:
        # polars contains doesn't have ignore_case; approximate via to_lowercase
        matches = x.str.to_lowercase().str.contains(
            str(pattern).lower(), literal=fixed,
        )
    if invert:
        matches = ~matches
    return matches.arg_true()


@grep.register(object, context=Context.EVAL, backend="polars")
def _grep_obj(
    pattern: Any,
    x: Any,
    ignore_case: bool = False,
    value: bool = False,
    fixed: bool = False,
    invert: bool = False,
) -> Any:
    """Indices where pattern matches (scalar, Series, or Expr)."""
    if isinstance(x, (list, tuple)):
        x = pl.Series(x)
    if isinstance(x, (pl.Series, pl.Expr)):
        matches = x.str.contains(pattern, literal=fixed)
        if ignore_case:
            matches = x.str.to_lowercase().str.contains(
                str(pattern).lower(), literal=fixed,
            )
        if invert:
            matches = ~matches
        return (matches.arg_true()).to_list()
    # scalar
    s = str(x)
    p = str(pattern)
    if ignore_case:
        s, p = s.lower(), p.lower()
    hit = (p in s) if fixed else bool(_re.search(p, s))
    if invert:
        hit = not hit
    return [0] if hit else []


# ---- grepl --------------------------------------------------------------


@grepl.register(pl.Expr, context=Context.EVAL, backend="polars")
def _grepl_expr(
    pattern: Any,
    x: pl.Expr,
    ignore_case: bool = False,
    fixed: bool = False,
) -> pl.Expr:
    """Logical vector indicating pattern matches."""
    if ignore_case:
        return x.str.to_lowercase().str.contains(
            str(pattern).lower(), literal=fixed,
        )
    return x.str.contains(pattern, literal=fixed)


@grepl.register(object, context=Context.EVAL, backend="polars")
def _grepl_obj(
    pattern: Any,
    x: Any,
    ignore_case: bool = False,
    fixed: bool = False,
) -> Any:
    """Logical vector for scalar, Series, or Expr."""
    if isinstance(x, (list, tuple)):
        x = pl.Series(x)
    if isinstance(x, (pl.Series, pl.Expr)):
        if ignore_case:
            return x.str.to_lowercase().str.contains(
                str(pattern).lower(), literal=fixed,
            )
        return x.str.contains(pattern, literal=fixed)
    s = str(x)
    p = str(pattern)
    if ignore_case:
        s, p = s.lower(), p.lower()
    return (p in s) if fixed else bool(_re.search(p, s))


# ---- gsub ---------------------------------------------------------------


def _to_polars_replacement(repl: str) -> str:
    """Convert \\1..\\9 backreferences to $1..$9 for polars regex engine.

    Python's re.sub uses \\N; polars uses $N. Convert unescaped backreference
    backslashes, preserving literal backslashes (\\N in the string means
    literal \\ followed by N, not a backreference).
    """
    return _re.sub(
        r'(?<!\\)((?:\\\\)*)\\(\d)',
        r'\1$\2',
        repl,
    )


@gsub.register(pl.Expr, context=Context.EVAL, backend="polars")
def _gsub_expr(
    pattern: Any,
    replacement: Any,
    x: pl.Expr,
    ignore_case: bool = False,
    fixed: bool = False,
) -> pl.Expr:
    """Replace all occurrences of pattern with replacement."""
    if ignore_case:
        return x.map_elements(
            lambda s: _re.sub(
                str(pattern), str(replacement), str(s),
                flags=_re.IGNORECASE,
            ),
            return_dtype=pl.Utf8,
        )
    if not fixed and isinstance(replacement, str):
        replacement = _to_polars_replacement(replacement)
    return x.str.replace_all(pattern, replacement, literal=fixed)


@gsub.register(object, context=Context.EVAL, backend="polars")
def _gsub_obj(
    pattern: Any,
    replacement: Any,
    x: Any,
    ignore_case: bool = False,
    fixed: bool = False,
) -> Any:
    """Replace all (scalar, Series, or Expr)."""
    if isinstance(x, (list, tuple)):
        result = _gsub_obj(pattern, replacement, pl.Series(x), ignore_case, fixed)
        return result.to_list() if isinstance(result, pl.Series) else result
    if isinstance(x, (pl.Series, pl.Expr)):
        if ignore_case:
            return x.map_elements(
                lambda s: _re.sub(
                    str(pattern), str(replacement), str(s),
                    flags=_re.IGNORECASE,
                ),
                return_dtype=pl.Utf8,
            )
        if not fixed and isinstance(replacement, str):
            replacement = _to_polars_replacement(replacement)
        return x.str.replace_all(pattern, replacement, literal=fixed)
    if fixed:
        return str(x).replace(str(pattern), str(replacement))
    flags = _re.IGNORECASE if ignore_case else 0
    return _re.sub(str(pattern), str(replacement), str(x), flags=flags)


# ---- strsplit -----------------------------------------------------------


@strsplit.register(pl.Expr, context=Context.EVAL, backend="polars")
def _strsplit_expr(
    x: pl.Expr,
    split: Any,
    fixed: bool = False,
    perl: bool = False,
    use_bytes: bool = False,
) -> pl.Expr:
    """Split strings by delimiter."""
    return x.str.split(by=str(split), literal=fixed)


@strsplit.register(object, context=Context.EVAL, backend="polars")
def _strsplit_obj(
    x: Any,
    split: Any,
    fixed: bool = False,
    perl: bool = False,
    use_bytes: bool = False,
) -> Any:
    """Split strings (scalar or Series)."""
    if isinstance(x, (list, tuple)):
        result = _strsplit_obj(pl.Series(x), split, fixed, perl, use_bytes)
        return result.to_list() if isinstance(result, pl.Series) else result
    if isinstance(x, pl.Series):
        return x.str.split(by=str(split), literal=fixed)
    return str(x).split(str(split))


# ---- strtoi -------------------------------------------------------------


@strtoi.register(pl.Expr, context=Context.EVAL, backend="polars")
def _strtoi_expr(x: pl.Expr, base: int = 0) -> pl.Expr:
    """Convert strings to integers."""
    if base == 10 or base == 0:
        return x.str.to_integer()
    return x.map_elements(
        lambda s: int(s, base=base), return_dtype=pl.Int64,
    )


@strtoi.register(object, context=Context.EVAL, backend="polars")
def _strtoi_obj(x: Any, base: int = 0) -> Any:
    """Convert to integer (scalar or Series)."""
    if isinstance(x, pl.Series):
        if base == 10 or base == 0:
            return x.str.to_integer()
        return x.map_elements(
            lambda s: int(s, base=base), return_dtype=pl.Int64,
        )
    return int(str(x), base=base)


# ---- sub ----------------------------------------------------------------


@sub.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sub_expr(
    pattern: Any,
    replacement: Any,
    x: pl.Expr,
    ignore_case: bool = False,
    fixed: bool = False,
) -> pl.Expr:
    """Replace first occurrence of pattern with replacement."""
    if ignore_case:
        return x.map_elements(
            lambda s: _re.sub(
                str(pattern), str(replacement), str(s), count=1,
                flags=_re.IGNORECASE,
            ),
            return_dtype=pl.Utf8,
        )
    return x.str.replace(pattern, replacement, literal=fixed, n=1)


@sub.register(object, context=Context.EVAL, backend="polars")
def _sub_obj(
    pattern: Any,
    replacement: Any,
    x: Any,
    ignore_case: bool = False,
    fixed: bool = False,
) -> Any:
    """Replace first (scalar, Series, or Expr)."""
    if isinstance(x, (list, tuple)):
        result = _sub_obj(pattern, replacement, pl.Series(x), ignore_case, fixed)
        return result.to_list() if isinstance(result, pl.Series) else result
    if isinstance(x, (pl.Series, pl.Expr)):
        if ignore_case:
            return x.map_elements(
                lambda s: _re.sub(
                    str(pattern), str(replacement), str(s), count=1,
                    flags=_re.IGNORECASE,
                ),
                return_dtype=pl.Utf8,
            )
        return x.str.replace(pattern, replacement, literal=fixed, n=1)
    if fixed:
        return str(x).replace(str(pattern), str(replacement), 1)
    flags = _re.IGNORECASE if ignore_case else 0
    return _re.sub(str(pattern), str(replacement), str(x), count=1, flags=flags)


# ---- substr -------------------------------------------------------------


@substr.register(pl.Expr, context=Context.EVAL, backend="polars")
def _substr_expr(x: pl.Expr, start: int, stop: int) -> pl.Expr:
    """Extract substring (0-indexed start, exclusive stop like Python slices)."""
    return x.str.slice(start, stop - start)


@substr.register(object, context=Context.EVAL, backend="polars")
def _substr_obj(x: Any, start: int, stop: int) -> Any:
    """Extract substring (scalar or Series)."""
    if isinstance(x, pl.Series):
        return x.str.slice(start, stop - start)
    return str(x)[start:stop]


# ---- substring ----------------------------------------------------------


@substring.register(pl.Expr, context=Context.EVAL, backend="polars")
def _substring_expr(x: pl.Expr, first: int, last: int | None = None) -> pl.Expr:
    """Extract substring (0-indexed first, exclusive last like Python slices)."""
    if last is None:
        return x.str.slice(first, None)
    return x.str.slice(first, last - first)


@substring.register(object, context=Context.EVAL, backend="polars")
def _substring_obj(x: Any, first: int, last: int | None = None) -> Any:
    """Extract substring (scalar or Series)."""
    if isinstance(x, pl.Series):
        if last is None:
            return x.str.slice(first, None)
        return x.str.slice(first, last - first)
    if last is None:
        return str(x)[first:]
    return str(x)[first:last]


# ---- sprintf ------------------------------------------------------------


@sprintf.register(pl.Expr, context=Context.EVAL, backend="polars")
def _sprintf_expr(fmt: Any, *args: Any) -> pl.Expr:
    """Format strings (C-style sprintf)."""
    py_fmt = _re.sub(r"%[+-]?\d*\.?\d*[a-zA-Z]", "{}", str(fmt))
    exprs = [
        a if isinstance(a, pl.Expr) else pl.lit(a)
        for a in args
    ]
    return pl.format(py_fmt, *exprs)


@sprintf.register(object, context=Context.EVAL, backend="polars")
def _sprintf_obj(fmt: Any, *args: Any) -> Any:
    """Format strings (scalar, Series, or Expr)."""
    has_polars = any(
        isinstance(a, (pl.Series, pl.Expr)) for a in args
    )
    if has_polars:
        # Convert to Expr path
        py_fmt = _re.sub(r"%[+-]?\d*\.?\d*[a-zA-Z]", "{}", str(fmt))
        exprs = []
        for a in args:
            if isinstance(a, pl.Series):
                exprs.append(pl.lit(a.to_list()).cast(pl.Utf8))
            elif isinstance(a, pl.Expr):
                exprs.append(a)
            else:
                exprs.append(pl.lit(a))
        return pl.format(py_fmt, *exprs)
    return str(fmt) % tuple(args)


# ---- trimws -------------------------------------------------------------


@trimws.register(pl.Expr, context=Context.EVAL, backend="polars")
def _trimws_expr(
    x: pl.Expr,
    which: str = "both",
    whitespace: str = " \t",
) -> pl.Expr:
    """Trim whitespace from strings."""
    if which == "left":
        return x.str.strip_chars_start(whitespace)  # type: ignore[arg-type]
    elif which == "right":
        return x.str.strip_chars_end(whitespace)  # type: ignore[arg-type]
    return x.str.strip_chars(whitespace)  # type: ignore[arg-type]


@trimws.register(object, context=Context.EVAL, backend="polars")
def _trimws_obj(
    x: Any,
    which: str = "both",
    whitespace: str = " \t",
) -> Any:
    """Trim whitespace (scalar or Series)."""
    if isinstance(x, pl.Series):
        if which == "left":
            return x.str.strip_chars_start(whitespace)
        elif which == "right":
            return x.str.strip_chars_end(whitespace)
        return x.str.strip_chars(whitespace)
    if which == "left":
        return str(x).lstrip(whitespace)
    elif which == "right":
        return str(x).rstrip(whitespace)
    return str(x).strip(whitespace)


# ---- startswith ---------------------------------------------------------


@startswith.register(pl.Expr, context=Context.EVAL, backend="polars")
def _startswith_expr(x: pl.Expr, prefix: Any) -> pl.Expr:
    """Check if strings start with a prefix."""
    return x.str.starts_with(str(prefix))


@startswith.register(object, context=Context.EVAL, backend="polars")
def _startswith_obj(x: Any, prefix: Any) -> Any:
    """Check prefix (scalar or Series)."""
    if isinstance(x, pl.Series):
        return x.str.starts_with(str(prefix))
    return str(x).startswith(str(prefix))


# ---- endswith -----------------------------------------------------------


@endswith.register(pl.Expr, context=Context.EVAL, backend="polars")
def _endswith_expr(x: pl.Expr, suffix: Any) -> pl.Expr:
    """Check if strings end with a suffix."""
    return x.str.ends_with(str(suffix))


@endswith.register(object, context=Context.EVAL, backend="polars")
def _endswith_obj(x: Any, suffix: Any) -> Any:
    """Check suffix (scalar or Series)."""
    if isinstance(x, pl.Series):
        return x.str.ends_with(str(suffix))
    return str(x).endswith(str(suffix))
