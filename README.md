# datar-polars

The [polars][1] backend for [datar][2] — dplyr-style data manipulation in Python, powered by lazy evaluation.

[![PyPI][3]][4] [![Python][5]][4] [![CI][6]][7] [![Docs][8]][9]

## Installation

```bash
pip install -U datar-polars
# or
pip install -U datar[polars]
```

## Usage

```python
from datar import f
from datar.dplyr import mutate, filter_, summarise, group_by
from datar.tibble import tibble

df = tibble(
    name=["a", "b", "c", "a", "b"],
    value=[1, 2, 3, 4, 5],
)

df >> group_by(f.name) >> summarise(total=f.value.sum())
"""# output:
       name  total
0         a      5
1         b      7
2         c      3
"""

df >> mutate(double=f.value * 2) >> filter_(f.double > 5)
"""# output:
       name  value  double
0         c      3       6
1         a      4       8
2         b      5      10
"""
```

## Features

All operations build lazy [polars][1] query plans. Computation is deferred until you call `.collect()` or materialize the result.

| Category | Status |
|---|---|
| Core dplyr (mutate, filter, select, arrange, summarise, joins, etc.) | Done |
| Group-by operations | Done |
| Row-wise operations | Done |
| Slice, reframe, distinct, count | Done |
| tidyr verbs (pivot, nest, fill, separate, etc.) | Done |
| forcats verbs (factor ordering, levels, etc.) | Done |
| base APIs (arithmetic, string, sequence, cumulative) | Done |
| Tibble, enframe, add_column, add_row | Done |

## Links

- [datar-polars documentation][9]
- [datar documentation][10]
- [datar on GitHub][2]
- [polars documentation][1]

[1]: https://pola.rs
[2]: https://github.com/pwwang/datar
[3]: https://img.shields.io/pypi/v/datar-polars?style=flat-square
[4]: https://pypi.org/project/datar-polars/
[5]: https://img.shields.io/pypi/pyversions/datar-polars?style=flat-square
[6]: https://img.shields.io/github/actions/workflow/status/pwwang/datar-polars/ci.yml?branch=master&style=flat-square
[7]: https://github.com/pwwang/datar-polars/actions
[8]: https://img.shields.io/github/actions/workflow/status/pwwang/datar-polars/docs.yml?branch=master&style=flat-square
[9]: https://pwwang.github.io/datar-polars/
[10]: https://pwwang.github.io/datar/
