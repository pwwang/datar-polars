# Example Notebooks

Hands-on examples demonstrating datar-polars with Polars backend. Each notebook shows practical usage of the corresponding API.

## Getting Started

Start with the [README notebook](notebooks/readme.ipynb) for a quick introduction.

## dplyr

Core data manipulation verbs:

- [mutate](notebooks/mutate.ipynb) — Create, modify, and delete columns
- [filter](notebooks/filter.ipynb) — Subset rows by conditions
- [select](notebooks/select.ipynb) — Select columns
- [group_by](notebooks/group_by.ipynb) — Group data
- [summarise](notebooks/summarise.ipynb) — Aggregate data
- [arrange](notebooks/arrange.ipynb) — Sort rows
- [join](notebooks/mutate-joins.ipynb) — Join tables
- [across](notebooks/across.ipynb) — Column-wise operations
- [case_when](notebooks/case_when.ipynb) — Conditional values
- [ranking](notebooks/ranking.ipynb) — Rank and order

## tidyr

Data reshaping:

- [pivot_longer / pivot_wider](notebooks/pivot_wider.ipynb) — Reshape data
- [nest / unnest](notebooks/nest.ipynb) — Nested data
- [complete / expand](notebooks/complete.ipynb) — Complete missing combinations
- [fill](notebooks/fill.ipynb) — Fill missing values
- [separate / unite](notebooks/separate.ipynb) — Split and combine columns

## forcats

Factor manipulation:

- [Level value modification](notebooks/forcats_lvl_value.ipynb)
- [Level addition and removal](notebooks/forcats_lvl_addrm.ipynb)
- [Level ordering](notebooks/forcats_lvl_order.ipynb)

## base

Base R-like mathematical and statistical functions:

- [base-funs](notebooks/base-funs.ipynb) — General base functions
- [base-arithmetic](notebooks/base-arithmetic.ipynb) — Arithmetic operations
- [base](notebooks/base.ipynb) — Core base functions
