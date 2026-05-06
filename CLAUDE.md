# datar-polars: Polars backend for datar

## Goal
Implement a complete polars backend for the datar framework, supporting polars >= 1.40.0.

## Architecture (Lazy-First)
- **Tibble**: `pl.LazyFrame` subclass in `datar_polars/tibble.py` — always lazy internally, carries `_datar` metadata. Auto-preserves `_datar` via `_from_pyldf` override. Call `.collect()` to materialize.
- **ContextEval**: `datar_polars/contexts.py` — `getattr()` returns `pl.col(ref)` (lazy Expr) instead of materialized Series. `f.x > 5` → `pl.col("x") > 5`.
- **Verbs**: registered for `Tibble` type via `@verb.register(Tibble, context=..., backend="polars")`. Use polars LazyFrame APIs: `.filter()`, `.with_columns()`, `.select()`, `.sort()`, `.group_by().agg()`, `.join()`, `.unique()` — all accept `pl.Expr`.
- **_datar preservation**: Tibble auto-copies `_datar` to new LazyFrame instances. No manual save/restore in verbs. Call `reconstruct_tibble(result, _data)` at end to ensure defaults.
- **Plugin**: Backends register via simplug: `[project.entry-points.datar] polars = "datar_polars:plugin"`

## Key References
- `/root/workspace/datar` — core framework
- `/root/workspace/datar-pandas` — primary backend reference

## Critical Design Principles
1. Always lazy — verbs build plans, `.collect()` materializes
2. `f.column` → `pl.col("column")` (Expr), never materialized
3. `_datar` auto-preserved by Tibble subclass — verbs don't manage it
4. Errors deferred to collect time (lazy semantics)
5. `pl.lit()` for scalars, `pl.col()` for column refs

## Implementation Progress
- ✅ Core utilities (tibble, contexts, broadcast, collections, common, operators)
- ✅ Core dplyr verbs (mutate, filter, select, arrange, group_by, summarise, join, distinct)
- ✅ Tests: 223 passed, 26 skipped, 2 xfailed
- ⬜ Remaining dplyr (across, pick, if_else, lead_lag, rank, recode, rows, sets, glimpse)
- ⬜ Base APIs (arithmetic, string, sequence, cumulative, etc.)
- ⬜ Tidyr verbs
- ⬜ Forcats verbs

## Environment
- Python venv: `/root/workspace/.venv` (polars 1.40.1, datar 0.15.17)
- Run tests: `uv run pytest`
- When testing a piece of code, save it to a temporary file and run with `uv run python temp.py` to ensure AST is tracked.
- Do NOT run code directly using `uv run python -c "..."` or in an interactive shell, as this will bypass the AST tracking and lead to incorrect test results.

## File creation rules
- Line length: 88 chars (black default)
- Type hints required on all public functions
- Imports: use `from ... import ...` patterns consistent with datar-pandas
