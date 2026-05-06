# datar-polars API Surface Audit

## Executive Summary

- **Total API surface**: 369 functions across 6 API groups
- **Implemented in datar-polars**: 24 functions (~6.5%)
- **Remaining**: 345 functions
- **Current state**: Only dplyr core verbs + misc (lazy/collect) are done

---

## 1. Complete API Inventory

### 1.1 dplyr API (115 functions)

#### Core Data Verbs (registered to Tibble)

| # | Function | Status | Notes |
|---|----------|--------|-------|
| 1 | `mutate` | ✅ | Fully implemented |
| 2 | `transmute` | ✅ | Delegates to mutate with _keep="none" |
| 3 | `select` | ✅ | Fully implemented |
| 4 | `filter_` | ✅ | Fully implemented |
| 5 | `arrange` | ✅ | Fully implemented |
| 6 | `distinct` | ✅ | Fully implemented |
| 7 | `summarise` | ✅ | Fully implemented |
| 8 | `reframe` | ✅ | Fully implemented |
| 9 | `group_by` | ✅ | Fully implemented |
| 10 | `ungroup` | ✅ | Fully implemented |
| 11 | `rowwise` | ✅ | Fully implemented |
| 12 | `group_by_drop_default` | ✅ | Fully implemented |
| 13 | `group_vars` | ✅ | Fully implemented |
| 14 | `inner_join` | ✅ | Fully implemented |
| 15 | `left_join` | ✅ | Fully implemented |
| 16 | `right_join` | ✅ | Fully implemented |
| 17 | `full_join` | ✅ | Fully implemented |
| 18 | `semi_join` | ✅ | Fully implemented |
| 19 | `anti_join` | ✅ | Fully implemented |
| 20 | `nest_join` | ✅ | Fully implemented |
| 21 | `cross_join` | ✅ | Fully implemented |
| 22 | `desc` | ✅ | Series-level only |

#### Remaining dplyr Verbs (93 functions)

| # | Function | Priority | File count | Notes |
|---|----------|----------|------------|-------|
| 23 | `pick` | HIGH | 1 | Column selection helper |
| 24 | `across` | HIGH | 1 | Apply function across columns |
| 25 | `c_across` | MEDIUM | 1 | Column selection in rowwise |
| 26 | `if_any` | HIGH | 1 | Used with filter/across |
| 27 | `if_all` | HIGH | 1 | Used with filter/across |
| 28 | `symdiff` | LOW | 1 | Set difference |
| 29 | `bind_rows` | HIGH | 1 | Concatenate data frames |
| 30 | `bind_cols` | MEDIUM | 1 | Column-bind data frames |
| 31 | `cur_column` | HIGH | 1 | Current column name in across |
| 32 | `cur_data` | HIGH | 1 | Current data in grouped ops |
| 33 | `n` | HIGH | 1 | Row count (used everywhere) |
| 34 | `cur_data_all` | MEDIUM | 1 | Current data with all columns |
| 35 | `cur_group` | MEDIUM | 1 | Current group dataframe |
| 36 | `cur_group_id` | MEDIUM | 1 | Current group integer id |
| 37 | `cur_group_rows` | MEDIUM | 1 | Current group row indices |
| 38 | `count` | HIGH | 1 | Count observations by group |
| 39 | `tally` | MEDIUM | 1 | Count rows by group |
| 40 | `add_count` | MEDIUM | 1 | Add count column |
| 41 | `add_tally` | MEDIUM | 1 | Add tally column |
| 42 | `n_distinct` | MEDIUM | 1 | Count distinct values |
| 43 | `glimpse` | LOW | 1 | Preview data (nice-to-have) |
| 44 | `slice_` | HIGH | 1 | Row indexing |
| 45 | `slice_head` | HIGH | 1 | First N rows |
| 46 | `slice_tail` | HIGH | 1 | Last N rows |
| 47 | `slice_sample` | HIGH | 1 | Random rows |
| 48 | `slice_min` | HIGH | 1 | Rows with min values |
| 49 | `slice_max` | HIGH | 1 | Rows with max values |
| 50 | `between` | MEDIUM | 1 | Value in range check |
| 51 | `cummean` | MEDIUM | 1 | Cumulative mean |
| 52 | `cumall` | MEDIUM | 1 | Cumulative all |
| 53 | `cumany` | MEDIUM | 1 | Cumulative any |
| 54 | `coalesce` | MEDIUM | 1 | First non-NA value |
| 55 | `consecutive_id` | MEDIUM | 1 | Consecutive group ids |
| 56 | `na_if` | MEDIUM | 1 | Replace value with NA |
| 57 | `near` | MEDIUM | 1 | Safe floating point equality |
| 58 | `nth` | HIGH | 1 | Extract nth element |
| 59 | `first` | HIGH | 1 | First element |
| 60 | `last` | HIGH | 1 | Last element |
| 61 | `group_indices` | MEDIUM | 1 | Integer group ids |
| 62 | `group_keys` | MEDIUM | 1 | Group key dataframes |
| 63 | `group_size` | MEDIUM | 1 | Size of each group |
| 64 | `group_rows` | MEDIUM | 1 | Row indices per group |
| 65 | `group_cols` | MEDIUM | 1 | Group column names |
| 66 | `group_data` | MEDIUM | 1 | Full group metadata |
| 67 | `n_groups` | MEDIUM | 1 | Number of groups |
| 68 | `group_map` | MEDIUM | 1 | Apply function per group |
| 69 | `group_modify` | MEDIUM | 1 | Modify each group |
| 70 | `group_split` | MEDIUM | 1 | Split into list of frames |
| 71 | `group_trim` | LOW | 1 | Remove empty groups |
| 72 | `group_walk` | LOW | 1 | Side-effect per group |
| 73 | `with_groups` | MEDIUM | 1 | Temporary regroup |
| 74 | `if_else` | HIGH | 1 | Vectorized if/else |
| 75 | `case_match` | HIGH | 1 | Multi-value switch |
| 76 | `case_when` | HIGH | 1 | Vectorized case statement |
| 77 | `lead` | HIGH | 1 | Shift forward |
| 78 | `lag` | HIGH | 1 | Shift backward |
| 79 | `order_by` | MEDIUM | 1 | Control window order |
| 80 | `with_order` | MEDIUM | 1 | Apply fn with custom order |
| 81 | `pull` | HIGH | 1 | Extract column as series |
| 82 | `row_number` | HIGH | 1 | Row number (wrapper) |
| 83 | `row_number_` | HIGH | 1 | Row number (impl) |
| 84 | `ntile` | MEDIUM | 1 | N-tile buckets (wrapper) |
| 85 | `ntile_` | MEDIUM | 1 | N-tile (impl) |
| 86 | `min_rank` | HIGH | 1 | Min rank (wrapper) |
| 87 | `min_rank_` | HIGH | 1 | Min rank (impl) |
| 88 | `dense_rank` | MEDIUM | 1 | Dense rank (wrapper) |
| 89 | `dense_rank_` | MEDIUM | 1 | Dense rank (impl) |
| 90 | `percent_rank` | MEDIUM | 1 | Percent rank (wrapper) |
| 91 | `percent_rank_` | MEDIUM | 1 | Percent rank (impl) |
| 92 | `cume_dist` | MEDIUM | 1 | Cumulative distribution (w) |
| 93 | `cume_dist_` | MEDIUM | 1 | Cumulative distribution (i) |
| 94 | `recode` | MEDIUM | 1 | Replace values in vector |
| 95 | `recode_factor` | MEDIUM | 1 | Replace factor levels |
| 96 | `relocate` | HIGH | 1 | Move columns |
| 97 | `rename` | HIGH | 1 | Rename columns |
| 98 | `rename_with` | HIGH | 1 | Rename with function |
| 99 | `rows_insert` | MEDIUM | 1 | Insert rows by key |
| 100 | `rows_update` | MEDIUM | 1 | Update rows by key |
| 101 | `rows_patch` | MEDIUM | 1 | Patch NA values by key |
| 102 | `rows_upsert` | MEDIUM | 1 | Upsert rows by key |
| 103 | `rows_delete` | MEDIUM | 1 | Delete rows by key |
| 104 | `rows_append` | MEDIUM | 1 | Append rows |
| 105 | `union_all` | MEDIUM | 1 | Combine two datasets |
| 106 | `where` | MEDIUM | 1 | Column selection helper |
| 107 | `everything` | MEDIUM | 1 | Select all columns |
| 108 | `last_col` | MEDIUM | 1 | Select last column |
| 109 | `starts_with` | MEDIUM | 1 | Column name pattern |
| 110 | `ends_with` | MEDIUM | 1 | Column name pattern |
| 111 | `contains` | MEDIUM | 1 | Column name pattern |
| 112 | `matches` | MEDIUM | 1 | Column name regex |
| 113 | `num_range` | MEDIUM | 1 | Column name range |
| 114 | `all_of` | MEDIUM | 1 | Strict column selection |
| 115 | `any_of` | MEDIUM | 1 | Loose column selection |

**dplyr subtotal: 22 done, 93 remaining**

### 1.2 base API (183 functions)

Categorized by module:

#### Arithmetic/Statistics (31 functions)
`ceiling`, `cov`, `floor`, `mean`, `median`, `pmax`, `pmin`, `sqrt`, `var`, `scale`,
`col_sums`, `col_means`, `col_sds`, `col_medians`, `row_sums`, `row_means`, `row_sds`, `row_medians`,
`min_`, `max_`, `round_`, `sum_`, `abs_`, `prod`, `sign`, `signif`, `trunc`, `sd`, `weighted_mean`, `quantile`, `proportions`

**Status: ⬜ ALL NOT STARTED** (0/31)

#### Log/Exp (7 functions)
`exp`, `log`, `log2`, `log10`, `log1p`

**Status: ⬜ ALL NOT STARTED** (0/5)
Note: `log` counted again below with variants.

Actually, let me recategorize:
- `exp`, `log`, `log2`, `log10`, `log1p` = 5

#### Bessel Functions (4 functions)
`bessel_i`, `bessel_j`, `bessel_k`, `bessel_y`

**Status: ⬜ ALL NOT STARTED** (0/4)

#### Type Conversion/Coercion (12 functions)
`as_double`, `as_integer`, `as_logical`, `as_character`, `as_factor`, `as_ordered`,
`as_date`, `as_numeric`, `as_complex`, `as_null`

Plus type-checking: `is_integer`, `is_numeric`, `is_double`, `is_logical`, `is_character`,
`is_complex`, `is_factor`, `is_ordered`, `is_atomic`, `is_null`, `is_na`, `is_finite`, `is_infinite`,
`is_true`, `is_false`, `any_na`

**Status: ⬜ ALL NOT STARTED** (0/~26)

#### Complex Numbers (6 functions)
`arg`, `conj`, `mod`, `re_`, `im`

**Status: ⬜ ALL NOT STARTED** (0/5)

#### Cumulative Functions (4 functions)
`cummax`, `cummin`, `cumprod`, `cumsum`

**Status: ⬜ ALL NOT STARTED** (0/4)

#### Factor/Level Functions (8 functions)
`droplevels`, `levels`, `set_levels`, `is_factor`, `is_ordered`, `nlevels`,
`factor`, `ordered`, `cut`

**Status: ⬜ ALL NOT STARTED** (0/9)

#### Sequence Functions (5 functions)
`seq`, `seq_along`, `seq_len`, `diff`, `rep`

**Status: ⬜ ALL NOT STARTED** (0/5)

#### Random Number Generation (8 functions)
`set_seed`, `rnorm`, `runif`, `rpois`, `rbinom`, `rcauchy`, `rchisq`, `rexp`

**Status: ⬜ ALL NOT STARTED** (0/8)

#### String Functions (19 functions)
`grep`, `grepl`, `sub`, `gsub`, `strsplit`, `paste`, `paste0`, `sprintf`,
`substr`, `substring`, `startswith`, `endswith`, `strtoi`, `trimws`,
`toupper`, `tolower`, `chartr`, `nchar`, `nzchar`

**Status: ⬜ ALL NOT STARTED** (0/19)

#### Trigonometry (16 functions)
`acos`, `acosh`, `asin`, `asinh`, `atan`, `atanh`, `cos`, `cosh`, `cospi`,
`sin`, `sinh`, `sinpi`, `tan`, `tanh`, `tanpi`, `atan2`

**Status: ⬜ ALL NOT STARTED** (0/16)

#### Special Math (13 functions)
`beta`, `lgamma`, `digamma`, `trigamma`, `choose`, `factorial`, `gamma`,
`lfactorial`, `lchoose`, `lbeta`, `psigamma`

**Status: ⬜ ALL NOT STARTED** (0/11)

#### Set/Vector Operations (15 functions)
`match`, `order`, `sort`, `rev`, `sample`, `c_`, `length`, `lengths`,
`append`, `intersect`, `setdiff`, `setequal`, `unique`, `union`, `duplicated`

**Status: ⬜ ALL NOT STARTED** (0/15)

#### Data Frame Inspection (12 functions)
`colnames`, `set_colnames`, `rownames`, `set_rownames`, `dim`, `ncol`, `nrow`,
`head`, `tail`, `complete_cases`, `diag`, `max_col`

**Status: ⬜ ALL NOT STARTED** (0/12)

#### Table Functions (2 functions)
`table`, `tabulate`

**Status: ⬜ ALL NOT STARTED** (0/2)

#### Other (7 functions)
`make_names`, `make_unique`, `rank`, `identity`, `expand_grid`, `outer`, `which`, `which_max`, `which_min`, `any_`, `all_`, `t`

**Status: ⬜ ALL NOT STARTED** (0/~12)

**base subtotal: 0 done, 183 remaining**

### 1.3 tibble API (14 functions)

| # | Function | Priority |
|---|----------|----------|
| 1 | `tibble` | HIGH (BLOCKER) |
| 2 | `tibble_` | MEDIUM |
| 3 | `tribble` | MEDIUM |
| 4 | `tibble_row` | MEDIUM |
| 5 | `as_tibble` | HIGH (BLOCKER) |
| 6 | `enframe` | MEDIUM |
| 7 | `deframe` | MEDIUM |
| 8 | `add_row` | MEDIUM |
| 9 | `add_column` | MEDIUM |
| 10 | `has_rownames` | LOW |
| 11 | `remove_rownames` | LOW |
| 12 | `rownames_to_column` | LOW |
| 13 | `rowid_to_column` | LOW |
| 14 | `column_to_rownames` | LOW |

**Status: ⬜ ALL NOT STARTED** (0/14)

NOTE: `as_tibble` and `tibble` already exist in `datar_polars/tibble.py` as the Tibble class and constructor, but not registered as API functions.

### 1.4 tidyr API (21 functions)

| # | Function | Priority |
|---|----------|----------|
| 1 | `full_seq` | MEDIUM |
| 2 | `chop` | MEDIUM |
| 3 | `unchop` | MEDIUM |
| 4 | `nest` | MEDIUM |
| 5 | `unnest` | HIGH |
| 6 | `pack` | LOW |
| 7 | `unpack` | LOW |
| 8 | `expand` | HIGH |
| 9 | `nesting` | MEDIUM |
| 10 | `crossing` | MEDIUM |
| 11 | `complete` | HIGH |
| 12 | `drop_na` | HIGH |
| 13 | `extract` | MEDIUM |
| 14 | `fill` | HIGH |
| 15 | `pivot_longer` | HIGH |
| 16 | `pivot_wider` | HIGH |
| 17 | `separate` | HIGH |
| 18 | `separate_rows` | MEDIUM |
| 19 | `uncount` | MEDIUM |
| 20 | `unite` | MEDIUM |
| 21 | `replace_na` | HIGH |

**Status: ⬜ ALL NOT STARTED** (0/21)

### 1.5 forcats API (34 functions)

| # | Function | Priority |
|---|----------|----------|
| 1 | `fct_relevel` | MEDIUM |
| 2 | `fct_inorder` | MEDIUM |
| 3 | `fct_infreq` | MEDIUM |
| 4 | `fct_inseq` | MEDIUM |
| 5 | `fct_reorder` | MEDIUM |
| 6 | `fct_reorder2` | LOW |
| 7 | `fct_shuffle` | LOW |
| 8 | `fct_rev` | MEDIUM |
| 9 | `fct_shift` | LOW |
| 10 | `first2` | LOW |
| 11 | `last2` | LOW |
| 12 | `fct_anon` | LOW |
| 13 | `fct_recode` | MEDIUM |
| 14 | `fct_collapse` | MEDIUM |
| 15 | `fct_lump` | MEDIUM |
| 16 | `fct_lump_min` | LOW |
| 17 | `fct_lump_prop` | LOW |
| 18 | `fct_lump_n` | LOW |
| 19 | `fct_lump_lowfreq` | LOW |
| 20 | `fct_other` | MEDIUM |
| 21 | `fct_relabel` | LOW |
| 22 | `fct_expand` | LOW |
| 23 | `fct_explicit_na` | MEDIUM |
| 24 | `fct_drop` | MEDIUM |
| 25 | `fct_unify` | LOW |
| 26 | `fct_c` | MEDIUM |
| 27 | `fct_cross` | LOW |
| 28 | `fct_count` | MEDIUM |
| 29 | `fct_match` | LOW |
| 30 | `fct_unique` | MEDIUM |
| 31 | `lvls_reorder` | LOW |
| 32 | `lvls_revalue` | LOW |
| 33 | `lvls_expand` | LOW |
| 34 | `lvls_union` | LOW |

**Status: ⬜ ALL NOT STARTED** (0/34)

### 1.6 misc API (2 functions)

| # | Function | Status |
|---|----------|--------|
| 1 | `lazy` | ✅ |
| 2 | `collect` | ✅ |

**misc subtotal: 2 done, 0 remaining**

---

## 2. Plugin Hook Status

| Hook | datar-numpy | datar-pandas | datar-polars |
|------|------------|-------------|-------------|
| `setup()` | ❌ | ✅ | ✅ |
| `get_versions()` | ✅ | ✅ | ✅ |
| `load_dataset(name, metadata)` | ❌ | ✅ | ✅ |
| `base_api()` | ✅ (default impls) | ✅ (pandas impls) | ⬜ (empty) |
| `dplyr_api()` | ❌ | ✅ | 🔶 (partial) |
| `tibble_api()` | ❌ | ✅ | ⬜ (empty) |
| `tidyr_api()` | ❌ | ✅ | ⬜ (empty) |
| `forcats_api()` | ❌ | ✅ | ⬜ (empty) |
| `misc_api()` | ❌ | ✅ | ✅ |
| `c_getitem(item)` | ✅ | ✅ | ✅ |
| `operate(op, x, y)` | ❌ | ✅ | ✅ |

---

## 3. Implementation by Backend Comparison

| Backend | dplyr | base | tibble | tidyr | forcats | misc | Total |
|---------|-------|------|--------|-------|---------|------|-------|
| datar-numpy (default) | 0 | 183 | 0 | 0 | 0 | 0 | 183 (~50%) |
| datar-pandas (ref) | ~110 | ~183 | ~14 | ~21 | ~34 | ~6 | ~368 (~100%) |
| datar-polars (current) | 22 | 0 | 0 | 0 | 0 | 2 | 24 (~6.5%) |

Note: datar-numpy provides default/base implementations for all 183 base API functions (the math, trig, string ops that work on plain numpy arrays). datar-pandas overrides ALL of these for pandas Series/DataFrame input plus adds dplyr/tidyr/forcats/tibble.

---

## 4. Prioritized Implementation Order

### Phase 1: Complete dplyr (BLOCKING for real-world use)
Estimated: ~93 verbs, 15-20 files

**Priority A (core/data verbs - used in every pipeline):**

| Order | Group | Verbs | Why |
|-------|-------|-------|-----|
| 1 | Context helpers | `n`, `cur_data`, `cur_column`, `cur_group_id` | Required by summarise/mutate |
| 2 | Window functions | `row_number`, `min_rank`, `dense_rank`, `percent_rank`, `cume_dist`, `ntile` | Required by arrange/mutate |
| 3 | Lead/lag | `lead`, `lag` | Required by mutate for time-series |
| 4 | if_else/case | `if_else`, `case_when`, `case_match` | Required by mutate |
| 5 | Column selection helpers | `everything`, `starts_with`, `ends_with`, `contains`, `matches`, `all_of`, `any_of`, `num_range`, `where`, `last_col` | Required by select/relocate |
| 6 | across/pick | `across`, `pick`, `if_any`, `if_all`, `c_across` | Required by mutate/summarise |
| 7 | slice | `slice_`, `slice_head`, `slice_tail`, `slice_sample`, `slice_min`, `slice_max` | Row subsetting |
| 8 | relocate/rename | `relocate`, `rename`, `rename_with` | Column management |
| 9 | count/tally | `count`, `tally`, `add_count`, `add_tally`, `n_distinct` | Common workflow |
| 10 | pull | `pull` | Extract column as series |
| 11 | bind | `bind_rows`, `bind_cols` | Combine data |

**Priority B (group metadata, often needed):**

| 12 | Group info | `group_indices`, `group_keys`, `group_size`, `group_rows`, `group_cols`, `n_groups`, `group_data` |
| 13 | Group ops | `group_map`, `group_modify`, `group_split`, `with_groups` |
| 14 | Order helpers | `order_by`, `with_order` |

**Priority C (utility, less urgent):**

| 15 | rows_* | `rows_insert`, `rows_update`, `rows_patch`, `rows_upsert`, `rows_delete`, `rows_append` |
| 16 | recode | `recode`, `recode_factor` |
| 17 | Misc dplyr | `between`, `cummean`, `cumall`, `cumany`, `coalesce`, `consecutive_id`, `na_if`, `near`, `nth`, `first`, `last` |
| 18 | Low-priority dplyr | `symdiff`, `glimpse`, `group_trim`, `group_walk`, `union_all`, `cur_group_rows` |

### Phase 2: base API (Essential Functions)
Estimated: ~50 critical, ~133 nice-to-have

**Priority A (needed by dplyr verbs in expressions):**

1. **Arithmetic/Stats** (10): `mean`, `sum_`, `min_`, `max_`, `abs_`, `sqrt`, `var`, `sd`, `median`, `round_`
2. **String functions** (10): `paste`, `paste0`, `toupper`, `tolower`, `nchar`, `grep`, `grepl`, `sub`, `gsub`, `trimws`
3. **Type checks** (12): `is_na`, `is_finite`, `is_infinite`, `is_numeric`, `is_character`, `is_logical`, `is_factor`, `is_null`, `is_integer`, `any_na`
4. **Type conversion** (8): `as_double`, `as_integer`, `as_character`, `as_factor`, `as_numeric`, `as_logical`

**Priority B (commonly used):**

5. **Log/Exp** (5): `log`, `log2`, `log10`, `log1p`, `exp`
6. **Cumulative** (4): `cumsum`, `cumprod`, `cummin`, `cummax`
7. **Sequence** (3): `seq`, `seq_along`, `seq_len`
8. **Set/Vector** (8): `c_`, `length`, `unique`, `intersect`, `union`, `setdiff`, `match`, `sort`
9. **Trig** (4): `cos`, `sin`, `tan` (+ maybe arctan)

**Priority C (specialized, lower volume):**

10. Remaining trig (12), special math (11), complex (5), bessel (4), random (8), factor ops (9), table (2), df inspection (12), other (7)

### Phase 3: tibble API
Estimated: ~14 verbs, 2-3 files

**Note**: `as_tibble` and `tibble` constructors already exist in `datar_polars/tibble.py`. They just need to be registered as API functions. `enframe`, `deframe`, `add_row`, `add_column` are medium priority. Row names functions are low priority (polite doesn't really use row names).

### Phase 4: tidyr API
Estimated: ~21 verbs, 10-12 files

**Priority A (highly used):**
`pivot_longer`, `pivot_wider`, `drop_na`, `fill`, `replace_na`, `separate`, `unnest`, `expand`, `complete`

**Priority B:**
`nest`, `unite`, `extract`, `separate_rows`, `uncount`, `nesting`, `crossing`

**Priority C:**
`chop`, `unchop`, `pack`, `unpack`, `full_seq`

### Phase 5: forcats API
Estimated: ~34 verbs, 5-6 files

All medium/low priority. Only needed when users work with categoricals heavily.

---

## 5. Summary Counts

| API Group | Total | Done | Remaining | Phase Estimate |
|-----------|-------|------|-----------|----------------|
| dplyr | 115 | 22 | 93 | Phase 1 |
| base | 183 | 0 | 183 | Phase 2 |
| tibble | 14 | 0 | 14 | Phase 3 |
| tidyr | 21 | 0 | 21 | Phase 4 |
| forcats | 34 | 0 | 34 | Phase 5 |
| misc | 2 | 2 | 0 | Done |
| **TOTAL** | **369** | **24** | **345** | |

---

## 6. Implementation Strategy Notes

1. **dplyr Phase 1A verbs (~30)** should be first. These are what users hit constantly: `n()`, ranking functions, `if_else`, `case_when`, `lead`/`lag`, tidyselect helpers, `across`, `slice_*`, `relocate`/`rename`.

2. **base API Phase 2A verbs (~40)** are needed because dplyr verb expressions call base functions: `mean(x)`, `sum(x)`, `paste(x, y)`, `is_na(x)`, etc. Without these, mutate/summarise/filter expressions can't use basic math/string operations.

3. Each backend module follows the same pattern: import the verb from `datar.dplyr` (or `datar.base`, etc.), register for Tibble with `@verb.register(Tibble, context=Context.PENDING, backend="polars")`, convert to polars expressions.

4. For base API functions, polars has native equivalents: `pl.col("x").mean()` for mean, `pl.col("x").str.to_uppercase()` for toupper, etc. These are straightforward.

5. The existing datar-numpy implementations can serve as algorithms for edge cases, but most functions have direct polars equivalents.

6. Reference implementation: datar-pandas at `/root/workspace/datar-pandas/datar_pandas/api/` has working implementations for ALL of these verbs that can be ported.
