# Remove `fusion_annotation` from `merge_fusion_calls.py`

## Context

`merge_fusion_calls.py` currently takes four inputs — `fusion_annotation` (1-2
files), `filtered_fusions`, `final_cff`, `arriba_fusions` — and left-joins
`fusion_annotation` with `filtered_fusions` on a composite `fusion_key` before
attaching `final_cff`/`arriba_fusions` cluster stats.

The user determined `fusion_annotation` is unnecessary: `filtered_fusions`
already carries every column the script's logic actually depends on, under
different names. This spec removes `fusion_annotation` entirely.

## Analysis: what the code actually needs from `fusion_annotation`

Auditing every reference to `fusion_annotation` columns in the script (not
just columns it happens to carry through as opaque output data):

1. **Key-building** (`Chr1`, `Pos1`, `Chr2`, `Pos2`, `Fusion`/`Gene1`/`Gene2`)
   — already redundant. `filtered_fusions` independently builds its own
   `fusion_key` from its own `fusion` + `breakpoint` columns
   (`load_filtered_fusions`); the two files' keys are only ever compared
   after the fact in `merge()`. No replacement column needed.
2. **`CallMethod`** — expanded into `Arriba`/`FusionCatcher`/`StarFusion`/
   `n_callers`. Confirmed with the user: `filtered_fusions`' `tool` column is
   the equivalent, and is already in the same letter-set format (e.g.
   `"AFS"`) — verified against `tests/fixtures/toy_filtered_fusions.tsv`.
3. **`TF`** — renamed to `TF_f1`, no other logic. `filtered_fusions` already
   has its own `TF` (currently renamed `TF_f2` to avoid the collision).

Everything else `fusion_annotation` contributes (`TumorId`, `Transcript1`/
`2`, `Site1`/`2Description`, `TotalReadSupport`, `FrameCallMethod`, `Note`,
`Annotation`, `Position`, `Significance`, `Kinase`) is pure passthrough — the
code never reads them, and none are currently used as `svm_text.py` training
features. Per the user: these simply disappear from the output; no
`filtered_fusions` equivalent needs to be found for them.

## Decisions

- **Scope of "used" columns**: code-logic-referenced only (key-building,
  `CallMethod`, `TF`), not passthrough-only columns. Confirmed with user.
- **`CallMethod` replacement**: `filtered_fusions.tool`. Confirmed with user.
- **`TF` naming**: `filtered_fusions`' `TF` reverts to plain `TF` in the
  output (no more `_f2` suffix, since there's no more collision with a
  `fusion_annotation`-sourced `TF_f1`). Confirmed with user.

## Design

### Removed entirely

- `--fusion-annotation` CLI flag (and its `nargs="+"` one-or-two-file
  handling / validation in `main()`)
- `load_fusion_annotation`
- `_combine_fusion_annotation`
- `FUSION_ANNOTATION_KEY_COLS`
- `merge()` — no longer needed (see below)
- Fixture files `tests/fixtures/toy_AllAnnotatedSVs.txt` and
  `tests/fixtures/toy_AllAnnotatedSVs.novel.txt`

### No more join step

`filtered_fusions` already builds its own `fusion_key` independently in
`load_filtered_fusions`. With `fusion_annotation` gone, there is nothing left
to join against — the output is simply `filtered_fusions` rows, augmented
with cluster stats, call-method flags, and somatic-flag encoding. `main()`
uses `df_filt` directly as the base passed to `attach_cluster_stats` /
`add_call_method_flags` / `add_somatic_flag_columns`, instead of a merged
frame. (The `merged` parameter name in these three functions is misleading
once nothing is merged — rename to `base`.)

### `CallMethod` → `tool`

`add_call_method_flags` reads `filtered_fusions.tool` instead of
`CallMethod`. Add `"tool"` to `FILTERED_FUSIONS_KEY_COLS` so a missing column
fails fast at load time via the existing `_require_cols` check (consistent
with how `final_cff`/`arriba_fusions` already require all their
structurally-needed columns upfront), rather than the current deferred
runtime check inside `add_call_method_flags`. Per-row blank/missing `tool`
values still degrade to `0, 0, 0, 0` (unchanged semantics, new source
column).

Add a short comment at the point of use noting that `filtered_fusions.tool`
(a letter-set naming every caller for a filtered_fusions row) is unrelated
to `final_cff.tool` (a single caller name per final_cff row) — same column
name, different file, different shape; a future reader could easily conflate
the two.

### `TF`

`load_filtered_fusions` drops its `.rename(columns={"TF": "TF_f2"})` — `TF`
stays `TF` in the output.

### Dead code this change orphans

`extract_gene_pair`'s `gene1`/`gene2` fallback parameters exist only to
support `fusion_annotation` rows that have `Gene1`/`Gene2` columns instead of
a `Fusion` string. `load_filtered_fusions` (the only remaining caller) only
ever calls `extract_gene_pair(fus)` with the fusion string. Simplify the
signature to drop the now-unreachable fallback branch and parameters.

### `summarize()` rewrite

`summarize()` currently reports `fusion_annotation`/`filtered_fusions` match
counts (`matched`, `unused`) — meaningless once there's only one base file
(no join, so every `filtered_fusions` row becomes exactly one output row).
Rewritten to report `filtered_fusions` row count directly:

```python
def summarize(df_filt, df_cff, df_arriba, stats, output) -> str:
    with_stats = output["cluster_size"].notna().sum()
    dropped = df_cff.attrs.get("n_dropped_na_cluster", 0)
    arriba_calls = int(stats["n_arriba"].sum())
    return (
        f"filtered_fusions rows: {len(df_filt)} | output rows: {len(output)}\n"
        f"final_cff: {len(stats)} clusters | rows dropped (no cluster): {dropped} | "
        f"output rows with cluster stats: {with_stats}\n"
        f"arriba_fusions rows: {len(df_arriba)} | arriba calls in clusters: {arriba_calls}"
    )
```

### Module docstring

Rewrite to drop the `fusion_annotation` description, two-file stacking
description, `TF_f1`/`TF_f2` disambiguation, and the "file 3"/"file 4"
numbering (now stale with only 3 inputs — refer to inputs by name
throughout instead of number). The `fusion_key` format section, `CallMethod`
section (renamed to describe `tool`), `somatic_flags` section, and
`final_cff` cluster-stats section stay conceptually the same, updated only
where they referenced `fusion_annotation`.

### CLI

`build_parser()` drops the `--fusion-annotation` argument. `--filtered-fusions`,
`--final-cff`, `--arriba-fusions`, `--output`, `--sep` are unchanged.

## Testing

TDD throughout (per repo convention).

- `tests/test_merge_fusion_calls.py::test_add_call_method_flags_adds_n_callers`
  — change its input `DataFrame` column from `CallMethod` to `tool`.
- `tests/test_cli_integration.py` — drop `--fusion-annotation` (and the two
  fixture paths) from the `main()` call. The toy `filtered_fusions.tool`
  values (`AFS`/`AF`/`F`/`AS`) already match what the old `CallMethod`
  fixture produced, so existing caller-flag assertions (c1: n_callers=3,
  `[Arriba,FusionCatcher,StarFusion]`=[1,1,1]; c3: `[0,1,0]`) need no new
  expected values — only the removed flag/fixtures.
- `tests/test_svm_text.py::test_train_on_merge_fusion_calls_output` — same
  `--fusion-annotation` removal from its `merge_main()` call.
- New/updated test for `load_filtered_fusions` requiring `tool` (missing
  column → `SystemExit`), mirroring the existing pattern for other required
  columns.
- Delete `tests/fixtures/toy_AllAnnotatedSVs.txt` and
  `tests/fixtures/toy_AllAnnotatedSVs.novel.txt`.

## Out of scope

- No behavior change to `final_cff`/`arriba_fusions` handling, rao-score
  weighting, `somatic_flags` encoding, or `arriba_conf_score`.
- No change to `svm_text.py`.
- No preservation path for `fusion_annotation`'s passthrough-only columns
  (`TumorId`, `Site1Description`, `Kinase`, etc.) — confirmed acceptable to
  drop per user.
