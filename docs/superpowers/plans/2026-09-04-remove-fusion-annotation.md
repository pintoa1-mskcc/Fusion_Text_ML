# Remove fusion_annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the `--fusion-annotation` input from `merge_fusion_calls.py` entirely; `filtered_fusions` becomes the sole base for the output, with `tool` replacing `CallMethod` as the source of caller flags. Additionally, `filtered_fusions` rows with `action == "drop"` (case-insensitive, whitespace-trimmed) are filtered out before they ever reach the output.

**Architecture:** `merge_fusion_calls.py` is a single-file CLI script (no package). This plan removes one whole input file's loading/join logic (`load_fusion_annotation`, `_combine_fusion_annotation`, `merge()`), retargets one derived-feature source column (`CallMethod` → `tool`), drops a now-unnecessary column rename (`TF` → `TF_f2`), adds a row-level filter on `filtered_fusions`' `action` column, and rewrites the docstring/summary/README to match. No change to `svm_text.py`, `final_cff`/`arriba_fusions` handling, rao-score weighting, `somatic_flags` encoding, or `arriba_conf_score`.

**Tech Stack:** Python 3.12, pandas, pytest. Run via `.venv/bin/python` (never bare `python3` — see repo `CLAUDE.md`).

**Spec:** `docs/superpowers/specs/2026-09-04-remove-fusion-annotation-design.md`

## Global Constraints

- TDD throughout: write the failing test, watch it fail for the right reason, then implement (repo `CLAUDE.md` / `superpowers:test-driven-development`).
- Run `.venv/bin/python -m pytest -q` after every task; all tests must pass before moving on.
- Syntax check with `.venv/bin/python -m py_compile merge_fusion_calls.py svm_text.py` after any production-code edit.
- `CLAUDE.md` (repo root) contains **no** references to `fusion_annotation`, `TF_f1`/`TF_f2`, `CallMethod`, or file-numbering — confirmed via grep. No `CLAUDE.md` edits are needed for this plan.
- Commit after each task, using the attribution footer already established in this repo's recent commits (`Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` + `Claude-Session:` line) if the executor is Claude Code; otherwise a plain descriptive commit message is fine.

---

### Task 1: Swap `CallMethod` → `filtered_fusions.tool` as the caller-flags source

**Files:**
- Modify: `merge_fusion_calls.py:127` (`FILTERED_FUSIONS_KEY_COLS`), `merge_fusion_calls.py:635-651` (`add_call_method_flags`)
- Test: `tests/test_merge_fusion_calls.py:175-178` (existing test), plus a new test in the same file

**Interfaces:**
- Produces: `add_call_method_flags(base: pd.DataFrame) -> pd.DataFrame` (renamed from `merged`; still returns the frame with `Arriba`/`FusionCatcher`/`StarFusion`/`n_callers` appended) — reads `base["tool"]` instead of `merged["CallMethod"]`, and no longer raises `SystemExit` on a missing column (that check moves to `load_filtered_fusions` in this task).
- Produces: `FILTERED_FUSIONS_KEY_COLS = ("fusion", "breakpoint", "tool")` — `load_filtered_fusions` now requires `tool` at load time via the existing `_require_cols` helper.

- [ ] **Step 1: Update the existing `add_call_method_flags` test to use `tool`**

Replace in `tests/test_merge_fusion_calls.py`:
```python
def test_add_call_method_flags_adds_n_callers():
    merged = pd.DataFrame({"CallMethod": ["AFS", "AF", "F", "", None]})
    out = add_call_method_flags(merged)
    assert out["n_callers"].tolist() == [3, 2, 1, 0, 0]
```
with:
```python
def test_add_call_method_flags_adds_n_callers():
    base = pd.DataFrame({"tool": ["AFS", "AF", "F", "", None]})
    out = add_call_method_flags(base)
    assert out["n_callers"].tolist() == [3, 2, 1, 0, 0]
```

- [ ] **Step 2: Add a new test for the load-time `tool`-column requirement**

Add to `tests/test_merge_fusion_calls.py`, in the `_arriba_conf_score` / general test area (add `load_filtered_fusions` to the `from merge_fusion_calls import (...)` block at the top of the file):

```python
def test_load_filtered_fusions_requires_tool_column(tmp_path):
    path = tmp_path / "filtered_fusions.tsv"
    path.write_text("fusion\tbreakpoint\nGENEA::GENEB\tchr1:1000:+|chr2:5000:-\n")
    with pytest.raises(SystemExit, match=r"missing column\(s\): tool"):
        load_filtered_fusions(str(path), sep="\t")
```

- [ ] **Step 3: Run both tests, verify they fail for the right reason**

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -k "add_call_method_flags_adds_n_callers or requires_tool_column" -v`
Expected:
- `test_add_call_method_flags_adds_n_callers` FAILS with `SystemExit: error: merged output has no 'CallMethod' column...` (uncaught, since the test doesn't wrap it in `pytest.raises`)
- `test_load_filtered_fusions_requires_tool_column` FAILS with "DID NOT RAISE" (current `FILTERED_FUSIONS_KEY_COLS` doesn't include `tool`, so no error is raised)

- [ ] **Step 4: Implement — retarget `add_call_method_flags` and require `tool` at load time**

In `merge_fusion_calls.py`, change:
```python
FILTERED_FUSIONS_KEY_COLS = ("fusion", "breakpoint")
```
to:
```python
FILTERED_FUSIONS_KEY_COLS = ("fusion", "breakpoint", "tool")
```

Replace the whole `add_call_method_flags` function body with:
```python
def add_call_method_flags(base: pd.DataFrame) -> pd.DataFrame:
    """Expand ``tool`` (filtered_fusions) into 0/1 ``Arriba`` / ``FusionCatcher`` /
    ``StarFusion``, plus ``n_callers`` (its character count -- each letter is one caller).

    Case-insensitive letter membership; a missing/blank ``tool`` value gives 0 for
    all three flags and ``n_callers`` for that row. New columns are appended at the end.

    Note: this is filtered_fusions' ``tool`` column (a letter-set naming every caller
    for this fusion call, e.g. ``"AFS"``) -- not to be confused with final_cff's ``tool``
    column, which names a single caller per final_cff row.
    """
    codes = base["tool"].map(lambda v: _clean(v).upper())
    for letter, col in CALL_METHOD_FLAGS:
        base[col] = codes.str.contains(letter, regex=False).astype(int)
    base["n_callers"] = codes.str.len()
    return base
```

(This removes the old `if "CallMethod" not in merged.columns: raise SystemExit(...)` guard — `load_filtered_fusions` now guarantees `tool` exists before this function is ever called from `main()`.)

- [ ] **Step 5: Run tests, verify they pass; run full suite**

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -k "add_call_method_flags_adds_n_callers or requires_tool_column" -v`
Expected: both PASS.

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass (the CLI integration test's `toy_filtered_fusions.tsv` already has a `tool` column in the same letter-set format `CallMethod` used, so no other test breaks).

- [ ] **Step 6: Syntax check and commit**

Run: `.venv/bin/python -m py_compile merge_fusion_calls.py svm_text.py`

```bash
git add merge_fusion_calls.py tests/test_merge_fusion_calls.py
git commit -m "Swap CallMethod for filtered_fusions.tool as the caller-flags source"
```

---

### Task 2: Stop renaming `filtered_fusions`' `TF` to `TF_f2`

**Files:**
- Modify: `merge_fusion_calls.py:320` (`load_filtered_fusions`)
- Test: `tests/test_merge_fusion_calls.py` (new test)

**Interfaces:**
- Consumes: `load_filtered_fusions(path: str, sep: str) -> pd.DataFrame` (existing signature, unchanged)
- Produces: the returned frame now keeps a column named `TF` (not `TF_f2`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_merge_fusion_calls.py`:

```python
def test_load_filtered_fusions_keeps_tf_column_name(tmp_path):
    path = tmp_path / "filtered_fusions.tsv"
    path.write_text(
        "fusion\tbreakpoint\ttool\tTF\n"
        "GENEA::GENEB\tchr1:1000:+|chr2:5000:-\tAFS\tFALSE\n"
    )
    df = load_filtered_fusions(str(path), sep="\t")
    assert "TF" in df.columns
    assert "TF_f2" not in df.columns
```

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -k test_load_filtered_fusions_keeps_tf_column_name -v`
Expected: FAILS — `assert "TF" in df.columns` is false (current code renames `TF` to `TF_f2`).

- [ ] **Step 3: Implement**

In `merge_fusion_calls.py`'s `load_filtered_fusions`, change:
```python
    df = df.rename(columns={"TF": "TF_f2"})
    df.insert(0, "fusion_key", keys)
    return df
```
to:
```python
    df.insert(0, "fusion_key", keys)
    return df
```

- [ ] **Step 4: Run it, verify it passes; run full suite**

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -k test_load_filtered_fusions_keeps_tf_column_name -v`
Expected: PASS.

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 5: Syntax check and commit**

Run: `.venv/bin/python -m py_compile merge_fusion_calls.py svm_text.py`

```bash
git add merge_fusion_calls.py tests/test_merge_fusion_calls.py
git commit -m "Stop renaming filtered_fusions' TF column to TF_f2"
```

---

### Task 3: Remove `fusion_annotation` loading, joining, and CLI flag entirely

**Files:**
- Modify: `merge_fusion_calls.py` — delete `load_fusion_annotation` (:284-304), `_combine_fusion_annotation` (:261-281), `FUSION_ANNOTATION_KEY_COLS` (:126), `merge()` (:610-615); simplify `extract_gene_pair` (:177-188); rewrite `summarize()` (:709-730); rewire `main()` (:774-793); remove `--fusion-annotation` from `build_parser()` (:741-749); rename `merged` → `base` in `attach_cluster_stats` (:618-628) and `add_somatic_flag_columns` (:688-706)
- Modify: `tests/test_cli_integration.py`, `tests/test_svm_text.py` (drop `--fusion-annotation` args from `main()`/`merge_main()` calls)
- Modify: `tests/test_merge_fusion_calls.py` (new `summarize()` test; add `summarize` to imports)
- Delete: `tests/fixtures/toy_AllAnnotatedSVs.txt`, `tests/fixtures/toy_AllAnnotatedSVs.novel.txt`

**Interfaces:**
- Consumes: `add_call_method_flags`/`FILTERED_FUSIONS_KEY_COLS` from Task 1, `load_filtered_fusions` (TF-plain) from Task 2.
- Produces: `main()` no longer accepts `--fusion-annotation`; `summarize(df_filt, df_cff, df_arriba, stats, output) -> str` (new 5-arg signature, `df_ann` and the old 6th positional `merged` param both gone — `output` replaces `merged`); `attach_cluster_stats(base, stats)`, `add_somatic_flag_columns(base)` (param renamed from `merged`, same behavior); `extract_gene_pair(fusion) -> str | None` (drops the `gene1`/`gene2` fallback parameters — `load_filtered_fusions` is now the only caller and only ever passes the fusion string).

- [ ] **Step 1: Update `tests/test_cli_integration.py` to drop `--fusion-annotation`**

Replace the `main([...])` call's argument list:
```python
    main(
        [
            "--fusion-annotation",
            str(FIXTURES / "toy_AllAnnotatedSVs.txt"),
            str(FIXTURES / "toy_AllAnnotatedSVs.novel.txt"),
            "--filtered-fusions",
            str(FIXTURES / "toy_filtered_fusions.tsv"),
            "--final-cff",
            str(FIXTURES / "toy_final.cff"),
            "--arriba-fusions",
            str(FIXTURES / "toy.fusions.tsv"),
            "--output",
            str(out_path),
        ]
    )
```
with:
```python
    main(
        [
            "--filtered-fusions",
            str(FIXTURES / "toy_filtered_fusions.tsv"),
            "--final-cff",
            str(FIXTURES / "toy_final.cff"),
            "--arriba-fusions",
            str(FIXTURES / "toy.fusions.tsv"),
            "--output",
            str(out_path),
        ]
    )
```

Also update the two stale comments in the same file (cosmetic, no assertion changes — the expected values are unchanged since `toy_filtered_fusions.tsv`'s `tool` column already carries the same letter-sets the old `CallMethod` fixture did):
- `# CallMethod flags derived correctly across both fusion_annotation files` → `# tool-derived caller flags`
- `# n_callers = len(CallMethod), per output row` → `# n_callers = len(tool), per output row`

- [ ] **Step 2: Update `tests/test_svm_text.py`'s `test_train_on_merge_fusion_calls_output` the same way**

Replace:
```python
    merge_main(
        [
            "--fusion-annotation",
            str(FIXTURES / "toy_AllAnnotatedSVs.txt"),
            str(FIXTURES / "toy_AllAnnotatedSVs.novel.txt"),
            "--filtered-fusions", str(FIXTURES / "toy_filtered_fusions.tsv"),
            "--final-cff", str(FIXTURES / "toy_final.cff"),
            "--arriba-fusions", str(FIXTURES / "toy.fusions.tsv"),
            "--output", str(merged_path),
        ]
    )
```
with:
```python
    merge_main(
        [
            "--filtered-fusions", str(FIXTURES / "toy_filtered_fusions.tsv"),
            "--final-cff", str(FIXTURES / "toy_final.cff"),
            "--arriba-fusions", str(FIXTURES / "toy.fusions.tsv"),
            "--output", str(merged_path),
        ]
    )
```

- [ ] **Step 3: Add the new `summarize()` test**

Add `summarize` to the `from merge_fusion_calls import (...)` block at the top of `tests/test_merge_fusion_calls.py`, then add:

```python
# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def test_summarize_reports_filtered_fusions_and_cluster_counts():
    df_filt = pd.DataFrame({"fusion_key": ["k1", "k2"]})
    df_cff = pd.DataFrame({"cluster": ["c1"]})
    df_cff.attrs["n_dropped_na_cluster"] = 1
    df_arriba = pd.DataFrame({"arriba_key": ["a1", "a2", "a3"]})
    stats = pd.DataFrame({"n_arriba": [2]}, index=pd.Index(["c1"], name="cluster"))
    output = pd.DataFrame({"cluster_size": [5, None]})

    result = summarize(df_filt, df_cff, df_arriba, stats, output)

    assert "filtered_fusions rows: 2" in result
    assert "output rows: 2" in result
    assert "final_cff: 1 clusters" in result
    assert "rows dropped (no cluster): 1" in result
    assert "output rows with cluster stats: 1" in result
    assert "arriba_fusions rows: 3" in result
    assert "arriba calls in clusters: 2" in result
```

- [ ] **Step 4: Run all three updated/new tests, verify they fail for the right reason**

Run: `.venv/bin/python -m pytest tests/test_cli_integration.py tests/test_svm_text.py::test_train_on_merge_fusion_calls_output tests/test_merge_fusion_calls.py::test_summarize_reports_filtered_fusions_and_cluster_counts -v`
Expected:
- `test_main_end_to_end_on_toy_fixtures` FAILS — `SystemExit: 2` / argparse `error: the following arguments are required: --fusion-annotation`
- `test_train_on_merge_fusion_calls_output` FAILS — same argparse error
- `test_summarize_reports_filtered_fusions_and_cluster_counts` FAILS — `TypeError: summarize() missing 1 required positional argument: 'merged'` (old 6-arg signature)

- [ ] **Step 5: Implement — delete `fusion_annotation` loading and the merge step**

In `merge_fusion_calls.py`:

1. Delete the `FUSION_ANNOTATION_KEY_COLS = ("Chr1", "Pos1", "Chr2", "Pos2")` line.

2. Simplify `extract_gene_pair` — replace:
```python
def extract_gene_pair(fusion, gene1=None, gene2=None) -> str | None:
    """Pull 'GENE1::GENE2' out of a fusion string (handles 'Fusion {A::B}').

    Falls back to ``gene1::gene2`` when the fusion string has no ``::`` pair.
    """
    match = FUSION_RE.search(_clean(fusion))
    if match:
        return match.group(1)
    g1, g2 = _clean(gene1), _clean(gene2)
    if g1 and g2:
        return f"{g1}::{g2}"
    return None
```
with:
```python
def extract_gene_pair(fusion) -> str | None:
    """Pull 'GENE1::GENE2' out of a fusion string (handles 'Fusion {A::B}')."""
    match = FUSION_RE.search(_clean(fusion))
    return match.group(1) if match else None
```

3. Delete `_combine_fusion_annotation` and `load_fusion_annotation` in full (the two functions between `_require_cols` and `load_filtered_fusions`).

4. Delete the `merge()` function in full:
```python
def merge(df_ann: pd.DataFrame, df_filt: pd.DataFrame) -> pd.DataFrame:
    ann_cols = [c for c in df_ann.columns if c != "fusion_key"]
    filt_cols = [c for c in df_filt.columns if c != "fusion_key"]

    merged = df_ann.merge(df_filt, on="fusion_key", how="left", suffixes=("_f1", "_f2"))
    return merged[["fusion_key"] + ann_cols + filt_cols]
```

5. Rename `merged` → `base` in `attach_cluster_stats` — replace:
```python
def attach_cluster_stats(merged: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Append the fourteen per-cluster stat columns by matching ``cluster``."""
    if "cluster" not in merged.columns:
        raise SystemExit(
            "error: merged output has no 'cluster' column to join final_cff stats on "
            "(expected from filtered_fusions)"
        )
    joined = merged.join(stats, on="cluster")
    for col in ("cluster_size", "n_arriba", "n_high", "n_med", "n_low"):
        joined[col] = joined[col].astype("Int64")
    return joined
```
with:
```python
def attach_cluster_stats(base: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Append the fourteen per-cluster stat columns by matching ``cluster``."""
    if "cluster" not in base.columns:
        raise SystemExit(
            "error: filtered_fusions has no 'cluster' column to join final_cff stats on"
        )
    joined = base.join(stats, on="cluster")
    for col in ("cluster_size", "n_arriba", "n_high", "n_med", "n_low"):
        joined[col] = joined[col].astype("Int64")
    return joined
```

6. Rename `merged` → `base` in `add_somatic_flag_columns` — replace:
```python
def add_somatic_flag_columns(merged: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ``somatic_flags`` into 21 named 0/1 columns, one per
    :data:`SOMATIC_FLAG_NAMES`.

    ``somatic_flags`` is a comma-separated list of database names (e.g.
    ``"Known,COSMIC"``); a name gets 1 in its column when present in the list,
    else 0. A token not in ``SOMATIC_FLAG_NAMES`` is silently ignored (no
    column, no error). A missing/blank ``somatic_flags`` gives 0 for all 21.
    New columns are appended at the end, in ``SOMATIC_FLAG_NAMES`` order.
    """
    if "somatic_flags" not in merged.columns:
        raise SystemExit(
            "error: merged output has no 'somatic_flags' column to encode "
            "(expected from filtered_fusions)"
        )
    tokens = merged["somatic_flags"].map(_parse_somatic_flags)
    for name in SOMATIC_FLAG_NAMES:
        merged[name] = tokens.map(lambda t, name=name: int(name in t))
    return merged
```
with:
```python
def add_somatic_flag_columns(base: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode ``somatic_flags`` into 21 named 0/1 columns, one per
    :data:`SOMATIC_FLAG_NAMES`.

    ``somatic_flags`` is a comma-separated list of database names (e.g.
    ``"Known,COSMIC"``); a name gets 1 in its column when present in the list,
    else 0. A token not in ``SOMATIC_FLAG_NAMES`` is silently ignored (no
    column, no error). A missing/blank ``somatic_flags`` gives 0 for all 21.
    New columns are appended at the end, in ``SOMATIC_FLAG_NAMES`` order.
    """
    if "somatic_flags" not in base.columns:
        raise SystemExit(
            "error: filtered_fusions has no 'somatic_flags' column to encode"
        )
    tokens = base["somatic_flags"].map(_parse_somatic_flags)
    for name in SOMATIC_FLAG_NAMES:
        base[name] = tokens.map(lambda t, name=name: int(name in t))
    return base
```

7. Rewrite `summarize()` — replace:
```python
def summarize(
    df_ann: pd.DataFrame,
    df_filt: pd.DataFrame,
    df_cff: pd.DataFrame,
    df_arriba: pd.DataFrame,
    stats: pd.DataFrame,
    merged: pd.DataFrame,
) -> str:
    filt_keys = set(df_filt["fusion_key"].dropna())
    ann_keys = set(df_ann["fusion_key"].dropna())
    matched = df_ann["fusion_key"].isin(filt_keys).sum()
    unused = (~df_filt["fusion_key"].isin(ann_keys)).sum()
    with_stats = merged["cluster_size"].notna().sum()
    dropped = df_cff.attrs.get("n_dropped_na_cluster", 0)
    arriba_calls = int(stats["n_arriba"].sum())
    return (
        f"fusion_annotation rows: {len(df_ann)} | matched: {matched} | "
        f"filtered_fusions rows unused: {unused} | output rows: {len(merged)}\n"
        f"final_cff: {len(stats)} clusters | rows dropped (no cluster): {dropped} | "
        f"output rows with cluster stats: {with_stats}\n"
        f"arriba_fusions rows: {len(df_arriba)} | arriba calls in clusters: {arriba_calls}"
    )
```
with:
```python
def summarize(
    df_filt: pd.DataFrame,
    df_cff: pd.DataFrame,
    df_arriba: pd.DataFrame,
    stats: pd.DataFrame,
    output: pd.DataFrame,
) -> str:
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

8. Remove `--fusion-annotation` from `build_parser()` — delete this block entirely:
```python
    p.add_argument(
        "--fusion-annotation",
        required=True,
        nargs="+",
        metavar="FILE",
        help="one or two fusion_annotation files (all columns kept). Two files "
        "must share the same column names (any order) and are stacked row-wise; "
        "overlapping fusions are not deduplicated.",
    )
```
(leave `--filtered-fusions`, `--final-cff`, `--arriba-fusions`, `--output`, `--sep` as-is)

9. Rewire `main()` — replace:
```python
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if not 1 <= len(args.fusion_annotation) <= 2:
        raise SystemExit("error: --fusion-annotation takes one or two files")

    df_ann = load_fusion_annotation(args.fusion_annotation, args.sep)
    df_filt = load_filtered_fusions(args.filtered_fusions, args.sep)
    df_cff = load_final_cff(args.final_cff, args.sep)
    df_arriba = load_arriba_fusions(args.arriba_fusions, args.sep)

    stats = cluster_stats(df_cff, df_arriba)
    merged = merge(df_ann, df_filt)
    merged = attach_cluster_stats(merged, stats)
    merged = add_call_method_flags(merged)
    merged = add_somatic_flag_columns(merged)

    merged.to_csv(args.output, index=False)
    print(summarize(df_ann, df_filt, df_cff, df_arriba, stats, merged))
    print(f"wrote {len(merged)} rows x {merged.shape[1]} cols -> {args.output}")
```
with:
```python
def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    df_filt = load_filtered_fusions(args.filtered_fusions, args.sep)
    df_cff = load_final_cff(args.final_cff, args.sep)
    df_arriba = load_arriba_fusions(args.arriba_fusions, args.sep)

    stats = cluster_stats(df_cff, df_arriba)
    output = attach_cluster_stats(df_filt, stats)
    output = add_call_method_flags(output)
    output = add_somatic_flag_columns(output)

    output.to_csv(args.output, index=False)
    print(summarize(df_filt, df_cff, df_arriba, stats, output))
    print(f"wrote {len(output)} rows x {output.shape[1]} cols -> {args.output}")
```

- [ ] **Step 6: Delete the orphaned fixture files**

```bash
git rm tests/fixtures/toy_AllAnnotatedSVs.txt tests/fixtures/toy_AllAnnotatedSVs.novel.txt
```

- [ ] **Step 7: Run the targeted tests, verify they pass; run full suite**

Run: `.venv/bin/python -m pytest tests/test_cli_integration.py tests/test_svm_text.py::test_train_on_merge_fusion_calls_output tests/test_merge_fusion_calls.py::test_summarize_reports_filtered_fusions_and_cluster_counts -v`
Expected: all PASS.

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, no references to the deleted fixtures remain anywhere in the suite.

- [ ] **Step 8: Syntax check and commit**

Run: `.venv/bin/python -m py_compile merge_fusion_calls.py svm_text.py`

```bash
git add merge_fusion_calls.py tests/test_cli_integration.py tests/test_svm_text.py tests/test_merge_fusion_calls.py
git commit -m "Remove fusion_annotation loading, joining, and CLI flag"
```

---

### Task 4: Filter `filtered_fusions` rows where `action == "drop"`

**Files:**
- Modify: `merge_fusion_calls.py` (`FILTERED_FUSIONS_KEY_COLS`, `load_filtered_fusions`, `summarize`)
- Modify: `tests/test_merge_fusion_calls.py` (new test; update the `summarize()` test from Task 3)
- Modify: `tests/fixtures/toy_filtered_fusions.tsv` (add a dropped row)
- Modify: `tests/test_cli_integration.py` (assert the dropped row never reaches output)

**Interfaces:**
- Produces: `FILTERED_FUSIONS_KEY_COLS = ("fusion", "breakpoint", "tool", "action")`.
- Produces: `load_filtered_fusions` now drops rows where `action` (case-insensitive, whitespace-trimmed) equals `"drop"` before key-building, and sets `df.attrs["n_dropped_action"]` on the returned frame (same pattern as `load_final_cff`'s existing `n_dropped_na_cluster`).
- Produces: `summarize()` reads `df_filt.attrs["n_dropped_action"]` and reports it — no signature change (it already receives `df_filt`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_merge_fusion_calls.py`:

```python
def test_load_filtered_fusions_drops_action_drop_rows(tmp_path):
    path = tmp_path / "filtered_fusions.tsv"
    path.write_text(
        "fusion\tbreakpoint\ttool\taction\n"
        "GENEA::GENEB\tchr1:1000:+|chr2:5000:-\tAFS\tNOVEL\n"
        "GENEC::GENED\tchr3:2000:+|chr4:6000:-\tAF\tDROP\n"
        "GENEE::GENEF\tchr5:3000:+|chr6:7000:+\tF\t drop \n"
    )
    df = load_filtered_fusions(str(path), sep="\t")
    assert len(df) == 1
    assert df.iloc[0]["fusion"] == "GENEA::GENEB"
    assert df.attrs["n_dropped_action"] == 2
```

This covers both case-insensitivity (`"DROP"`) and whitespace-trimming (`" drop "`) in one test, plus the dropped-count tracking.

- [ ] **Step 2: Run it, verify it fails**

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -k test_load_filtered_fusions_drops_action_drop_rows -v`
Expected: FAILS — either a `SystemExit` (missing `action` column, since `FILTERED_FUSIONS_KEY_COLS` doesn't require it yet) or, once that's added, `len(df) == 3` (no filtering happens yet) / `KeyError` on `df.attrs["n_dropped_action"]`.

- [ ] **Step 3: Implement**

In `merge_fusion_calls.py`, change:
```python
FILTERED_FUSIONS_KEY_COLS = ("fusion", "breakpoint", "tool")
```
to:
```python
FILTERED_FUSIONS_KEY_COLS = ("fusion", "breakpoint", "tool", "action")
```

Replace `load_filtered_fusions`'s body — current state (after Tasks 1-2; Task 3 doesn't touch this function):
```python
def load_filtered_fusions(path: str, sep: str) -> pd.DataFrame:
    df = _read(path, sep, "filtered_fusions")
    _require_cols(df, FILTERED_FUSIONS_KEY_COLS, "filtered_fusions")

    keys = []
    for fus, bp in zip(df["fusion"], df["breakpoint"]):
        parsed = parse_breakpoint(bp)
        if parsed is None:
            keys.append(None)
            continue
        c1, p1, c2, p2 = parsed
        keys.append(build_key(extract_gene_pair(fus), c1, p1, c2, p2))

    df.insert(0, "fusion_key", keys)
    return df
```
with:
```python
def load_filtered_fusions(path: str, sep: str) -> pd.DataFrame:
    df = _read(path, sep, "filtered_fusions")
    _require_cols(df, FILTERED_FUSIONS_KEY_COLS, "filtered_fusions")

    is_drop = df["action"].astype("string").str.strip().str.lower() == "drop"
    n_dropped = int(is_drop.sum())
    df = df[~is_drop.to_numpy()].copy()

    keys = []
    for fus, bp in zip(df["fusion"], df["breakpoint"]):
        parsed = parse_breakpoint(bp)
        if parsed is None:
            keys.append(None)
            continue
        c1, p1, c2, p2 = parsed
        keys.append(build_key(extract_gene_pair(fus), c1, p1, c2, p2))

    df.insert(0, "fusion_key", keys)
    df.attrs["n_dropped_action"] = n_dropped
    return df
```

- [ ] **Step 4: Run it, verify it passes**

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -k test_load_filtered_fusions_drops_action_drop_rows -v`
Expected: PASS.

- [ ] **Step 5: Update `summarize()` to report the dropped count — write the failing test first**

Update the `test_summarize_reports_filtered_fusions_and_cluster_counts` test added in Task 3:
```python
def test_summarize_reports_filtered_fusions_and_cluster_counts():
    df_filt = pd.DataFrame({"fusion_key": ["k1", "k2"]})
    df_filt.attrs["n_dropped_action"] = 3
    df_cff = pd.DataFrame({"cluster": ["c1"]})
    df_cff.attrs["n_dropped_na_cluster"] = 1
    df_arriba = pd.DataFrame({"arriba_key": ["a1", "a2", "a3"]})
    stats = pd.DataFrame({"n_arriba": [2]}, index=pd.Index(["c1"], name="cluster"))
    output = pd.DataFrame({"cluster_size": [5, None]})

    result = summarize(df_filt, df_cff, df_arriba, stats, output)

    assert "filtered_fusions rows: 2" in result
    assert "dropped action==drop: 3" in result
    assert "output rows: 2" in result
    assert "final_cff: 1 clusters" in result
    assert "rows dropped (no cluster): 1" in result
    assert "output rows with cluster stats: 1" in result
    assert "arriba_fusions rows: 3" in result
    assert "arriba calls in clusters: 2" in result
```

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -k test_summarize_reports_filtered_fusions_and_cluster_counts -v`
Expected: FAILS — `assert "dropped action==drop: 3" in result` is false (current `summarize()` doesn't report this yet).

- [ ] **Step 6: Implement the `summarize()` change**

Replace:
```python
def summarize(
    df_filt: pd.DataFrame,
    df_cff: pd.DataFrame,
    df_arriba: pd.DataFrame,
    stats: pd.DataFrame,
    output: pd.DataFrame,
) -> str:
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
with:
```python
def summarize(
    df_filt: pd.DataFrame,
    df_cff: pd.DataFrame,
    df_arriba: pd.DataFrame,
    stats: pd.DataFrame,
    output: pd.DataFrame,
) -> str:
    with_stats = output["cluster_size"].notna().sum()
    dropped = df_cff.attrs.get("n_dropped_na_cluster", 0)
    dropped_action = df_filt.attrs.get("n_dropped_action", 0)
    arriba_calls = int(stats["n_arriba"].sum())
    return (
        f"filtered_fusions rows: {len(df_filt)} (dropped action==drop: {dropped_action}) | "
        f"output rows: {len(output)}\n"
        f"final_cff: {len(stats)} clusters | rows dropped (no cluster): {dropped} | "
        f"output rows with cluster stats: {with_stats}\n"
        f"arriba_fusions rows: {len(df_arriba)} | arriba calls in clusters: {arriba_calls}"
    )
```

- [ ] **Step 7: Run it, verify it passes; run full suite**

Run: `.venv/bin/python -m pytest tests/test_merge_fusion_calls.py -v`
Expected: all PASS.

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 8: Add a dropped row to the shared fixture and assert it never reaches CLI output**

Append one row to `tests/fixtures/toy_filtered_fusions.tsv` (matching the existing header's column order — `sample, cluster, tool, fusion, total_support, breakpoint, action, reason, fp_tools, fp_flags, frame_status_br, frame_status_cl, tx5, tx3, Fusion_effect, somatic_flags, TF`):
```
toy_sample	c5	A	GENEI::GENEJ	5	chr9:9000:+|chr10:10000:-	DROP	NO_SIG_GENE	NA	NA	A	A	ENST00000000009	ENST00000000010	in-frame	NA	FALSE
```

In `tests/test_cli_integration.py`'s `test_main_end_to_end_on_toy_fixtures`, add (after `assert len(df) == 4`):
```python
    assert "c5" not in df.index.tolist()  # action == "DROP" -> filtered before output
```

- [ ] **Step 9: Run the full suite, verify everything still passes**

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass — `test_main_end_to_end_on_toy_fixtures`'s `len(df) == 4` assertion still holds (the new `c5` row is filtered out, not added to output), and the new `c5 not in df.index` assertion passes.

- [ ] **Step 10: Syntax check and commit**

Run: `.venv/bin/python -m py_compile merge_fusion_calls.py svm_text.py`

```bash
git add merge_fusion_calls.py tests/test_merge_fusion_calls.py tests/test_cli_integration.py tests/fixtures/toy_filtered_fusions.tsv
git commit -m "Filter filtered_fusions rows where action == drop"
```

---

### Task 5: Update documentation and final verification

**Files:**
- Modify: `merge_fusion_calls.py:1-111` (module docstring)
- Modify: `README.md` (lines ~31-54, ~74)

**Interfaces:** None — prose only, no behavior change.

- [ ] **Step 1: Rewrite the module docstring**

Replace the entire module docstring (from `"""Merge the fusion_annotation...` through the closing `"""` right before `from __future__ import annotations`) with:

```python
"""Merge the filtered_fusions, final_cff, and arriba_fusions files into one CSV for
svm_text.py.

filtered_fusions carries all of its columns through unchanged (after dropping
rows with ``action == "drop"``, see below) and builds the fusion_key described
below. final_cff and arriba_fusions are not merged in -- their per-cluster
statistics are computed separately (see "final_cff cluster stats" below) and
attached to filtered_fusions rows by matching cluster.

action==drop filtering
-----------------------
filtered_fusions rows whose ``action`` column is ``"drop"`` (case-insensitive,
whitespace-trimmed) are dropped before key-building and never appear in the
output.

fusion_key format
-----------------
    GENE1::GENE2|chr1:pos1|chr2:pos2

- gene pair: 5'->3', order-sensitive, taken from the fusion string
- chr: leading ``chr`` stripped ("chr17" and "17" compare equal)
- pos: bare integer ("8500970.0" tolerated)

filtered_fusions builds this key from its own ``fusion`` and ``breakpoint``
columns. The key is written to the output so downstream consumers can derive
the same string.

CallMethod caller flags
-----------------------
``tool`` (from filtered_fusions) is a letter set naming the callers that made
each fusion call -- e.g. ``AFS``, ``F``, ``FS``, ``AF``. It is expanded into
three 0/1 columns appended at the end of the output: ``Arriba`` (letter ``A``),
``FusionCatcher`` (``F``), ``StarFusion`` (``S``), plus ``n_callers`` (its
character count -- the number of callers that reported that specific fusion
call). The check is case-insensitive; a missing/blank ``tool`` value yields
``0, 0, 0, 0``. Note: this is filtered_fusions' ``tool`` column, not
final_cff's ``tool`` column (a single caller name per final_cff row) -- same
name, different file, different shape.

somatic_flags encoding
-----------------------
``somatic_flags`` (from filtered_fusions) is a comma-separated list of somatic
fusion database names (e.g. ``"Known,COSMIC"``). It is one-hot encoded into
21 named 0/1 columns, one per known database (see ``SOMATIC_FLAG_NAMES``),
appended at the end of the output. A token not in that known list is
silently ignored; a missing/blank ``somatic_flags`` gives 0 for all 21.

final_cff cluster stats
-----------------------
The final_cff file is NOT merged. Rows with a missing/blank ``cluster`` are
dropped; the rest are grouped by ``cluster`` and per-cluster values are attached
to the output by matching the output's ``cluster`` column (contributed by
filtered_fusions):

- ``cluster_size``   number of final_cff rows in the cluster
- ``BP1_rao_score``  ``log1p`` of Rao's quadratic entropy over the distinct
                     ``gene5_breakpoint`` positions in the cluster, weighted by
                     how often each position occurs
- ``BP2_rao_score``  same, over ``gene3_breakpoint``
- ``BP1_rao_score_reads`` / ``BP2_rao_score_reads``
                     same, weighted by read support instead of occurrence
                     count. Per-row "reads" = ``max_split_cnt + max_span_cnt``,
                     or just ``max_span_cnt`` when ``max_split_cnt`` is the
                     ``-1`` sentinel.
- ``BP1_rao_score_callers`` / ``BP2_rao_score_callers``
                     same, weighted by the number of distinct ``tool`` values
                     (callers) reported at each position
- ``min_reads``      minimum per-row "reads" (see above) across the cluster's
                     final_cff rows
- ``cv_reads``       coefficient of variation of "reads" across the cluster's
                     final_cff rows: population stdev (``ddof=0``) / mean.
                     ``0.0`` (not ``NaN``) for a single-row cluster -- keeps
                     the column free of NaN for LinearSVC, consistent with
                     rao_qe's own single-point convention above
- ``n_arriba``       number of final_cff rows in the cluster with ``tool`` == arriba
- ``n_high`` / ``n_med`` / ``n_low``
                     confidence counts from the arriba_fusions file, for the
                     cluster's arriba rows matched on
                     ``gene5::gene3|chr:pos|chr:pos`` (reann_gene5/3_symbol +
                     gene5/3_chr + gene5/3_breakpoint  <->  #gene1/gene2 +
                     breakpoint1/breakpoint2). When ``reann_gene5_symbol`` or
                     ``reann_gene3_symbol`` is NA the breakpoint is intergenic:
                     that gene is dropped from the key (arriba fills its side
                     with a distance list like ``SPANXB1(63183),LDOC1(44716)``,
                     which is ignored) and the row is matched on the surviving
                     gene symbol(s) plus both breakpoints. Every ``tool`` ==
                     arriba row in final_cff MUST match an arriba_fusions row --
                     an unmatched one is a hard error, and so is an intergenic
                     match that lands on arriba rows of differing confidence --
                     so n_high + n_med + n_low == n_arriba.
- ``arriba_conf_score``
                     saturating ratio, not a per-call average: ``raw_sum / (raw_sum
                     + ARRIBA_CONF_SATURATION_K)`` where ``raw_sum = 3*n_high +
                     2*n_med + n_low``. Rewards corroborating evidence -- more
                     matching calls raise the score even at mixed confidence --
                     while staying bounded. See ``_arriba_conf_score``'s docstring
                     for the full reasoning (chosen to be nonlinear in n_high/
                     n_med/n_low so it adds information svm_text.py's LinearSVC
                     can't already learn from those columns directly).

A cluster present in final_cff but with no arriba rows gets 0 for the five
arriba columns (score 0.0). Output rows whose ``cluster`` is missing, or is a
cluster not present in final_cff, get ``NaN`` for all eight.

Usage
-----
    .venv/bin/python merge_fusion_calls.py \\
        --filtered-fusions filtered_fusions.tsv \\
        --final-cff final_cff.tsv \\
        --arriba-fusions arriba_fusions.tsv \\
        --output merged.csv [--sep '\\t']

Inputs are read with ``--sep`` (tab by default); the output is a
comma-separated CSV so ``svm_text.py``'s ``pd.read_csv`` reads it directly.
"""
```

- [ ] **Step 2: Update `README.md`**

Replace the `## \`merge_fusion_calls.py\`` section's intro paragraph, command example, and flag table:
```markdown
Merges `fusion_annotation` + `filtered_fusions` on a composite `fusion_key`
(`GENE1::GENE2|chr1:pos1|chr2:pos2`), then attaches per-cluster statistics derived from `final_cff`
and `arriba_fusions`, plus row-level caller/somatic-flag features. Input files are tab-separated by
default (`--sep`); output is always CSV.

```bash
.venv/bin/python merge_fusion_calls.py \
    --fusion-annotation ann.tsv [ann2.tsv] \
    --filtered-fusions filtered.tsv \
    --final-cff final_cff.tsv \
    --arriba-fusions arriba.tsv \
    --output merged.csv
```

| Flag | Notes |
| --- | --- |
| `--fusion-annotation` | One or two files. Two files must share column names and are stacked row-wise (no dedup). Its `TF` column is renamed `TF_f1`. |
| `--filtered-fusions` | Left-joined onto `fusion_annotation` on `fusion_key`. Its `TF` becomes `TF_f2`. |
| `--final-cff` | Grouped by `cluster` (not row-merged); rows with a blank/`NA` cluster are dropped first. |
| `--arriba-fusions` | Source of the per-cluster arriba confidence counts. |
| `--output` | Path to write the merged CSV. |
| `--sep` | Field separator for the **input** files (default: tab). |
```
with:
```markdown
Builds a `fusion_key` (`GENE1::GENE2|chr1:pos1|chr2:pos2`) from `filtered_fusions` (after dropping
rows where `action == "drop"`, case-insensitive), then attaches per-cluster statistics derived from
`final_cff` and `arriba_fusions`, plus row-level caller/somatic-flag features. Input files are
tab-separated by default (`--sep`); output is always CSV.

```bash
.venv/bin/python merge_fusion_calls.py \
    --filtered-fusions filtered.tsv \
    --final-cff final_cff.tsv \
    --arriba-fusions arriba.tsv \
    --output merged.csv
```

| Flag | Notes |
| --- | --- |
| `--filtered-fusions` | Source of `fusion_key`, `tool` (caller flags), `somatic_flags`, and `TF`. Rows with `action == "drop"` (case-insensitive) are filtered out first. |
| `--final-cff` | Grouped by `cluster` (not row-merged); rows with a blank/`NA` cluster are dropped first. |
| `--arriba-fusions` | Source of the per-cluster arriba confidence counts. |
| `--output` | Path to write the merged CSV. |
| `--sep` | Field separator for the **input** files (default: tab). |
```

Then update the `n_callers` bullet in the "Row-level (not cluster-aggregated) columns" list:
```markdown
- `n_callers` — character count of `CallMethod` (e.g. `"AFS"` → `3`) for that specific row, plus
  three 0/1 flags derived from the same letters: `Arriba` (`A`), `FusionCatcher` (`F`),
  `StarFusion` (`S`).
```
to:
```markdown
- `n_callers` — character count of `filtered_fusions`' `tool` column (e.g. `"AFS"` → `3`) for that
  specific row, plus three 0/1 flags derived from the same letters: `Arriba` (`A`),
  `FusionCatcher` (`F`), `StarFusion` (`S`).
```

- [ ] **Step 3: Syntax check and final full-suite verification**

Run: `.venv/bin/python -m py_compile merge_fusion_calls.py svm_text.py`
Expected: no output (success).

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass, no warnings besides the pre-existing sklearn imputer warnings in `test_train_on_merge_fusion_calls_output` (unrelated to this change).

Run a manual smoke test to confirm the CLI works end-to-end without `--fusion-annotation`:
```bash
.venv/bin/python merge_fusion_calls.py \
    --filtered-fusions tests/fixtures/toy_filtered_fusions.tsv \
    --final-cff tests/fixtures/toy_final.cff \
    --arriba-fusions tests/fixtures/toy.fusions.tsv \
    --output /tmp/smoke_merged.csv
```
Expected: prints a summary line and `wrote 4 rows x N cols -> /tmp/smoke_merged.csv`, no traceback.

- [ ] **Step 4: Commit**

```bash
git add merge_fusion_calls.py README.md
git commit -m "Update docs for fusion_annotation removal"
```
