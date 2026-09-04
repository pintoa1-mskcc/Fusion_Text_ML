# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file CLI (`svm_text.py`) that trains, evaluates, and runs a tabular linear SVM
classifier over fusion-call feature data produced by `merge_fusion_calls.py`. No package,
no framework.

## Environment

The machine's anaconda base env has a broken numpy 2 / scipy build — **do not use `python3` directly**.
Always use the project venv:

```bash
.venv/bin/python svm_text.py ...
```

Recreate it if missing:

```bash
/opt/anaconda3/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Commands

```bash
# train: stratified k-fold CV for stable metrics, fits final pipeline on all rows, dumps model.joblib
# --text-col names the feature columns from merge_fusion_calls.py's output (comma-separated);
# --label-col is added by hand after the merge (see merge_fusion_calls.py's own doc)
.venv/bin/python svm_text.py train --data merged.csv \
    --text-col n_callers,cluster_size,BP1_rao_score,tool,somatic_flags --label-col label \
    --model-out model.joblib

# evaluate: scores a saved model against a labelled CSV
.venv/bin/python svm_text.py evaluate --data merged.csv --model model.joblib

# predict: a CSV of rows (--output writes a CSV instead of stdout)
.venv/bin/python svm_text.py predict --model model.joblib --data rows.csv --output preds.csv
```

Unit tests live in `tests/` (pytest, added 2026-09-04) — run with `.venv/bin/python -m pytest`.
Follow TDD (`superpowers:test-driven-development`) for new work in this repo: write a failing
test under `tests/` before implementing. Syntax check:
`.venv/bin/python -m py_compile svm_text.py merge_fusion_calls.py`.

This repo follows the superpowers protocol: invoke the matching skill
(`superpowers:brainstorming` for new features, `superpowers:test-driven-development` before
implementation, `superpowers:systematic-debugging` for bugs) even when a change looks
fully-specified already.

## Architecture

`svm_text.py` is organised in three bands, top to bottom: data loading → model → subcommands → CLI.

- **Multi-column features, dtype-routed.** `--text-col` takes a comma-separated list of feature
  column names — no column names are hardcoded in the script. `parse_text_cols` splits it;
  `select_feature_columns` validates each name exists and returns that raw `df[cols]` slice
  (dtypes preserved, nothing combined into a string). `build_pipeline`'s `ColumnTransformer` then
  routes each column by its own pandas dtype: numeric → median-impute + `StandardScaler`;
  everything else → constant-impute (`"missing"`) + `OneHotEncoder(handle_unknown="ignore")`.
- **Model payload.** `joblib.dump` writes a dict, not a bare pipeline:
  `{"pipeline", "text_cols", "label_col", "labels"}`. `evaluate` and `predict` fall back to the
  stored `text_cols` / `label_col` when their flags are omitted, so a saved model carries its own
  column contract. `_load_model` rejects anything that isn't this dict shape.
- **Pipeline.** `build_pipeline` = dtype-routed `ColumnTransformer` (see above) → `LinearSVC(class_weight="balanced")`.
  LinearSVC (not `SVC`) is deliberate — it scales to the resulting sparse, high-dimensional
  one-hot feature space. No probability output; `predict` reports `abs(decision_function)` (max
  over classes when multiclass) as a `confidence` column.
- **Error convention.** User-facing failures (missing file/column, <2 classes, `predict` without
  `--data`) raise `SystemExit("error: ...")` for a clean one-line exit with no traceback.
  `select_feature_columns` raises `ValueError` internally and `load_dataset` converts it.
- **`train --label-col` default** is applied in `main()` after parsing (not via argparse `default=`)
  so that `evaluate`/`predict` can distinguish "flag omitted" from an explicit value and fall back
  to the model's stored column.

## Vault logging

This project is tracked in the Obsidian vault at `~/CCS_Projects/obsidian-vault` under
`Work/ClaudeCode/text_ml_clinical_onsite/`. Follow the session start/end protocol in
`~/.claude/CLAUDE.md`.
