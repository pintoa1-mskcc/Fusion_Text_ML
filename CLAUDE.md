# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A single-file CLI (`svm_text.py`) that trains, evaluates, and runs a TF-IDF + linear SVM
classifier over clinical free text. No package, no framework, no git repo yet.

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
# train: fits pipeline on a stratified split, prints test metrics, dumps model.joblib
.venv/bin/python svm_text.py train --data sample_data.csv \
    --text-col chief_complaint,hpi --label-col label --model-out model.joblib

# evaluate: scores a saved model against a labelled CSV
.venv/bin/python svm_text.py evaluate --data sample_data.csv --model model.joblib

# predict: one string, or a CSV of rows (--output writes a CSV instead of stdout)
.venv/bin/python svm_text.py predict --model model.joblib --text "crushing chest pain"
.venv/bin/python svm_text.py predict --model model.joblib --data rows.csv --output preds.csv
```

Unit tests live in `tests/` (pytest, added 2026-09-04) — run with `.venv/bin/python -m pytest`.
Follow TDD (`superpowers:test-driven-development`) for new work in this repo: write a failing
test under `tests/` before implementing. `sample_data.csv` (120 synthetic rows:
`chief_complaint`, `hpi`, `label`) remains the smoke-test fixture for the CLI end-to-end loop
above. Syntax check: `.venv/bin/python -m py_compile svm_text.py merge_fusion_calls.py`.

This repo follows the superpowers protocol: invoke the matching skill
(`superpowers:brainstorming` for new features, `superpowers:test-driven-development` before
implementation, `superpowers:systematic-debugging` for bugs) even when a change looks
fully-specified already.

## Architecture

`svm_text.py` is organised in three bands, top to bottom: data loading → model → subcommands → CLI.

- **Multi-column text.** `--text-col` takes a comma-separated list. `parse_text_cols` splits it;
  `combine_text_columns` validates each name exists, fills NaN with `""`, and space-joins the
  columns row-wise into one string before vectorizing. Single-column is just the len-1 case.
- **Model payload.** `joblib.dump` writes a dict, not a bare pipeline:
  `{"pipeline", "text_cols", "label_col", "labels"}`. `evaluate` and `predict` fall back to the
  stored `text_cols` / `label_col` when their flags are omitted, so a saved model carries its own
  column contract. `_load_model` rejects anything that isn't this dict shape.
- **Pipeline.** `build_pipeline` = `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)`
  → `LinearSVC(class_weight="balanced")`. LinearSVC (not `SVC`) is deliberate — it scales to sparse
  high-dimensional text. No probability output; `predict` reports `abs(decision_function)` (max over
  classes when multiclass) as a `confidence` column.
- **Error convention.** User-facing failures (missing file/column, <2 classes, `predict` with
  neither `--data` nor `--text`) raise `SystemExit("error: ...")` for a clean one-line exit with no
  traceback. `combine_text_columns` raises `ValueError` internally and `load_dataset` converts it.
- **`train --label-col` default** is applied in `main()` after parsing (not via argparse `default=`)
  so that `evaluate`/`predict` can distinguish "flag omitted" from an explicit value and fall back
  to the model's stored column.

## Vault logging

This project is tracked in the Obsidian vault at `~/CCS_Projects/obsidian-vault` under
`Work/ClaudeCode/text_ml_clinical_onsite/`. Follow the session start/end protocol in
`~/.claude/CLAUDE.md`.
