# text_ml_clinical_onsite

Two standalone command-line tools for working with fusion-caller output:

| Script | Purpose |
| --- | --- |
| `merge_fusion_calls.py` | Merge fusion-caller output files into a single feature CSV. |
| `svm_text.py` | Train, evaluate, and run a linear SVM classifier over that feature CSV. |

Each script is self-contained (no package, no framework). `argparse` subcommands / flags are the
only interface.

## Requirements

- Python 3.12
- Dependencies in `requirements.txt` (scikit-learn, pandas, numpy, scipy, joblib, pytest)

### Setup

```bash
/opt/anaconda3/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run everything through the venv interpreter:

```bash
.venv/bin/python svm_text.py ...
```

## `merge_fusion_calls.py`

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

Fourteen per-cluster columns from `final_cff` + `arriba_fusions` attach to output rows by matching
`cluster`:

- `cluster_size` — row count in the `final_cff` cluster.
- `BP1_rao_score` / `BP2_rao_score` — `rao_qe` breakpoint-clustering score, weighted by occurrence count.
- `BP1_rao_score_reads` / `BP2_rao_score_reads` — same score, weighted by per-row read support instead.
- `BP1_rao_score_callers` / `BP2_rao_score_callers` — same score, weighted by distinct-caller count.
- `min_reads` / `cv_reads` — min and coefficient of variation (population stdev / mean) of per-row
  read support across the cluster. `cv_reads` is `0.0` (never `NaN`) for a single-row cluster.
- `n_arriba`, `n_high` / `n_med` / `n_low`, `arriba_conf_score` — arriba confidence counts and a
  `(3*n_high + 2*n_med + n_low) / max(n_arriba, 1)` score. Every `tool == arriba` `final_cff` row
  must match an `arriba_fusions` row, or the run hard-errors. A cluster with no arriba rows gets
  `0` / `0.0` for these five; a cluster absent from `final_cff` gets `NaN` for all fourteen.

Row-level (not cluster-aggregated) columns:

- `n_callers` — character count of `CallMethod` (e.g. `"AFS"` → `3`) for that specific row, plus
  three 0/1 flags derived from the same letters: `Arriba` (`A`), `FusionCatcher` (`F`),
  `StarFusion` (`S`).
- 21 one-hot `somatic_flags` columns (one per known database name, e.g. `COSMIC`, `Known`,
  `ChimerKB 4.0` — see `SOMATIC_FLAG_NAMES` in the script for the full list). An unrecognized
  token in the comma-separated `somatic_flags` field is silently ignored.

## `svm_text.py`

A `ColumnTransformer` → `LinearSVC(class_weight="balanced")` pipeline. Each `--text-col` column is
routed by its own pandas dtype — no column names are hardcoded — numeric columns are
median-imputed and scaled (`StandardScaler`), everything else is imputed with a `"missing"`
placeholder and one-hot encoded (`OneHotEncoder(handle_unknown="ignore")`). `LinearSVC` (not `SVC`)
is used deliberately so it scales to the resulting sparse, high-dimensional feature space; it also
can't consume raw strings or `NaN`, which is why this preprocessing step exists at all. There is no
probability output; `predict` reports `abs(decision_function)` (max over classes when multiclass)
as a `confidence` column.

### Model payload

`train` writes a dict via `joblib.dump`, not a bare pipeline:
`{"pipeline", "text_cols", "label_col", "labels"}`. `evaluate` and `predict` fall back to the stored
`text_cols` / `label_col` when the corresponding flag is omitted, so a saved model carries its own
column contract.

### Commands

```bash
# train: fit on a stratified split, print test metrics, dump model.joblib
.venv/bin/python svm_text.py train --data merged.csv \
    --text-col n_callers,cluster_size,BP1_rao_score,tool,somatic_flags --label-col label \
    --model-out model.joblib

# evaluate: score a saved model against a labelled CSV
.venv/bin/python svm_text.py evaluate --data merged.csv --model model.joblib

# predict: a CSV of rows (--output writes a CSV instead of stdout)
.venv/bin/python svm_text.py predict --model model.joblib --data rows.csv --output preds.csv
```

| `train` flag | Default | Notes |
| --- | --- | --- |
| `--data` | (required) | Path to a labelled CSV. |
| `--text-col` | `text` | Feature column, or comma-separated list of feature columns. |
| `--label-col` | `label` | Label column. |
| `--model-out` | `model.joblib` | Output path for the saved model. |
| `--test-size` | `0.2` | Held-out fraction for the stratified split. |
| `--seed` | `42` | Random seed for the split and the SVM. |

For `evaluate` / `predict`, `--text-col` and `--label-col` default to the values stored in the
model. `predict` requires `--data`.

## Conventions

- User-facing failures (missing file/column, fewer than 2 classes, bad flag combinations) exit via
  `SystemExit("error: ...")` — one clean line, no traceback.
- Syntax check: `.venv/bin/python -m py_compile svm_text.py merge_fusion_calls.py`
- Tests live in `tests/` (pytest) — run with `.venv/bin/python -m pytest`. This repo follows TDD:
  a failing test goes under `tests/` before implementation.
