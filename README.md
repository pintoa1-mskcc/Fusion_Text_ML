# text_ml_clinical_onsite

Two standalone command-line tools for working with clinical / genomic tabular data:

| Script | Purpose |
| --- | --- |
| `svm_text.py` | Train, evaluate, and run a TF-IDF + linear SVM classifier over clinical free text. |
| `merge_fusion_calls.py` | Merge fusion-caller output files into a single feature CSV suitable for `svm_text.py`. |

Each script is self-contained (no package, no framework). `argparse` subcommands / flags are the only interface.

## Requirements

- Python 3.12
- Dependencies in `requirements.txt` (scikit-learn, pandas, numpy, scipy, joblib)

### Setup

```bash
/opt/anaconda3/bin/python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run everything through the venv interpreter:

```bash
.venv/bin/python svm_text.py ...
```

## `svm_text.py`

A `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` → `LinearSVC(class_weight="balanced")`
pipeline. `LinearSVC` (not `SVC`) is used deliberately so it scales to sparse, high-dimensional text.
There is no probability output; `predict` reports `abs(decision_function)` (max over classes when
multiclass) as a `confidence` column.

### Multi-column text

`--text-col` accepts a single column name or a comma-separated list. Listed columns have their NaNs
filled with `""` and are space-joined row-wise into one string before vectorizing.

### Model payload

`train` writes a dict via `joblib.dump`, not a bare pipeline:
`{"pipeline", "text_cols", "label_col", "labels"}`. `evaluate` and `predict` fall back to the stored
`text_cols` / `label_col` when the corresponding flag is omitted, so a saved model carries its own
column contract.

### Commands

```bash
# train: fit on a stratified split, print test metrics, dump model.joblib
.venv/bin/python svm_text.py train --data data.csv \
    --text-col chief_complaint,hpi --label-col label --model-out model.joblib

# evaluate: score a saved model against a labelled CSV
.venv/bin/python svm_text.py evaluate --data data.csv --model model.joblib

# predict: a single string ...
.venv/bin/python svm_text.py predict --model model.joblib --text "crushing chest pain"

# ... or a CSV of rows (--output writes a CSV instead of stdout)
.venv/bin/python svm_text.py predict --model model.joblib --data rows.csv --output preds.csv
```

| `train` flag | Default | Notes |
| --- | --- | --- |
| `--data` | (required) | Path to a labelled CSV. |
| `--text-col` | `text` | Column, or comma-separated list of columns to concatenate. |
| `--label-col` | `label` | Label column. |
| `--model-out` | `model.joblib` | Output path for the saved model. |
| `--test-size` | `0.2` | Held-out fraction for the stratified split. |
| `--seed` | `42` | Random seed for the split and the SVM. |

For `evaluate` / `predict`, `--text-col` and `--label-col` default to the values stored in the model.
`predict` requires exactly one of `--data` or `--text`.

## `merge_fusion_calls.py`

Merges `fusion_annotation` + `filtered_fusions` on a composite `fusion_key`
(`GENE1::GENE2|chr1:pos1|chr2:pos2`), then attaches per-cluster statistics derived from `final_cff`
and `arriba_fusions`. Input files are tab-separated by default (`--sep`); output is always CSV.

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
| `--final-cff` | Grouped by `cluster` (not merged). Contributes `cluster_size`, `BP1_rao_score`, `BP2_rao_score`, `n_arriba`, `n_high` / `n_med` / `n_low`. |
| `--arriba-fusions` | Source of the per-cluster arriba confidence counts. |
| `--output` | Path to write the merged CSV. |
| `--sep` | Field separator for the **input** files (default: tab). |

The `CallMethod` letter set from `fusion_annotation` is expanded into three 0/1 columns:
`Arriba` (`A`), `FusionCatcher` (`F`), `StarFusion` (`S`).

## Conventions

- User-facing failures (missing file/column, fewer than 2 classes, bad flag combinations) exit via
  `SystemExit("error: ...")` — one clean line, no traceback.
- Syntax check: `.venv/bin/python -m py_compile svm_text.py merge_fusion_calls.py`
- There is no test suite. Running `train` → `evaluate` → `predict` end-to-end on a small labelled CSV
  is the verification loop.
