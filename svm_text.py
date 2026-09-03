#!/usr/bin/env python3
"""Clinical text classification with a TF-IDF + linear SVM pipeline.

Subcommands
-----------
train     Fit a TfidfVectorizer + LinearSVC pipeline on a labelled CSV and save it.
evaluate  Score a saved model against a labelled CSV.
predict   Predict labels for new text (from a CSV or a single --text string).

One or more text columns may be given to --text-col as a comma-separated list;
their values are concatenated per row (space separated) before vectorizing.
"""

from __future__ import annotations

import argparse
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def parse_text_cols(text_col: str) -> list[str]:
    """Split a comma-separated --text-col value into a clean list of names."""
    cols = [c.strip() for c in text_col.split(",") if c.strip()]
    if not cols:
        raise SystemExit("error: --text-col did not name any columns")
    return cols


def combine_text_columns(df: pd.DataFrame, text_cols: list[str]) -> pd.Series:
    """Concatenate the given columns row-wise into a single text Series."""
    missing = [c for c in text_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"text column(s) not found in data: {', '.join(missing)} "
            f"(available: {', '.join(map(str, df.columns))})"
        )
    filled = df[text_cols].fillna("").astype(str)
    if len(text_cols) == 1:
        return filled[text_cols[0]]
    return filled.agg(" ".join, axis=1)


def load_dataset(
    path: str,
    text_cols: list[str],
    label_col: str | None = None,
) -> tuple[pd.Series, pd.Series | None]:
    """Load a CSV and return (combined_text, labels). labels is None if not requested."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        raise SystemExit(f"error: data file not found: {path}")
    except Exception as exc:  # noqa: BLE001 - surface pandas parse errors cleanly
        raise SystemExit(f"error: could not read {path}: {exc}")

    if df.empty:
        raise SystemExit(f"error: {path} contains no rows")

    try:
        text = combine_text_columns(df, text_cols)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")

    labels = None
    if label_col is not None:
        if label_col not in df.columns:
            raise SystemExit(
                f"error: label column {label_col!r} not found in {path} "
                f"(available: {', '.join(map(str, df.columns))})"
            )
        labels = df[label_col]

    return text, labels


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_pipeline(seed: int) -> Pipeline:
    """TF-IDF features -> linear SVM. LinearSVC scales to sparse high-dim text."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("svm", LinearSVC(C=1.0, class_weight="balanced", random_state=seed)),
        ]
    )


def print_metrics(y_true, y_pred, labels: list) -> None:
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"\naccuracy            {acc:.4f}")
    print(f"macro precision     {macro_p:.4f}")
    print(f"macro recall        {macro_r:.4f}")
    print(f"macro f1            {macro_f1:.4f}")
    print("\nper-class report")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("confusion matrix (rows = true, cols = pred)")
    print(f"labels: {labels}")
    print(confusion_matrix(y_true, y_pred, labels=labels))


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> None:
    text_cols = parse_text_cols(args.text_col)
    X, y = load_dataset(args.data, text_cols, args.label_col)

    if y.nunique() < 2:
        raise SystemExit(
            f"error: need at least 2 classes to train, found {y.nunique()} "
            f"in column {args.label_col!r}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    pipe = build_pipeline(args.seed)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    labels = sorted(y.unique().tolist(), key=str)
    print(f"trained on {len(X_train)} rows, tested on {len(X_test)} rows")
    print(f"text columns: {text_cols}")
    print_metrics(y_test, y_pred, labels)

    payload = {
        "pipeline": pipe,
        "text_cols": text_cols,
        "label_col": args.label_col,
        "labels": labels,
    }
    joblib.dump(payload, args.model_out)
    print(f"\nsaved model -> {args.model_out}")


def _load_model(path: str) -> dict:
    try:
        payload = joblib.load(path)
    except FileNotFoundError:
        raise SystemExit(f"error: model file not found: {path}")
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise SystemExit(f"error: {path} is not a model saved by this script")
    return payload


def cmd_evaluate(args: argparse.Namespace) -> None:
    payload = _load_model(args.model)
    text_cols = parse_text_cols(args.text_col) if args.text_col else payload["text_cols"]
    label_col = args.label_col or payload["label_col"]

    X, y = load_dataset(args.data, text_cols, label_col)
    y_pred = payload["pipeline"].predict(X)

    labels = payload.get("labels") or sorted(y.unique().tolist(), key=str)
    print(f"evaluated {len(X)} rows from {args.data}")
    print(f"text columns: {text_cols}")
    print_metrics(y, y_pred, labels)


def cmd_predict(args: argparse.Namespace) -> None:
    if not args.data and args.text is None:
        raise SystemExit("error: predict needs either --data or --text")
    if args.data and args.text is not None:
        raise SystemExit("error: pass only one of --data or --text")

    payload = _load_model(args.model)
    pipe = payload["pipeline"]

    if args.text is not None:
        X = pd.Series([args.text])
        source = pd.DataFrame({"text": X})
    else:
        text_cols = (
            parse_text_cols(args.text_col) if args.text_col else payload["text_cols"]
        )
        X, _ = load_dataset(args.data, text_cols, None)
        source = pd.DataFrame({"combined_text": X})

    preds = pipe.predict(X)
    margins = pipe.decision_function(X)
    # decision_function is 1-D for binary, 2-D (one column per class) for multiclass
    confidence = np.abs(margins) if margins.ndim == 1 else np.max(margins, axis=1)

    out = source.copy()
    out["prediction"] = preds
    out["confidence"] = np.round(confidence, 4)

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"wrote {len(out)} predictions -> {args.output}")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 0):
            print(out.to_string(index=False))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TF-IDF + linear SVM text classifier for clinical notes."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_col_args(p: argparse.ArgumentParser, text_col_default: str | None) -> None:
        p.add_argument(
            "--text-col",
            default=text_col_default,
            help="text column, or comma-separated list of columns to concatenate "
            "(default: %(default)s)",
        )
        p.add_argument(
            "--label-col",
            default=None,
            help="label column (train default: 'label'; "
            "evaluate default: value stored in the model)",
        )

    p_train = sub.add_parser("train", help="fit and save a model")
    p_train.add_argument("--data", required=True, help="path to a labelled CSV")
    add_col_args(p_train, text_col_default="text")
    p_train.add_argument("--model-out", default="model.joblib", help="(default: %(default)s)")
    p_train.add_argument("--test-size", type=float, default=0.2, help="(default: %(default)s)")
    p_train.add_argument("--seed", type=int, default=42, help="(default: %(default)s)")
    p_train.set_defaults(func=cmd_train, label_col_fallback="label")

    p_eval = sub.add_parser("evaluate", help="score a saved model on a labelled CSV")
    p_eval.add_argument("--data", required=True, help="path to a labelled CSV")
    p_eval.add_argument("--model", default="model.joblib", help="(default: %(default)s)")
    add_col_args(p_eval, text_col_default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="predict labels for new text")
    p_pred.add_argument("--model", default="model.joblib", help="(default: %(default)s)")
    p_pred.add_argument("--data", help="CSV of rows to classify")
    p_pred.add_argument("--text", help="a single text string to classify")
    add_col_args(p_pred, text_col_default=None)
    p_pred.add_argument("--output", help="write predictions to this CSV instead of stdout")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # train's --label-col defaults to 'label' only after parsing so evaluate can
    # fall back to the model's stored column instead.
    if args.command == "train" and args.label_col is None:
        args.label_col = "label"

    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
