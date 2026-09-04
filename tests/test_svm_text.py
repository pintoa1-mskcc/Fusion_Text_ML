"""Tests for svm_text.py's tabular (numeric + categorical) feature pipeline.

Covers the TF-IDF -> ColumnTransformer redesign: select_feature_columns no longer
concatenates named columns into free text, build_pipeline auto-routes numeric columns
to a scaler and non-numeric columns to one-hot encoding by dtype (no column names
hardcoded), and predict's --text flag is gone (--data is required).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest

import svm_text
from merge_fusion_calls import main as merge_main

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# select_feature_columns
# --------------------------------------------------------------------------- #
def test_select_feature_columns_missing_column_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="not found in data"):
        svm_text.select_feature_columns(df, ["a", "missing"])


def test_select_feature_columns_returns_raw_dataframe_slice():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"], "c": [3, 4]})
    out = svm_text.select_feature_columns(df, ["a", "b"])
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]
    assert out["a"].tolist() == [1, 2]
    assert out["b"].tolist() == ["x", "y"]


# --------------------------------------------------------------------------- #
# build_pipeline
# --------------------------------------------------------------------------- #
def test_build_pipeline_fits_mixed_numeric_and_categorical():
    X = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0],
            "tool": ["arriba", "starfusion", "arriba", "starfusion"],
        }
    )
    y = pd.Series(["pass", "drop", "pass", "drop"])
    pipe = svm_text.build_pipeline(seed=0)
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == 4


def test_build_pipeline_imputes_missing_numeric_and_categorical():
    X = pd.DataFrame(
        {
            "score": [1.0, None, 3.0, 4.0],
            "tool": ["arriba", "starfusion", None, "starfusion"],
        }
    )
    y = pd.Series(["pass", "drop", "pass", "drop"])
    pipe = svm_text.build_pipeline(seed=0)
    pipe.fit(X, y)  # must not raise on NaN in either column
    preds = pipe.predict(X)
    assert len(preds) == 4


# --------------------------------------------------------------------------- #
# CLI: train / predict
# --------------------------------------------------------------------------- #
def _train_toy_model(tmp_path):
    data = pd.DataFrame(
        {
            "score": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "tool": ["arriba", "starfusion", "arriba", "starfusion", "arriba", "starfusion"],
            "label": ["pass", "drop", "pass", "drop", "pass", "drop"],
        }
    )
    data_path = tmp_path / "train.csv"
    data.to_csv(data_path, index=False)
    model_path = tmp_path / "model.joblib"
    svm_text.main(
        [
            "train",
            "--data", str(data_path),
            "--text-col", "score,tool",
            "--label-col", "label",
            "--model-out", str(model_path),
            "--test-size", "0.34",
        ]
    )
    return model_path


def test_predict_requires_data(tmp_path):
    model_path = _train_toy_model(tmp_path)
    with pytest.raises(SystemExit, match="error: predict needs --data"):
        svm_text.main(["predict", "--model", str(model_path)])


def test_predict_text_flag_removed():
    # --text as a distinct flag is gone; note argparse abbreviation still resolves
    # "--text" as a prefix of "--text-col", so parse_args(["--text", ...]) alone
    # wouldn't detect removal -- check the parser's own action list instead.
    predict_parser = next(
        action.choices["predict"]
        for action in svm_text.build_parser()._subparsers._group_actions
        if action.dest == "command"
    )
    assert not any(action.dest == "text" for action in predict_parser._actions)


def test_predict_outputs_prediction_and_confidence(tmp_path):
    model_path = _train_toy_model(tmp_path)
    pred_data = pd.DataFrame({"score": [1.5, 5.5], "tool": ["arriba", "starfusion"]})
    pred_path = tmp_path / "pred.csv"
    pred_data.to_csv(pred_path, index=False)
    out_path = tmp_path / "out.csv"
    svm_text.main(
        [
            "predict",
            "--model", str(model_path),
            "--data", str(pred_path),
            "--output", str(out_path),
        ]
    )
    out = pd.read_csv(out_path)
    assert "prediction" in out.columns
    assert "confidence" in out.columns
    assert len(out) == 2


# --------------------------------------------------------------------------- #
# End-to-end: merge_fusion_calls.py output -> svm_text.py train
# --------------------------------------------------------------------------- #
def test_train_on_merge_fusion_calls_output(tmp_path):
    merged_path = tmp_path / "merged.csv"
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

    df = pd.read_csv(merged_path)
    df["label"] = ["pass", "drop", "pass", "drop"][: len(df)]
    df.to_csv(merged_path, index=False)

    feature_cols = ",".join(c for c in df.columns if c not in ("label", "fusion_key"))
    model_path = tmp_path / "model.joblib"
    svm_text.main(
        [
            "train",
            "--data", str(merged_path),
            "--text-col", feature_cols,
            "--label-col", "label",
            "--model-out", str(model_path),
            "--test-size", "0.5",
        ]
    )

    assert model_path.exists()
    payload = joblib.load(model_path)
    assert payload["label_col"] == "label"
    assert payload["text_cols"] == feature_cols.split(",")
