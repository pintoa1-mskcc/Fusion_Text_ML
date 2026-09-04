"""Tests for the rao-score weighting variants in merge_fusion_calls.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from merge_fusion_calls import (
    SOMATIC_FLAG_NAMES,
    _arriba_conf_score,
    _compute_reads,
    _side_rao_score,
    _side_rao_score_callers,
    _side_rao_score_reads,
    add_call_method_flags,
    add_somatic_flag_columns,
    cluster_stats,
    load_arriba_fusions,
    load_final_cff,
    load_filtered_fusions,
    rao_qe,
)

FINAL_CFF_HEADER = (
    "cluster\ttool\tgene5_chr\tgene5_breakpoint\treann_gene5_symbol\t"
    "gene3_chr\tgene3_breakpoint\treann_gene3_symbol\tmax_split_cnt\tmax_span_cnt\n"
)


def _write(path, text: str) -> str:
    path.write_text(text)
    return str(path)


# --------------------------------------------------------------------------- #
# _compute_reads
# --------------------------------------------------------------------------- #
def test_compute_reads_sums_split_and_span():
    reads = _compute_reads(pd.Series([10, 3]), pd.Series([5, 2]))
    assert reads.tolist() == [15, 5]


def test_compute_reads_sentinel_uses_span_only():
    # max_split_cnt == -1 -> reads is max_span_cnt alone, not split + span
    reads = _compute_reads(pd.Series([10, -1]), pd.Series([5, 8]))
    assert reads.tolist() == [15, 8]


def test_compute_reads_non_numeric_is_nan():
    assert _compute_reads(pd.Series(["x"]), pd.Series([5])).isna().all()


# --------------------------------------------------------------------------- #
# _arriba_conf_score
# --------------------------------------------------------------------------- #
def test_arriba_conf_score_more_corroboration_beats_lone_high_confidence_call():
    # 2 high + 2 low (more corroborating evidence, mixed confidence) should score
    # higher than 1 high alone -- the invariant the saturating-ratio redesign fixes.
    lone_high = _arriba_conf_score(n_high=1, n_med=0, n_low=0)
    two_high_two_low = _arriba_conf_score(n_high=2, n_med=0, n_low=2)
    assert two_high_two_low > lone_high


def test_arriba_conf_score_zero_calls_is_zero():
    assert _arriba_conf_score(n_high=0, n_med=0, n_low=0) == 0.0


def test_arriba_conf_score_hand_derived_value():
    # raw_sum = 3*2 + 2*0 + 1*2 = 8; K = 3 -> 8 / (8 + 3)
    assert _arriba_conf_score(n_high=2, n_med=0, n_low=2) == pytest.approx(8 / 11)


# --------------------------------------------------------------------------- #
# _side_rao_score_reads / _side_rao_score_callers
# --------------------------------------------------------------------------- #
def test_side_rao_score_reads_collapses_duplicate_positions_by_summing_reads():
    breakpoints = pd.Series([1000, 1000, 1002])
    reads = pd.Series([15, 8, 5])
    got = _side_rao_score_reads(breakpoints, reads)
    expected = float(np.log1p(rao_qe(np.array([1000, 1002]), np.array([23, 5]))))
    assert got == pytest.approx(expected)


def test_side_rao_score_reads_empty_is_nan():
    got = _side_rao_score_reads(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert np.isnan(got)


def test_side_rao_score_callers_counts_distinct_callers_not_occurrences():
    # 3 rows at pos 1000 but only 2 distinct callers there; 1 row (1 caller) at pos 1002
    breakpoints = pd.Series([1000, 1000, 1000, 1002])
    callers = pd.Series(["starfusion", "starfusion", "fusioncatcher", "arriba"])

    got = _side_rao_score_callers(breakpoints, callers)
    expected = float(np.log1p(rao_qe(np.array([1000, 1002]), np.array([2, 1]))))
    assert got == pytest.approx(expected)

    # differs from occurrence-weighting (3 vs 1) over the same rows
    occurrence = _side_rao_score(breakpoints)
    assert got != pytest.approx(occurrence)


def test_side_rao_score_callers_empty_is_nan():
    got = _side_rao_score_callers(pd.Series([], dtype=float), pd.Series([], dtype=object))
    assert np.isnan(got)


# --------------------------------------------------------------------------- #
# cluster_stats integration (real load_final_cff / load_arriba_fusions path)
# --------------------------------------------------------------------------- #
def test_cluster_stats_reads_and_callers_columns(tmp_path):
    cff_path = _write(
        tmp_path / "final_cff.tsv",
        FINAL_CFF_HEADER
        + "1\tstarfusion\t1\t1000\tGENEA\t2\t5000\tGENEB\t10\t5\n"
        + "1\tfusioncatcher\t1\t1000\tGENEA\t2\t5000\tGENEB\t-1\t8\n"
        + "1\tarriba\t1\t1002\tGENEA\t2\t5001\tGENEB\t3\t2\n",
    )
    arriba_path = _write(
        tmp_path / "arriba_fusions.tsv",
        "#gene1\tgene2\tbreakpoint1\tbreakpoint2\tconfidence\n"
        "GENEA\tGENEB\t1:1002\t2:5001\thigh\n",
    )

    df_cff = load_final_cff(cff_path, "\t")
    df_arriba = load_arriba_fusions(arriba_path, "\t")
    stats = cluster_stats(df_cff, df_arriba)

    row = stats.loc["1"]

    # reads per row: [10+5, 8 (split=-1 sentinel -> span only), 3+2] = [15, 8, 5]
    # at gene5_breakpoint positions [1000, 1000, 1002] -> grouped {1000: 23, 1002: 5}
    expected_reads = float(np.log1p(rao_qe(np.array([1000, 1002]), np.array([23, 5]))))
    assert row["BP1_rao_score_reads"] == pytest.approx(expected_reads)

    # gene5_breakpoint 1000 has 2 distinct tools (starfusion, fusioncatcher); 1002 has 1 (arriba)
    expected_callers = float(np.log1p(rao_qe(np.array([1000, 1002]), np.array([2, 1]))))
    assert row["BP1_rao_score_callers"] == pytest.approx(expected_callers)

    # the reads-weighted score is genuinely different from the pre-existing occurrence score
    assert row["BP1_rao_score"] != pytest.approx(row["BP1_rao_score_reads"])

    # min_reads / cv_reads over all 3 rows' reads [15, 8, 5] (not grouped by position)
    # cv_reads uses population stdev (ddof=0): NaN-free even for single-row clusters, and
    # consistent with rao_qe's own n=1 convention of 0.0 (see below).
    reads_arr = np.array([15.0, 8.0, 5.0])
    assert row["min_reads"] == 5.0
    expected_cv = reads_arr.std(ddof=0) / reads_arr.mean()
    assert row["cv_reads"] == pytest.approx(expected_cv)


def test_cluster_stats_cv_reads_is_zero_for_a_single_row_cluster(tmp_path):
    # population stdev of one point is 0 (not NaN) -- avoids feeding LinearSVC a NaN feature,
    # and matches rao_qe's own single-point convention elsewhere in this file
    cff_path = _write(
        tmp_path / "final_cff.tsv",
        FINAL_CFF_HEADER + "2\tarriba\t1\t1002\tGENEA\t2\t5001\tGENEB\t3\t2\n",
    )
    arriba_path = _write(
        tmp_path / "arriba_fusions.tsv",
        "#gene1\tgene2\tbreakpoint1\tbreakpoint2\tconfidence\n"
        "GENEA\tGENEB\t1:1002\t2:5001\thigh\n",
    )
    df_cff = load_final_cff(cff_path, "\t")
    df_arriba = load_arriba_fusions(arriba_path, "\t")
    row = cluster_stats(df_cff, df_arriba).loc["2"]

    assert row["min_reads"] == 5.0  # split(3) + span(2)
    assert row["cv_reads"] == 0.0


# --------------------------------------------------------------------------- #
# n_callers (add_call_method_flags)
# --------------------------------------------------------------------------- #
def test_add_call_method_flags_adds_n_callers():
    base = pd.DataFrame({"tool": ["AFS", "AF", "F", "", None]})
    out = add_call_method_flags(base)
    assert out["n_callers"].tolist() == [3, 2, 1, 0, 0]


def test_load_filtered_fusions_requires_tool_column(tmp_path):
    path = tmp_path / "filtered_fusions.tsv"
    path.write_text("fusion\tbreakpoint\nGENEA::GENEB\tchr1:1000:+|chr2:5000:-\n")
    with pytest.raises(SystemExit, match=r"missing column\(s\): tool"):
        load_filtered_fusions(str(path), sep="\t")


def test_load_filtered_fusions_keeps_tf_column_name(tmp_path):
    path = tmp_path / "filtered_fusions.tsv"
    path.write_text(
        "fusion\tbreakpoint\ttool\tTF\n"
        "GENEA::GENEB\tchr1:1000:+|chr2:5000:-\tAFS\tFALSE\n"
    )
    df = load_filtered_fusions(str(path), sep="\t")
    assert "TF" in df.columns
    assert "TF_f2" not in df.columns


# --------------------------------------------------------------------------- #
# somatic_flags one-hot encoding
# --------------------------------------------------------------------------- #
def test_add_somatic_flag_columns_one_hot_encodes_known_names():
    merged = pd.DataFrame(
        {
            "somatic_flags": [
                "Known,COSMIC",
                "TCGA",
                None,
                "Alaei-Mahabadi 18 cancers",
            ]
        }
    )
    out = add_somatic_flag_columns(merged)

    # every known name got its own column
    for name in SOMATIC_FLAG_NAMES:
        assert name in out.columns

    assert out.loc[0, "Known"] == 1
    assert out.loc[0, "COSMIC"] == 1
    assert out.loc[0, "TCGA"] == 0  # only set on row 1, not row 0
    assert out.loc[1, "TCGA"] == 1
    assert out.loc[2, list(SOMATIC_FLAG_NAMES)].tolist() == [0] * len(SOMATIC_FLAG_NAMES)  # NaN row
    assert out.loc[3, "Alaei-Mahabadi 18 cancers"] == 1


def test_add_somatic_flag_columns_ignores_unrecognized_tokens():
    # a token not in SOMATIC_FLAG_NAMES is silently dropped, not an error
    merged = pd.DataFrame({"somatic_flags": ["TCGA,SomeFutureSource"]})
    out = add_somatic_flag_columns(merged)
    assert out.loc[0, "TCGA"] == 1
    assert "SomeFutureSource" not in out.columns


def test_load_final_cff_requires_max_split_and_span_cnt(tmp_path):
    path = _write(
        tmp_path / "final_cff_missing_cols.tsv",
        "cluster\ttool\tgene5_chr\tgene5_breakpoint\treann_gene5_symbol\t"
        "gene3_chr\tgene3_breakpoint\treann_gene3_symbol\n"
        "1\tstarfusion\t1\t1000\tGENEA\t2\t5000\tGENEB\n",
    )
    with pytest.raises(SystemExit, match="max_split_cnt"):
        load_final_cff(path, "\t")
