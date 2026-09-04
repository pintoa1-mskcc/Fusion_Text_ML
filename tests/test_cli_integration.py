"""End-to-end test of `merge_fusion_calls.py main()` against toy fixtures modeled on the
real 3-file input shape (see tests/fixtures/) -- fake gene names/positions, real column
structure. Exercises: the action=="drop" row filter, the reads/callers rao-score
weightings together with the pre-existing occurrence weighting, a cluster with no arriba
rows (0-default branch), and an intergenic arriba match (gene5 side dropped from the key).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from merge_fusion_calls import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_main_end_to_end_on_toy_fixtures(tmp_path):
    out_path = tmp_path / "test.csv"
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

    df = pd.read_csv(out_path)
    assert len(df) == 4  # one row per surviving filtered_fusions row (c1..c4)
    df = df.set_index("cluster")
    assert "c5" not in df.index.tolist()  # action == "DROP" -> filtered before output

    # c1: 3 final_cff rows (incl. a max_split_cnt == -1 sentinel row) + 1 arriba "high" match
    assert df.loc["c1", "n_arriba"] == 1
    assert df.loc["c1", "n_high"] == 1
    # raw_sum = 3*1 = 3; K = 3 -> 3 / (3 + 3)
    assert df.loc["c1", "arriba_conf_score"] == pytest.approx(0.5)
    # reads-weighting is a genuinely different number from the occurrence-weighted score
    assert df.loc["c1", "BP1_rao_score_reads"] != df.loc["c1", "BP1_rao_score"]

    # c2: 1 arriba "medium" match
    assert df.loc["c2", "n_arriba"] == 1
    assert df.loc["c2", "n_med"] == 1
    # raw_sum = 2*1 = 2; K = 3 -> 2 / (2 + 3)
    assert df.loc["c2", "arriba_conf_score"] == pytest.approx(0.4)

    # c3: no tool==arriba rows in the cluster -> the 5 arriba columns default to 0 / 0.0
    assert df.loc["c3", "n_arriba"] == 0
    assert df.loc["c3", ["n_high", "n_med", "n_low"]].tolist() == [0, 0, 0]
    assert df.loc["c3", "arriba_conf_score"] == 0.0

    # c4: intergenic arriba row (reann_gene5_symbol is NA) still resolves via the
    # gene3-only partial key, matching the "low" confidence toy arriba row
    assert df.loc["c4", "n_arriba"] == 1
    assert df.loc["c4", "n_low"] == 1
    # raw_sum = 1*1 = 1; K = 3 -> 1 / (1 + 3)
    assert df.loc["c4", "arriba_conf_score"] == pytest.approx(0.25)

    # tool-derived caller flags
    assert df.loc["c1", ["Arriba", "FusionCatcher", "StarFusion"]].tolist() == [1, 1, 1]  # AFS
    assert df.loc["c3", ["Arriba", "FusionCatcher", "StarFusion"]].tolist() == [0, 1, 0]  # F

    # n_callers = len(tool), per output row
    assert df.loc["c1", "n_callers"] == 3  # AFS
    assert df.loc["c2", "n_callers"] == 2  # AF
    assert df.loc["c3", "n_callers"] == 1  # F
    assert df.loc["c4", "n_callers"] == 2  # AS

    # min_reads / cv_reads: c1 has 3 final_cff rows with reads [15, 8, 5] -> min 5, cv defined;
    # c4 has a single final_cff row -> cv_reads is 0.0 (population stdev of one point)
    assert df.loc["c1", "min_reads"] == 5.0
    assert df.loc["c1", "cv_reads"] > 0
    assert df.loc["c4", "min_reads"] == 2.0  # max_split_cnt(1) + max_span_cnt(1)
    assert df.loc["c4", "cv_reads"] == 0.0

    # somatic_flags one-hot encoding: c1 "Known,COSMIC"; c2 "TCGA,SomeFutureSource" (unknown
    # token silently dropped); c3 "NA" (all zero); c4 a name containing a comma-free hyphen
    assert df.loc["c1", ["Known", "COSMIC"]].tolist() == [1, 1]
    assert df.loc["c1", "TCGA"] == 0
    assert df.loc["c2", "TCGA"] == 1
    assert "SomeFutureSource" not in df.columns
    assert df.loc["c3", "Known"] == 0
    assert df.loc["c4", "Alaei-Mahabadi 18 cancers"] == 1
