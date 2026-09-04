"""End-to-end test of `merge_fusion_calls.py main()` against toy fixtures modeled on the
real 4-file input shape (see tests/fixtures/) -- fake gene names/positions, real column
structure. Exercises: two-file --fusion-annotation stacking, the reads/callers rao-score
weightings together with the pre-existing occurrence weighting, a cluster with no arriba
rows (0-default branch), and an intergenic arriba match (gene5 side dropped from the key).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from merge_fusion_calls import main

FIXTURES = Path(__file__).parent / "fixtures"


def test_main_end_to_end_on_toy_fixtures(tmp_path):
    out_path = tmp_path / "test.csv"
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

    df = pd.read_csv(out_path)
    assert len(df) == 4  # one row per cluster (c1..c4), all matched
    df = df.set_index("cluster")

    # c1: 3 final_cff rows (incl. a max_split_cnt == -1 sentinel row) + 1 arriba "high" match
    assert df.loc["c1", "n_arriba"] == 1
    assert df.loc["c1", "n_high"] == 1
    assert df.loc["c1", "arriba_conf_score"] == 3.0
    # reads-weighting is a genuinely different number from the occurrence-weighted score
    assert df.loc["c1", "BP1_rao_score_reads"] != df.loc["c1", "BP1_rao_score"]

    # c2: 1 arriba "medium" match
    assert df.loc["c2", "n_arriba"] == 1
    assert df.loc["c2", "n_med"] == 1
    assert df.loc["c2", "arriba_conf_score"] == 2.0

    # c3: no tool==arriba rows in the cluster -> the 5 arriba columns default to 0 / 0.0
    assert df.loc["c3", "n_arriba"] == 0
    assert df.loc["c3", ["n_high", "n_med", "n_low"]].tolist() == [0, 0, 0]
    assert df.loc["c3", "arriba_conf_score"] == 0.0

    # c4: intergenic arriba row (reann_gene5_symbol is NA) still resolves via the
    # gene3-only partial key, matching the "low" confidence toy arriba row
    assert df.loc["c4", "n_arriba"] == 1
    assert df.loc["c4", "n_low"] == 1
    assert df.loc["c4", "arriba_conf_score"] == 1.0

    # CallMethod flags derived correctly across both fusion_annotation files
    assert df.loc["c1", ["Arriba", "FusionCatcher", "StarFusion"]].tolist() == [1, 1, 1]  # AFS
    assert df.loc["c3", ["Arriba", "FusionCatcher", "StarFusion"]].tolist() == [0, 1, 0]  # F
