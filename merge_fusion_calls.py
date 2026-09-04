#!/usr/bin/env python3
"""Merge the fusion_annotation and filtered_fusions files into one CSV for svm_text.py.

fusion_annotation carries all of its columns through unchanged (its ``TF`` column
is renamed ``TF_f1``). ``--fusion-annotation`` accepts one or two files; two files
must share the same column names (any order) and are stacked row-wise before
anything else, with overlapping fusions left un-deduplicated. filtered_fusions is
left-joined onto fusion_annotation on a composite ``fusion_key`` and contributes
its own columns (its ``TF`` becomes ``TF_f2``).

fusion_key format
-----------------
    GENE1::GENE2|chr1:pos1|chr2:pos2

- gene pair: 5'->3', order-sensitive, taken from the fusion string
- chr: leading ``chr`` stripped ("chr17" and "17" compare equal)
- pos: bare integer ("8500970.0" tolerated)

The same key is built for both files, so a fusion_annotation row and a
filtered_fusions row match only when their gene pair and both breakpoint loci
(chr + pos, in order) agree. Strand is kept as data but not used for matching.
The key is written to the output so file 4 can later derive the same string and
join on it.

CallMethod caller flags
-----------------------
``CallMethod`` (from fusion_annotation) is a letter set naming the callers that
made each fusion -- e.g. ``AFS``, ``F``, ``FS``, ``AF``. It is expanded into
three 0/1 columns appended at the end of the output: ``Arriba`` (letter ``A``),
``FusionCatcher`` (``F``), ``StarFusion`` (``S``), plus ``n_callers`` (its
character count -- the number of callers that reported that specific fusion
call; not an aggregate over every final_cff row sharing its cluster). The
check is case-insensitive; a missing/blank ``CallMethod`` yields ``0, 0, 0, 0``.

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
    .venv/bin/python merge_fusion_calls.py \
        --fusion-annotation fusion_annotation.tsv [fusion_annotation2.tsv] \
        --filtered-fusions filtered_fusions.tsv \
        --final-cff final_cff.tsv \
        --arriba-fusions arriba_fusions.tsv \
        --output merged.csv [--sep '\\t']

Inputs are read with ``--sep`` (tab by default); the output is a
comma-separated CSV so ``svm_text.py``'s ``pd.read_csv`` reads it directly.
"""

from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

FUSION_RE = re.compile(r"([A-Za-z0-9_.\-]+::[A-Za-z0-9_.\-]+)")

# Columns each input must contain to build the key.
FILTERED_FUSIONS_KEY_COLS = ("fusion", "breakpoint", "tool", "action")
FINAL_CFF_KEY_COLS = (
    "cluster",
    "tool",
    "gene5_chr",
    "gene5_breakpoint",
    "reann_gene5_symbol",
    "gene3_chr",
    "gene3_breakpoint",
    "reann_gene3_symbol",
    "max_split_cnt",
    "max_span_cnt",
)
ARRIBA_KEY_COLS = ("#gene1", "gene2", "breakpoint1", "breakpoint2", "confidence")
CONFIDENCE_LEVELS = ("high", "medium", "low")
# Anchors _arriba_conf_score so one lone high-confidence call (raw_sum=3) scores 0.5 --
# half of the saturating ceiling of 1.0. See _arriba_conf_score for the full reasoning.
ARRIBA_CONF_SATURATION_K = 3
NA_CLUSTER_TOKENS = ("", "na", "nan", "none", "null", ".")


# --------------------------------------------------------------------------- #
# Field normalization
# --------------------------------------------------------------------------- #
def _clean(val) -> str:
    """Stringify, strip, and treat pandas NaN / empty as ``""``."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def normalize_chr(val) -> str | None:
    """'chr17' / 'Chr17' / '17' -> '17'. Empty -> None."""
    s = _clean(val)
    if not s:
        return None
    return re.sub(r"^chr", "", s, flags=re.IGNORECASE)


def normalize_pos(val) -> str | None:
    """'8500970' or '8500970.0' -> '8500970'. Non-numeric / empty -> None."""
    s = _clean(val)
    if not s:
        return None
    try:
        return str(int(float(s)))
    except ValueError:
        return None


def extract_gene_pair(fusion) -> str | None:
    """Pull 'GENE1::GENE2' out of a fusion string (handles 'Fusion {A::B}')."""
    match = FUSION_RE.search(_clean(fusion))
    return match.group(1) if match else None


def parse_breakpoint(bp) -> tuple[str, str, str, str] | None:
    """'chr17:8242985:-|chr17:8500970:-' -> ('17', '8242985', '17', '8500970').

    Returns None unless both loci parse to a chr and an integer position.
    Strand, if present as a third ``:``-field, is ignored.
    """
    s = _clean(bp)
    if "|" not in s:
        return None
    left, _, right = s.partition("|")
    loci = []
    for part in (left, right):
        fields = part.split(":")
        if len(fields) < 2:
            return None
        c, p = normalize_chr(fields[0]), normalize_pos(fields[1])
        if c is None or p is None:
            return None
        loci.append((c, p))
    (c1, p1), (c2, p2) = loci
    return c1, p1, c2, p2


def build_key(gene_pair, chr1, pos1, chr2, pos2) -> str | None:
    """Assemble the composite key, or None if any component is missing."""
    c1, p1 = normalize_chr(chr1), normalize_pos(pos1)
    c2, p2 = normalize_chr(chr2), normalize_pos(pos2)
    if not gene_pair or None in (c1, p1, c2, p2):
        return None
    return f"{gene_pair}|{c1}:{p1}|{c2}:{p2}"


def _clean_gene_symbol(val) -> str:
    """Arriba gene fields can carry ',' lists or '(...)' notes; keep the head symbol."""
    return re.split(r"[,(]", _clean(val), maxsplit=1)[0].strip()


def parse_locus(val) -> tuple[str | None, str | None]:
    """'3:50175188' or 'chr3:50175188' -> ('3', '50175188')."""
    s = _clean(val)
    if ":" not in s:
        return None, None
    c, _, p = s.partition(":")
    return normalize_chr(c), normalize_pos(p)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read(path: str, sep: str, which: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=sep, dtype=str)
    except FileNotFoundError:
        raise SystemExit(f"error: {which} file not found: {path}")
    except Exception as exc:  # noqa: BLE001 - surface parse errors cleanly
        raise SystemExit(f"error: could not read {which} {path}: {exc}")
    if df.empty:
        raise SystemExit(f"error: {which} {path} contains no rows")
    return df


def _require_cols(df: pd.DataFrame, cols, which: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"error: {which} is missing column(s): {', '.join(missing)} "
            f"(found: {', '.join(map(str, df.columns))})"
        )


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


# --------------------------------------------------------------------------- #
# final_cff cluster stats
# --------------------------------------------------------------------------- #
def rao_qe(pos, counts):
    """Rao's quadratic entropy over 1-D positions weighted by occurrence counts."""
    pos = np.asarray(pos, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if len(pos) == 1:
        return 0.0
    p = counts / counts.sum()
    d = squareform(pdist(pos.reshape(-1, 1)))
    return np.sum(np.outer(p, p) * d)


def _side_rao_score(breakpoints: pd.Series) -> float:
    """log1p(rao_qe) for one fusion side; NaN if no usable numeric breakpoints."""
    vals = pd.to_numeric(breakpoints, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    freq = vals.value_counts()
    return float(np.log1p(rao_qe(freq.index.to_numpy(), freq.to_numpy())))


def _compute_reads(split_cnt: pd.Series, span_cnt: pd.Series) -> pd.Series:
    """Per-row read support: max_split_cnt + max_span_cnt, or just max_span_cnt
    when max_split_cnt is the ``-1`` sentinel (not added in)."""
    split = pd.to_numeric(split_cnt, errors="coerce")
    span = pd.to_numeric(span_cnt, errors="coerce")
    return (split + span).where(split != -1, span)


def _side_rao_score_reads(breakpoints: pd.Series, reads: pd.Series) -> float:
    """log1p(rao_qe) for one fusion side, weighted by read support.

    breakpoints: raw breakpoint positions (may repeat across callers)
    reads: supporting read counts, same index/order as breakpoints
    """
    df = pd.DataFrame({
        "pos": pd.to_numeric(breakpoints, errors="coerce"),
        "reads": pd.to_numeric(reads, errors="coerce"),
    }).dropna()

    if df.empty:
        return float("nan")

    # collapse duplicate positions (multiple callers hitting same breakpoint)
    # by summing their read support
    agg = df.groupby("pos", as_index=False)["reads"].sum()

    return float(np.log1p(rao_qe(agg["pos"].to_numpy(), agg["reads"].to_numpy())))


def _side_rao_score_callers(breakpoints: pd.Series, caller: pd.Series) -> float:
    """log1p(rao_qe) weighted by number of distinct callers per position."""
    df = pd.DataFrame({
        "pos": pd.to_numeric(breakpoints, errors="coerce"),
        "caller": caller,
    }).dropna()

    if df.empty:
        return float("nan")

    agg = df.groupby("pos")["caller"].nunique().reset_index(name="n_callers")
    return float(np.log1p(rao_qe(agg["pos"].to_numpy(), agg["n_callers"].to_numpy())))


def _arriba_conf_score(n_high: int, n_med: int, n_low: int) -> float:
    """Saturating-ratio confidence score for a cluster's Arriba-matched rows.

    raw_sum / (raw_sum + ARRIBA_CONF_SATURATION_K), where
    raw_sum = 3*n_high + 2*n_med + n_low mirrors Arriba's own confidence
    ranking (high > medium > low).

    Why a saturating ratio and not a plain weighted average or a raw sum:
    svm_text.py trains a LinearSVC on these columns (n_high/n_med/n_low/n_arriba
    included), and a linear model already learns its own linear combination of
    them directly from data. A hand-engineered feature that is itself linear in
    those columns (e.g. the raw weighted sum, or the previous per-call average
    raw_sum / n_arriba) is redundant -- collinear with information the model
    already has. This formula is deliberately nonlinear (division by a sum that
    includes the raw_sum itself) so it encodes something a linear model cannot
    reconstruct on its own: more corroborating evidence should raise confidence
    (unlike the old per-call average, where e.g. 2 high + 2 low scored *lower*
    than 1 high alone), while the saturation keeps raw cluster size from
    dominating the score outright. ARRIBA_CONF_SATURATION_K=3 anchors one lone
    high-confidence call (raw_sum=3) to a score of 0.5 -- half of the ceiling
    the ratio approaches as evidence accumulates.
    """
    raw_sum = 3 * n_high + 2 * n_med + n_low
    return raw_sum / (raw_sum + ARRIBA_CONF_SATURATION_K)


def load_final_cff(path: str, sep: str) -> pd.DataFrame:
    df = _read(path, sep, "final_cff")
    _require_cols(df, FINAL_CFF_KEY_COLS, "final_cff")
    cl = df["cluster"].astype("string").str.strip()
    keep = cl.notna() & ~cl.str.lower().isin(NA_CLUSTER_TOKENS)
    kept = df[keep.to_numpy()].copy()
    kept.attrs["n_dropped_na_cluster"] = len(df) - len(kept)
    if kept.empty:
        raise SystemExit(f"error: final_cff {path} has no rows with a cluster")
    kept["_reads"] = _compute_reads(kept["max_split_cnt"], kept["max_span_cnt"])
    return kept


def load_arriba_fusions(path: str, sep: str) -> pd.DataFrame:
    """arriba output -> DataFrame of (arriba_key, confidence), deduped on the key.

    Alongside the full ``arriba_key`` (``G5::G3|chr:pos|chr:pos``) three partial
    keys are built with ``*`` standing in for one dropped gene, so a final_cff
    row whose ``reann_gene5/3_symbol`` is NA (intergenic breakpoint) can still be
    matched on the surviving gene plus both loci:

    - ``arriba_key_g5``      ``G5::*|chr:pos|chr:pos``  (3' gene dropped)
    - ``arriba_key_g3``      ``*::G3|chr:pos|chr:pos``  (5' gene dropped)
    - ``arriba_key_nogene``  ``*::*|chr:pos|chr:pos``   (both breakpoints only)
    """
    df = _read(path, sep, "arriba_fusions")
    _require_cols(df, ARRIBA_KEY_COLS, "arriba_fusions")

    keys, keys_g5, keys_g3, keys_nogene, confs = [], [], [], [], []
    for g1, g2, bp1, bp2, conf in zip(
        df["#gene1"], df["gene2"], df["breakpoint1"], df["breakpoint2"], df["confidence"]
    ):
        c1, p1 = parse_locus(bp1)
        c2, p2 = parse_locus(bp2)
        s1, s2 = _clean_gene_symbol(g1), _clean_gene_symbol(g2)
        pair = f"{s1}::{s2}" if s1 and s2 else None
        keys.append(build_key(pair, c1, p1, c2, p2))
        keys_g5.append(build_key(f"{s1}::*" if s1 else None, c1, p1, c2, p2))
        keys_g3.append(build_key(f"*::{s2}" if s2 else None, c1, p1, c2, p2))
        keys_nogene.append(build_key("*::*", c1, p1, c2, p2))
        c = _clean(conf).lower()
        confs.append(c if c in CONFIDENCE_LEVELS else None)

    out = pd.DataFrame(
        {
            "arriba_key": keys,
            "arriba_key_g5": keys_g5,
            "arriba_key_g3": keys_g3,
            "arriba_key_nogene": keys_nogene,
            "confidence": confs,
        }
    )
    return out.dropna(subset=["arriba_key"]).drop_duplicates("arriba_key", keep="first")


def _conf_options(df_arriba: pd.DataFrame, col: str) -> dict[str, set]:
    """Map a partial arriba key -> set of distinct confidence levels seen for it.

    A key that resolves to more than one distinct level is ambiguous and makes
    the intergenic fallback in :func:`cluster_stats` a hard error.
    """
    out: dict[str, set] = {}
    for key, conf in zip(df_arriba[col], df_arriba["confidence"]):
        if key is None:
            continue
        out.setdefault(key, set()).add(conf)
    return out


def _match_arriba_conf(
    g5, g3, c5, p5, c3, p3, conf_by_key, opts_g5, opts_g3, opts_nogene
) -> tuple[str | None, str]:
    """Resolve one final_cff ``tool==arriba`` row to an arriba confidence level.

    Returns ``(confidence, status)`` with status ``"ok"``, ``"unmatched"``, or
    ``"ambiguous"``. When ``reann_gene5_symbol`` / ``reann_gene3_symbol`` is NA
    the gene on that side is intergenic and dropped from the match: the row is
    matched on the surviving gene symbol(s) plus both ``chr:pos`` loci. An
    intergenic match that lands on several arriba rows with differing confidence
    is ``"ambiguous"``.
    """
    s5, s3 = _clean_gene_symbol(g5), _clean_gene_symbol(g3)
    if s5 and s3:
        key = build_key(f"{s5}::{s3}", c5, p5, c3, p3)
        conf = conf_by_key.get(key) if key is not None else None
        return (conf, "ok" if conf is not None else "unmatched")

    if s5:
        key = build_key(f"{s5}::*", c5, p5, c3, p3)
        cand = opts_g5.get(key) if key is not None else None
    elif s3:
        key = build_key(f"*::{s3}", c5, p5, c3, p3)
        cand = opts_g3.get(key) if key is not None else None
    else:
        key = build_key("*::*", c5, p5, c3, p3)
        cand = opts_nogene.get(key) if key is not None else None

    real = {c for c in cand if c is not None} if cand else set()
    if len(real) > 1:
        return (None, "ambiguous")
    if not real:
        return (None, "unmatched")
    return (next(iter(real)), "ok")


def cluster_stats(df_cff: pd.DataFrame, df_arriba: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster size, Rao scores, and arriba confidence stats, indexed by ``cluster``."""
    conf_by_key = df_arriba.set_index("arriba_key")["confidence"]

    opts_g5 = _conf_options(df_arriba, "arriba_key_g5")
    opts_g3 = _conf_options(df_arriba, "arriba_key_g3")
    opts_nogene = _conf_options(df_arriba, "arriba_key_nogene")

    is_arriba = df_cff["tool"].astype("string").str.strip().str.lower() == "arriba"
    cff_arr = df_cff[is_arriba.to_numpy()].copy()
    resolved = [
        _match_arriba_conf(
            g5, g3, c5, p5, c3, p3, conf_by_key, opts_g5, opts_g3, opts_nogene
        )
        for g5, g3, c5, p5, c3, p3 in zip(
            cff_arr["reann_gene5_symbol"],
            cff_arr["reann_gene3_symbol"],
            cff_arr["gene5_chr"],
            cff_arr["gene5_breakpoint"],
            cff_arr["gene3_chr"],
            cff_arr["gene3_breakpoint"],
        )
    ]
    cff_arr["_conf"] = [c for c, _ in resolved]
    cff_arr["_status"] = [s for _, s in resolved]

    _preview_cols = [
        "cluster", "reann_gene5_symbol", "gene5_chr", "gene5_breakpoint",
        "reann_gene3_symbol", "gene3_chr", "gene3_breakpoint",
    ]
    ambiguous = cff_arr[cff_arr["_status"] == "ambiguous"]
    if not ambiguous.empty:
        preview = ambiguous[_preview_cols].head(10)
        raise SystemExit(
            f"error: {len(ambiguous)} final_cff row(s) with tool==arriba and an "
            f"intergenic gene match multiple arriba_fusions rows with differing "
            f"confidence (matched on breakpoint + the non-intergenic gene). "
            f"first ambiguous:\n{preview.to_string(index=False)}"
        )

    unmatched = cff_arr[cff_arr["_status"] == "unmatched"]
    if not unmatched.empty:
        preview = unmatched[_preview_cols].head(10)
        raise SystemExit(
            f"error: {len(unmatched)} final_cff row(s) with tool==arriba have no matching "
            f"arriba_fusions row (matched on gene pair + both chr:pos; an intergenic "
            f"side is matched on breakpoint + the other gene). "
            f"first unmatched:\n{preview.to_string(index=False)}"
        )

    rows = []
    for cluster, grp in df_cff.groupby("cluster", dropna=True):
        arr = cff_arr[cff_arr["cluster"] == cluster]
        cvc = arr["_conf"].value_counts()
        n_arriba = len(arr)
        n_high, n_med, n_low = (int(cvc.get(k, 0)) for k in CONFIDENCE_LEVELS)
        rows.append(
            {
                "cluster": cluster,
                "cluster_size": len(grp),
                "BP1_rao_score": _side_rao_score(grp["gene5_breakpoint"]),
                "BP2_rao_score": _side_rao_score(grp["gene3_breakpoint"]),
                "BP1_rao_score_reads": _side_rao_score_reads(grp["gene5_breakpoint"], grp["_reads"]),
                "BP2_rao_score_reads": _side_rao_score_reads(grp["gene3_breakpoint"], grp["_reads"]),
                "BP1_rao_score_callers": _side_rao_score_callers(grp["gene5_breakpoint"], grp["tool"]),
                "BP2_rao_score_callers": _side_rao_score_callers(grp["gene3_breakpoint"], grp["tool"]),
                "min_reads": grp["_reads"].min(),
                "cv_reads": grp["_reads"].std(ddof=0) / grp["_reads"].mean(),
                "n_arriba": n_arriba,
                "n_high": n_high,
                "n_med": n_med,
                "n_low": n_low,
                "arriba_conf_score": _arriba_conf_score(n_high, n_med, n_low),
            }
        )
    cols = [
        "cluster", "cluster_size", "BP1_rao_score", "BP2_rao_score",
        "BP1_rao_score_reads", "BP2_rao_score_reads",
        "BP1_rao_score_callers", "BP2_rao_score_callers",
        "min_reads", "cv_reads",
        "n_arriba", "n_high", "n_med", "n_low", "arriba_conf_score",
    ]
    return pd.DataFrame(rows, columns=cols).set_index("cluster")


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


# Letter in CallMethod -> output column name.
CALL_METHOD_FLAGS = (("A", "Arriba"), ("F", "FusionCatcher"), ("S", "StarFusion"))


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


# Known somatic_flags database names -> their own 0/1 output column, in this order.
SOMATIC_FLAG_NAMES = (
    "Alaei-Mahabadi 18 cancers",
    "DepMap CCLE",
    "CCLE Klijn",
    "CCLE Vellichirammal",
    "Cancer Genome Project",
    "ChimerKB 4.0",
    "ChimerPub 4.0",
    "ChimerSeq 4.0",
    "COSMIC",
    "Bao gliomas",
    "Known",
    "Mitelman DB",
    "TCGA oesophageal carcinomas",
    "Bailey pancreatic cancers",
    "PCAWG",
    "Robinson prostate cancers",
    "TCGA",
    "TumorFusions tumor",
    "TCGA Gao",
    "TCGA Vellichirammal",
    "TICdb",
)


def _parse_somatic_flags(val) -> set[str]:
    """Comma-separated ``somatic_flags`` value -> the set of its (stripped) tokens."""
    s = _clean(val)
    if not s:
        return set()
    return {t.strip() for t in s.split(",") if t.strip()}


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


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Merge filtered_fusions with final_cff per-cluster stats and "
        "arriba_fusions confidence stats, writing one CSV for svm_text.py."
    )
    p.add_argument(
        "--filtered-fusions",
        required=True,
        help="path to the filtered_fusions file (base table for the output; "
        "contributes fusion_key, tool, TF, somatic_flags)",
    )
    p.add_argument(
        "--final-cff",
        required=True,
        help="path to the final_cff file (grouped by cluster; not merged)",
    )
    p.add_argument(
        "--arriba-fusions",
        required=True,
        help="path to the arriba fusions file (confidence stats per cluster)",
    )
    p.add_argument("--output", required=True, help="path to write the merged CSV")
    p.add_argument(
        "--sep",
        default="\t",
        help=r"field separator for the INPUT files (default: tab). Output is always CSV.",
    )
    return p


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


if __name__ == "__main__":
    sys.exit(main())
