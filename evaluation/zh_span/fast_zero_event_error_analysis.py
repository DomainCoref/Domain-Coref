#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


TARGET_TYPES = {
    "Zero": "零指代",
    "Event": "事件提及",
}


def canon_pair(doc, s1, s2):
    a = tuple(s1)
    b = tuple(s2)
    if a <= b:
        return (doc, a, b)
    return (doc, b, a)


def distance_bucket(d):
    if d == 0:
        return "same_sentence"
    if d == 1:
        return "1_sentence"
    if d <= 3:
        return "2-3_sentences"
    return "4+_sentences"


def write_tsv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t",
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(rows)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--adapter", required=True)
    ap.add_argument("--pair-scores", required=True)
    ap.add_argument("--type-map", required=True)
    ap.add_argument("--gold-conll", required=True)
    ap.add_argument("--diagnostic", required=True)
    ap.add_argument("--script-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threshold", type=float, default=0.50)

    a = ap.parse_args()

    sys.path.insert(
        0,
        str(Path(a.script_dir))
    )

    from score_target_mention_pair_diagnostics import (
        build_token_global_map,
        normalize_doc_key,
        span_tuple,
        as_bool,
    )

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    token_map, valid_docs = build_token_global_map(
        a.gold_conll
    )

    # --------------------------------------------------
    # 1. mention metadata
    # --------------------------------------------------

    mention_info = {}

    with open(
        a.type_map,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        for r in csv.DictReader(f):

            if (
                r["split"] != "test"
                or r["mode"] != "with_zero"
            ):
                continue

            doc = r["doc_id"]

            sid = int(r["sentence_id"])
            ts = int(r["token_start"])
            te = int(r["token_end"])

            gs = token_map[(doc, sid, ts)]
            ge = token_map[(doc, sid, te)]

            key = (doc, gs, ge)

            mention_info[key] = {
                "mention_id": r["mention_id"],
                "mention_type": r["mention_type"],
                "chain_type": r["chain_type"],
                "chain_id": r["conll_chain_id"],
                "sentence_id": sid,
                "domain": r["domain"],
                "category_primary":
                    r["category_primary"],
                "referent_semantics":
                    r["referent_semantics"],
            }

    # --------------------------------------------------
    # 2. pair scores
    # --------------------------------------------------

    score_map = {}

    with open(
        a.pair_scores,
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            doc = normalize_doc_key(
                x["doc_key"],
                valid_docs
            )

            s1 = span_tuple(
                x["global_span1"]
            )

            s2 = span_tuple(
                x["global_span2"]
            )

            key = canon_pair(
                doc,
                s1,
                s2
            )

            if key in score_map:
                raise RuntimeError(
                    f"duplicate score pair: {key}"
                )

            score_map[key] = float(
                x["p_coref"]
            )

    # --------------------------------------------------
    # 3. expected official diagnostic
    # --------------------------------------------------

    expected = {}

    with open(
        a.diagnostic,
        encoding="utf-8",
        newline=""
    ) as f:

        for r in csv.DictReader(
            f,
            delimiter="\t"
        ):

            en = r["Mention_Type_EN"]

            if en in TARGET_TYPES:

                expected[en] = {
                    "Gold_Target_Mentions":
                        int(
                            r[
                                "Gold_Target_Mentions"
                            ]
                        ),
                    "TP": int(r["TP"]),
                    "FP": int(r["FP"]),
                    "FN": int(r["FN"]),
                    "Precision":
                        float(r["Precision"]),
                    "Recall":
                        float(r["Recall"]),
                    "F1":
                        float(r["F1"]),
                }

    # --------------------------------------------------
    # 4. collect errors
    # --------------------------------------------------

    errors = {
        "Zero": [],
        "Event": [],
    }

    row_count = 0
    score_match = 0

    with open(
        a.adapter,
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row_count += 1

            x = json.loads(line)
            meta = x["_meta"]

            doc = normalize_doc_key(
                meta["doc_key"],
                valid_docs
            )

            s1 = span_tuple(
                meta["global_span1"]
            )

            s2 = span_tuple(
                meta["global_span2"]
            )

            pair_key = canon_pair(
                doc,
                s1,
                s2
            )

            if pair_key not in score_map:
                raise RuntimeError(
                    f"missing score: {pair_key}"
                )

            score_match += 1

            p = score_map[pair_key]

            gold = as_bool(x["label"])
            pred = p >= a.threshold

            if gold == pred:
                continue

            err = (
                "FP"
                if (not gold and pred)
                else "FN"
            )

            k1 = (
                doc,
                s1[0],
                s1[1]
            )

            k2 = (
                doc,
                s2[0],
                s2[1]
            )

            if (
                k1 not in mention_info
                or k2 not in mention_info
            ):
                raise RuntimeError(
                    f"mention metadata missing: "
                    f"{k1} / {k2}"
                )

            i1 = mention_info[k1]
            i2 = mention_info[k2]

            target = x.get(
                "target",
                {}
            )

            text1 = target.get(
                "span1_text",
                ""
            )

            text2 = target.get(
                "span2_text",
                ""
            )

            sent_dist = abs(
                i1["sentence_id"]
                -
                i2["sentence_id"]
            )

            extent = (
                max(s1[1], s2[1])
                -
                min(s1[0], s2[0])
                + 1
            )

            involved_types = {
                i1["mention_type"],
                i2["mention_type"],
            }

            for target_en, target_cn \
                    in TARGET_TYPES.items():

                if target_cn not in involved_types:
                    continue

                if (
                    i1["mention_type"] == target_cn
                    and
                    i2["mention_type"] == target_cn
                ):
                    partner_type = target_cn
                    target_endpoint = "both"

                elif i1["mention_type"] == target_cn:
                    partner_type = i2["mention_type"]
                    target_endpoint = "span1"

                else:
                    partner_type = i1["mention_type"]
                    target_endpoint = "span2"

                row = {
                    "Target_Type": target_en,
                    "Error": err,
                    "doc_id": doc,
                    "domain": i1["domain"],
                    "category_primary":
                        i1["category_primary"],

                    "mention1_id":
                        i1["mention_id"],
                    "mention1_type":
                        i1["mention_type"],
                    "mention1_text":
                        text1,
                    "mention1_sentence":
                        i1["sentence_id"],
                    "mention1_span":
                        f"{s1[0]}-{s1[1]}",

                    "mention2_id":
                        i2["mention_id"],
                    "mention2_type":
                        i2["mention_type"],
                    "mention2_text":
                        text2,
                    "mention2_sentence":
                        i2["sentence_id"],
                    "mention2_span":
                        f"{s2[0]}-{s2[1]}",

                    "target_endpoint":
                        target_endpoint,
                    "partner_type":
                        partner_type,

                    "gold_label":
                        int(gold),
                    "pred_label":
                        int(pred),

                    "p_coref":
                        f"{p:.8f}",

                    "score_margin":
                        f"{abs(p-a.threshold):.8f}",

                    "sentence_distance":
                        sent_dist,

                    "distance_bucket":
                        distance_bucket(
                            sent_dist
                        ),

                    "pair_extent_tokens":
                        extent,

                    "same_sentence":
                        int(sent_dist == 0),
                }

                errors[target_en].append(
                    row
                )

    if row_count != len(score_map):
        raise RuntimeError(
            f"adapter/score rows mismatch: "
            f"{row_count} vs "
            f"{len(score_map)}"
        )

    if score_match != row_count:
        raise RuntimeError(
            "not all scores matched"
        )

    # --------------------------------------------------
    # 5. output errors + top30 + summaries
    # --------------------------------------------------

    fields = [
        "Target_Type",
        "Error",
        "doc_id",
        "domain",
        "category_primary",
        "mention1_id",
        "mention1_type",
        "mention1_text",
        "mention1_sentence",
        "mention1_span",
        "mention2_id",
        "mention2_type",
        "mention2_text",
        "mention2_sentence",
        "mention2_span",
        "target_endpoint",
        "partner_type",
        "gold_label",
        "pred_label",
        "p_coref",
        "score_margin",
        "sentence_distance",
        "distance_bucket",
        "pair_extent_tokens",
        "same_sentence",
    ]

    paper_rows = []

    for target_en in [
        "Zero",
        "Event"
    ]:

        rows = errors[target_en]

        fps = [
            r for r in rows
            if r["Error"] == "FP"
        ]

        fns = [
            r for r in rows
            if r["Error"] == "FN"
        ]

        # strongest over-linking
        fps.sort(
            key=lambda r:
                -float(r["p_coref"])
        )

        # strongest missed gold links
        fns.sort(
            key=lambda r:
                float(r["p_coref"])
        )

        root = (
            out /
            target_en.lower()
        )

        write_tsv(
            root / "all_errors.tsv",
            rows,
            fields,
        )

        write_tsv(
            root / "fp_top30.tsv",
            fps[:30],
            fields,
        )

        write_tsv(
            root / "fn_top30.tsv",
            fns[:30],
            fields,
        )

        summary = []

        def add_counts(
            dimension,
            counter,
            error_type
        ):

            for value, n in \
                    counter.most_common():

                summary.append({
                    "Dimension": dimension,
                    "Value": value,
                    "Error": error_type,
                    "Count": n,
                })

        for err_name, subset in [
            ("FP", fps),
            ("FN", fns),
        ]:

            add_counts(
                "Domain",
                Counter(
                    r["domain"]
                    for r in subset
                ),
                err_name,
            )

            add_counts(
                "Partner_Type",
                Counter(
                    r["partner_type"]
                    for r in subset
                ),
                err_name,
            )

            add_counts(
                "Distance_Bucket",
                Counter(
                    r["distance_bucket"]
                    for r in subset
                ),
                err_name,
            )

        write_tsv(
            root / "summary.tsv",
            summary,
            [
                "Dimension",
                "Value",
                "Error",
                "Count",
            ],
        )

        exp = expected[target_en]

        if (
            len(fps) != exp["FP"]
            or len(fns) != exp["FN"]
        ):
            raise RuntimeError(
                f"{target_en} mismatch: "
                f"observed FP/FN="
                f"{len(fps)}/{len(fns)}, "
                f"expected="
                f"{exp['FP']}/{exp['FN']}"
            )

        fp_partner = (
            Counter(
                r["partner_type"]
                for r in fps
            ).most_common(1)
        )

        fn_partner = (
            Counter(
                r["partner_type"]
                for r in fns
            ).most_common(1)
        )

        fp_domain = (
            Counter(
                r["domain"]
                for r in fps
            ).most_common(1)
        )

        fn_domain = (
            Counter(
                r["domain"]
                for r in fns
            ).most_common(1)
        )

        paper_rows.append({
            "Type": target_en,
            "Gold_Target_Mentions":
                exp[
                    "Gold_Target_Mentions"
                ],
            "TP": exp["TP"],
            "FP": exp["FP"],
            "FN": exp["FN"],
            "Precision":
                f"{exp['Precision']:.6f}",
            "Recall":
                f"{exp['Recall']:.6f}",
            "F1":
                f"{exp['F1']:.6f}",

            "Most_Common_FP_Partner":
                (
                    fp_partner[0][0]
                    if fp_partner
                    else "NA"
                ),

            "Most_Common_FN_Partner":
                (
                    fn_partner[0][0]
                    if fn_partner
                    else "NA"
                ),

            "Most_Common_FP_Domain":
                (
                    fp_domain[0][0]
                    if fp_domain
                    else "NA"
                ),

            "Most_Common_FN_Domain":
                (
                    fn_domain[0][0]
                    if fn_domain
                    else "NA"
                ),
        })

        print(
            f"{target_en}: "
            f"FP={len(fps)} "
            f"FN={len(fns)} "
            f"TOTAL_ERRORS={len(rows)}"
        )

    paper_fields = [
        "Type",
        "Gold_Target_Mentions",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
        "Most_Common_FP_Partner",
        "Most_Common_FN_Partner",
        "Most_Common_FP_Domain",
        "Most_Common_FN_Domain",
    ]

    write_tsv(
        out /
        "error_analysis_paper_summary.tsv",
        paper_rows,
        paper_fields,
    )

    print()
    print(
        (
            out /
            "error_analysis_paper_summary.tsv"
        ).read_text(
            encoding="utf-8"
        )
    )

    print(
        "[PASS] Zero/Event fast error analysis complete"
    )


if __name__ == "__main__":
    main()
