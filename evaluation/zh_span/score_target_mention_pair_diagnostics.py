#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict


TYPE_ORDER = [
    "人称代词",
    "指示代词",
    "有定描述",
    "零指代",
    "事件提及",
]

TYPE_EN = {
    "人称代词": "Pronoun",
    "指示代词": "Demonstrative",
    "有定描述": "DefiniteNP",
    "零指代": "Zero",
    "事件提及": "Event",
}


def as_bool(v):

    if isinstance(v, bool):
        return v

    s = str(v).strip().lower()

    if s in {
        "true", "1", "yes",
        "positive", "coref"
    }:
        return True

    if s in {
        "false", "0", "no",
        "negative", "noncoref"
    }:
        return False

    raise ValueError(
        f"unknown label: {v!r}"
    )


def span_tuple(x):

    return (
        int(x[0]),
        int(x[1])
    )


def normalize_doc_key(
    raw,
    valid_docs
):

    if raw in valid_docs:
        return raw

    if "_" in raw:

        candidate = raw.rsplit(
            "_",
            1
        )[0]

        if candidate in valid_docs:
            return candidate

    raise KeyError(
        f"cannot normalize doc_key={raw}"
    )


def build_token_global_map(
    conll_path
):

    begin_re = re.compile(
        r'^#begin document \((.+?)\); part (\d+)'
    )

    token_map = {}

    docs = set()

    doc = None
    sent_id = 1
    global_idx = 0
    sentence_has_tokens = False

    with open(
        conll_path,
        encoding="utf-8"
    ) as f:

        for raw in f:

            line = raw.rstrip("\n")

            m = begin_re.match(line)

            if m:

                doc = m.group(1)

                docs.add(doc)

                sent_id = 1
                global_idx = 0
                sentence_has_tokens = False

                continue

            if line.startswith(
                "#end document"
            ):

                doc = None
                continue

            if doc is None:
                continue

            if not line.strip():

                if sentence_has_tokens:

                    sent_id += 1
                    sentence_has_tokens = False

                continue

            if line.startswith("#"):
                continue

            fs = line.split()

            if len(fs) < 4:
                raise RuntimeError(
                    f"bad CoNLL line: {line}"
                )

            token_idx = int(
                fs[2]
            )

            token_map[
                (
                    doc,
                    sent_id,
                    token_idx
                )
            ] = global_idx

            global_idx += 1
            sentence_has_tokens = True

    return token_map, docs


def safe_prf(
    tp,
    fp,
    fn
):

    p = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    r = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f = (
        2 * p * r / (p + r)
        if p + r
        else 0.0
    )

    return p, r, f


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--adapter",
        required=True
    )

    ap.add_argument(
        "--pair-scores",
        required=True
    )

    ap.add_argument(
        "--type-map",
        required=True
    )

    ap.add_argument(
        "--gold-conll",
        required=True
    )

    ap.add_argument(
        "--mode",
        required=True,
        choices=[
            "with_zero",
            "surface_only"
        ]
    )

    ap.add_argument(
        "--split",
        default="test"
    )

    ap.add_argument(
        "--threshold",
        type=float,
        default=0.50
    )

    ap.add_argument(
        "--output",
        required=True
    )

    a = ap.parse_args()

    token_map, valid_docs = \
        build_token_global_map(
            a.gold_conll
        )

    type_by_span = {}

    gold_mentions_by_type = \
        defaultdict(set)

    with open(
        a.type_map,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        for r in csv.DictReader(f):

            if (
                r["split"] != a.split
                or r["mode"] != a.mode
            ):
                continue

            doc = r["doc_id"]

            sid = int(
                r["sentence_id"]
            )

            ts = int(
                r["token_start"]
            )

            te = int(
                r["token_end"]
            )

            try:

                gs = token_map[
                    (
                        doc,
                        sid,
                        ts
                    )
                ]

                ge = token_map[
                    (
                        doc,
                        sid,
                        te
                    )
                ]

            except KeyError as e:

                raise RuntimeError(
                    "type-map -> CoNLL "
                    f"alignment failed: {r}"
                ) from e

            key = (
                doc,
                gs,
                ge
            )

            mt = r["mention_type"]

            if (
                key in type_by_span
                and
                type_by_span[key]
                != mt
            ):

                raise RuntimeError(
                    "conflicting mention type: "
                    f"{key}"
                )

            type_by_span[key] = mt

            gold_mentions_by_type[
                mt
            ].add(key)

    score_by_pair = {}

    with open(
        a.pair_scores,
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            raw_doc = x["doc_key"]

            doc = normalize_doc_key(
                raw_doc,
                valid_docs
            )

            s1 = span_tuple(
                x["global_span1"]
            )

            s2 = span_tuple(
                x["global_span2"]
            )

            key = (
                doc,
                s1,
                s2
            )

            if key in score_by_pair:

                raise RuntimeError(
                    f"duplicate score pair: {key}"
                )

            score_by_pair[key] = \
                float(x["p_coref"])

    stats = {
        t: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "pairs": 0,
            "gold_pos": 0,
            "endpoint_mentions": set(),
        }
        for t in TYPE_ORDER
    }

    adapter_rows = 0
    matched_scores = 0

    with open(
        a.adapter,
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            adapter_rows += 1

            x = json.loads(line)

            meta = x["_meta"]

            raw_doc = meta["doc_key"]

            doc = normalize_doc_key(
                raw_doc,
                valid_docs
            )

            s1 = span_tuple(
                meta["global_span1"]
            )

            s2 = span_tuple(
                meta["global_span2"]
            )

            pair_key = (
                doc,
                s1,
                s2
            )

            if pair_key not in score_by_pair:

                raise RuntimeError(
                    "score missing for pair: "
                    f"{pair_key}"
                )

            matched_scores += 1

            prob = score_by_pair[
                pair_key
            ]

            gold = as_bool(
                x["label"]
            )

            pred = (
                prob >= a.threshold
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

            if k1 not in type_by_span:

                raise RuntimeError(
                    "mention type missing: "
                    f"{k1}"
                )

            if k2 not in type_by_span:

                raise RuntimeError(
                    "mention type missing: "
                    f"{k2}"
                )

            t1 = type_by_span[k1]
            t2 = type_by_span[k2]

            involved = {
                t1,
                t2
            }

            for t in involved:

                if t not in stats:
                    continue

                st = stats[t]

                st["pairs"] += 1

                if gold:
                    st["gold_pos"] += 1

                if t1 == t:
                    st[
                        "endpoint_mentions"
                    ].add(k1)

                if t2 == t:
                    st[
                        "endpoint_mentions"
                    ].add(k2)

                if gold and pred:
                    st["tp"] += 1

                elif (
                    not gold
                    and pred
                ):
                    st["fp"] += 1

                elif (
                    gold
                    and not pred
                ):
                    st["fn"] += 1

                else:
                    st["tn"] += 1

    if (
        adapter_rows
        != len(score_by_pair)
    ):

        raise RuntimeError(
            "adapter/score row mismatch: "
            f"{adapter_rows} vs "
            f"{len(score_by_pair)}"
        )

    if (
        matched_scores
        != adapter_rows
    ):

        raise RuntimeError(
            "not all scores matched"
        )

    out = Path(a.output)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fields = [
        "Mention_Type",
        "Mention_Type_EN",
        "Gold_Target_Mentions",
        "Target_Mentions_In_Pairs",
        "Endpoint_Coverage",
        "Target_Involving_Pairs",
        "Gold_Positive_Pairs",
        "TP",
        "FP",
        "FN",
        "TN",
        "Precision",
        "Recall",
        "F1",
        "Pair_Threshold",
    ]

    with out.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t"
        )

        w.writeheader()

        for t in TYPE_ORDER:

            gold_n = len(
                gold_mentions_by_type[t]
            )

            seen_n = len(
                stats[t][
                    "endpoint_mentions"
                ]
            )

            coverage = (
                seen_n / gold_n
                if gold_n
                else 0.0
            )

            tp = stats[t]["tp"]
            fp = stats[t]["fp"]
            fn = stats[t]["fn"]
            tn = stats[t]["tn"]

            p, r, f1 = safe_prf(
                tp,
                fp,
                fn
            )

            w.writerow({
                "Mention_Type": t,
                "Mention_Type_EN":
                    TYPE_EN[t],

                "Gold_Target_Mentions":
                    gold_n,

                "Target_Mentions_In_Pairs":
                    seen_n,

                "Endpoint_Coverage":
                    f"{coverage:.6f}",

                "Target_Involving_Pairs":
                    stats[t]["pairs"],

                "Gold_Positive_Pairs":
                    stats[t]["gold_pos"],

                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,

                "Precision":
                    f"{p:.6f}",

                "Recall":
                    f"{r:.6f}",

                "F1":
                    f"{f1:.6f}",

                "Pair_Threshold":
                    f"{a.threshold:.2f}",
            })

    print(
        "ADAPTER_ROWS =",
        adapter_rows
    )

    print(
        "PAIR_SCORE_ROWS =",
        len(score_by_pair)
    )

    print(
        "MATCHED_SCORE_ROWS =",
        matched_scores
    )

    print()
    print(
        out.read_text(
            encoding="utf-8"
        )
    )

    print(
        "[PASS] target mention pair diagnostic complete"
    )


if __name__ == "__main__":
    main()
