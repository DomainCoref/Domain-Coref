# -*- coding: utf-8 -*-

import csv
import sys
from collections import defaultdict


def parse_coref(path):

    spans = {}
    stacks = defaultdict(list)

    doc = None
    sid = 0

    with open(path, "r", encoding="utf-8") as f:

        for raw in f:

            line = raw.rstrip("\n")

            if line.startswith("#begin document"):

                doc = line.split("(", 1)[1].split(")", 1)[0]
                sid = 0
                continue

            if line.startswith("#end document"):

                doc = None
                continue

            if not line or line.startswith("#"):
                continue

            parts = line.split()

            tid = int(parts[2])

            if tid == 0:
                sid += 1

            coref = parts[-1]

            if coref == "-":
                continue

            for mark in coref.split("|"):

                if mark.startswith("(") and mark.endswith(")"):

                    spans[
                        (doc, sid, tid, tid)
                    ] = mark[1:-1]

                elif mark.startswith("("):

                    cid = mark[1:]

                    stacks[
                        (doc, cid)
                    ].append(
                        (sid, tid)
                    )

                elif mark.endswith(")"):

                    cid = mark[:-1]

                    start_sid, start_tid = (
                        stacks[(doc, cid)].pop()
                    )

                    assert start_sid == sid

                    spans[
                        (doc, sid, start_tid, tid)
                    ] = cid

    return spans


mention_map = sys.argv[1]
gold_path = sys.argv[2]
pred_path = sys.argv[3]
model = sys.argv[4]
detail_out = sys.argv[5]
summary_out = sys.argv[6]


with open(
    mention_map,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:

    rows = [
        r
        for r in csv.DictReader(f)
        if r["mode"] == "with_zero"
        and r["split"] == "test"
        and r["is_exported"] == "1"
    ]


meta = {}

for r in rows:

    span = (
        r["doc_id"],
        int(r["sentence_id"]),
        int(r["token_start"]),
        int(r["token_end"])
    )

    meta[span] = r


assert len(meta) == 4554


gold = parse_coref(gold_path)
pred = parse_coref(pred_path)

assert set(gold) == set(meta)
assert set(pred) == set(meta)


gold_groups = defaultdict(list)
pred_groups = defaultdict(list)

for span, cid in gold.items():

    gold_groups[
        (span[0], cid)
    ].append(span)


for span, cid in pred.items():

    pred_groups[
        (span[0], cid)
    ].append(span)


zeros = [
    span
    for span, r in meta.items()
    if r["is_zero"] == "1"
]


assert len(zeros) == 349


def order_key(span):

    return (
        span[1],
        span[2],
        span[3]
    )


records = []


for z in sorted(zeros):

    gold_cid = gold[z]
    pred_cid = pred[z]

    gold_members = gold_groups[
        (z[0], gold_cid)
    ]

    pred_members = pred_groups[
        (z[0], pred_cid)
    ]

    prior_gold = [
        s
        for s in gold_members
        if order_key(s) < order_key(z)
    ]

    antecedent_hit = any(
        s in pred_members
        for s in prior_gold
    )

    cocluster_hit = any(
        s != z
        and gold[s] == gold_cid
        for s in pred_members
    )

    cross_chain = any(
        gold[s] != gold_cid
        for s in pred_members
    )

    singleton = (
        len(pred_members) == 1
    )

    records.append({
        "doc_id": z[0],
        "sentence_id": z[1],
        "token_start": z[2],
        "gold_chain": gold_cid,
        "pred_chain": pred_cid,
        "gold_prior_count": len(prior_gold),
        "pred_cluster_size": len(pred_members),
        "antecedent_hit": int(antecedent_hit),
        "gold_cocluster_hit": int(cocluster_hit),
        "pred_singleton": int(singleton),
        "cross_gold_chain": int(cross_chain),
    })


fields = list(records[0].keys())


with open(
    detail_out,
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
    w.writerows(records)


with_prior = [
    r
    for r in records
    if r["gold_prior_count"] > 0
]


def rate(num, den):

    return (
        100.0 * num / den
        if den
        else 0.0
    )


summary = [
    ("model", model),
    ("literal_zero_mentions", len(records)),
    ("zeros_with_gold_prior", len(with_prior)),
    (
        "antecedent_hits",
        sum(r["antecedent_hit"] for r in with_prior)
    ),
    (
        "antecedent_hit_rate",
        "%.2f" % rate(
            sum(r["antecedent_hit"] for r in with_prior),
            len(with_prior)
        )
    ),
    (
        "gold_cocluster_hit_rate",
        "%.2f" % rate(
            sum(r["gold_cocluster_hit"] for r in records),
            len(records)
        )
    ),
    (
        "pred_singleton_rate",
        "%.2f" % rate(
            sum(r["pred_singleton"] for r in records),
            len(records)
        )
    ),
    (
        "cross_gold_chain_rate",
        "%.2f" % rate(
            sum(r["cross_gold_chain"] for r in records),
            len(records)
        )
    ),
]


with open(
    summary_out,
    "w",
    encoding="utf-8"
) as f:

    f.write("metric\tvalue\n")

    for k, v in summary:

        f.write(
            "%s\t%s\n"
            % (k, v)
        )


print("[PASS] model=%s" % model)
print("[PASS] literal_zero_mentions=%d" % len(records))
print("[PASS] detail=%s" % detail_out)
print("[PASS] summary=%s" % summary_out)
