#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import re
import sys
from collections import defaultdict, Counter

token_map = sys.argv[1]
pred_file = sys.argv[2]

BEGIN = re.compile(
    r"^#begin document\s+\((.+?)\);\s*part\s+(\d+)"
)

zeros = set()

with open(token_map, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):

        if r.get("mode", "").strip() != "with_zero":
            continue

        if r.get("split", "").strip() != "test":
            continue

        if not r.get("zero_mention_id", "").strip():
            continue

        zeros.add((
            r["doc_id"].strip(),
            int(r["sentence_id"]),
            int(r["token_id"])
        ))

mentions = []
clusters = defaultdict(list)

doc = None
sent = 1
sent_has_token = False
opened = defaultdict(list)

with open(pred_file, encoding="utf-8") as f:

    for lineno, raw in enumerate(f, 1):

        line = raw.rstrip("\n")

        if line.startswith("#begin document"):

            m = BEGIN.match(line)

            if not m:
                raise RuntimeError(
                    "Bad begin line: " + line
                )

            doc = m.group(1)
            sent = 1
            sent_has_token = False
            opened = defaultdict(list)
            continue

        if line.startswith("#end document"):
            doc = None
            continue

        if not line.strip():

            if doc is not None and sent_has_token:
                sent += 1
                sent_has_token = False

            continue

        if line.startswith("#"):
            continue

        cols = line.split()

        tok = int(cols[2])
        coref = cols[-1]

        sent_has_token = True

        if coref == "-":
            continue

        for atom in coref.split("|"):

            atom = atom.strip()

            if atom.startswith("(") and atom.endswith(")"):

                cid = atom[1:-1]

                m = (
                    doc, sent,
                    tok, tok,
                    cid
                )

                mentions.append(m)
                clusters[(doc, cid)].append(m)

            elif atom.startswith("("):

                cid = atom[1:]

                opened[cid].append(
                    (sent, tok)
                )

            elif atom.endswith(")"):

                cid = atom[:-1]

                if not opened[cid]:
                    raise RuntimeError(
                        f"Close without open: "
                        f"line={lineno} cid={cid}"
                    )

                ss, st = opened[cid].pop()

                if ss != sent:
                    raise RuntimeError(
                        "Cross-sentence span"
                    )

                m = (
                    doc, sent,
                    st, tok,
                    cid
                )

                mentions.append(m)
                clusters[(doc, cid)].append(m)

exact = set()
starts = set()
ends = set()
contains = set()
resolved_contains = set()
ambiguous = set()

lengths = Counter()

by_zero = defaultdict(list)

for z in zeros:

    zd, zs, zt = z

    hits = []

    for m in mentions:

        md, ms, start, end, cid = m

        if md != zd or ms != zs:
            continue

        if start <= zt <= end:
            hits.append(m)

    if hits:
        contains.add(z)

    if len(hits) > 1:
        ambiguous.add(z)

    for m in hits:

        md, ms, start, end, cid = m

        by_zero[z].append(m)

        lengths[end - start + 1] += 1

        if start == zt and end == zt:
            exact.add(z)

        if start == zt:
            starts.add(z)

        if end == zt:
            ends.add(z)

        if len(clusters[(md, cid)]) > 1:
            resolved_contains.add(z)

print("=" * 72)
print("STRUCTURAL_ZERO =", len(zeros))
print("EXACT_SINGLETON_ZERO =", len(exact))
print("MENTION_STARTS_AT_ZERO =", len(starts))
print("MENTION_ENDS_AT_ZERO =", len(ends))
print("MENTION_CONTAINS_ZERO =", len(contains))
print("CONTAINING_MENTION_IN_MULTI_MENTION_CLUSTER =", len(resolved_contains))
print("ZERO_WITH_MULTIPLE_CONTAINING_PRED_MENTIONS =", len(ambiguous))

if zeros:
    print(
        "ANCHOR_CONTAINMENT_RATE = {:.2f}%".format(
            100.0 * len(contains) / len(zeros)
        )
    )

print("PREDICTED_SPAN_LENGTH_DISTRIBUTION =", dict(sorted(lengths.items())))

print("=" * 72)
print("FIRST 30 ZERO-ANCHORED PREDICTED SPANS")

n = 0

for z in sorted(by_zero):

    for m in by_zero[z]:

        print(
            "ZERO =", z,
            "PRED_SPAN =", m
        )

        n += 1

        if n >= 30:
            sys.exit(0)
