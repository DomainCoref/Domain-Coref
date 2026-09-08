#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def args_parser():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--pair-scores",
        required=True
    )

    p.add_argument(
        "--gold",
        required=True
    )

    p.add_argument(
        "--threshold",
        required=True,
        type=float
    )

    p.add_argument(
        "--output",
        required=True
    )

    return p.parse_args()


def parse_gold(path):

    lines = Path(path).read_text(
        encoding="utf-8"
    ).splitlines(True)

    docs = {}
    mentions = defaultdict(set)
    token_line = {}

    current = None
    token_idx = 0

    open_spans = defaultdict(list)

    begin_re = re.compile(
        r'^#begin document \((.+)\); part (\d+)'
    )

    for line_no, raw in enumerate(lines):

        line = raw.rstrip("\n")

        m = begin_re.match(line)

        if m:

            current = (
                m.group(1),
                m.group(2)
            )

            docs[current] = []

            token_idx = 0
            open_spans = defaultdict(list)

            continue

        if line.startswith(
            "#end document"
        ):

            if any(open_spans.values()):

                raise RuntimeError(
                    f"{current}: unclosed gold span"
                )

            current = None
            continue

        if current is None:
            continue

        if not line.strip():
            continue

        fields = line.split()

        if len(fields) != 12:

            raise RuntimeError(
                f"{current}: "
                f"expected 12 fields, "
                f"got {len(fields)}"
            )

        docs[current].append(
            fields[3]
        )

        token_line[
            (
                current,
                token_idx
            )
        ] = line_no

        coref = fields[-1]

        if coref != "-":

            for mark in coref.split("|"):

                if (
                    mark.startswith("(")
                    and mark.endswith(")")
                ):

                    mentions[
                        current
                    ].add(
                        (
                            token_idx,
                            token_idx
                        )
                    )

                elif mark.startswith("("):

                    cid = mark[1:]

                    open_spans[
                        cid
                    ].append(
                        token_idx
                    )

                elif mark.endswith(")"):

                    cid = mark[:-1]

                    if not open_spans[cid]:

                        raise RuntimeError(
                            f"{current}: "
                            f"close without open "
                            f"cid={cid}"
                        )

                    start = open_spans[
                        cid
                    ].pop()

                    mentions[
                        current
                    ].add(
                        (
                            start,
                            token_idx
                        )
                    )

                else:

                    raise RuntimeError(
                        f"{current}: "
                        f"bad mark {mark}"
                    )

        token_idx += 1

    return (
        lines,
        docs,
        mentions,
        token_line
    )


def map_doc_key(
    doc_key,
    gold_docs
):

    exact = (
        doc_key,
        "000"
    )

    if exact in gold_docs:
        return exact

    m = re.match(
        r'^(.*)_([0-9]+)$',
        doc_key
    )

    if m:

        candidate = (
            m.group(1),
            f"{int(m.group(2)):03d}"
        )

        if candidate in gold_docs:
            return candidate

    raise RuntimeError(
        f"cannot map pair doc_key: "
        f"{doc_key}"
    )


class UnionFind:

    def __init__(self, items):

        self.parent = {
            x: x
            for x in items
        }

    def find(self, x):

        while self.parent[x] != x:

            self.parent[x] = (
                self.parent[
                    self.parent[x]
                ]
            )

            x = self.parent[x]

        return x

    def union(self, a, b):

        ra = self.find(a)
        rb = self.find(b)

        if ra != rb:
            self.parent[rb] = ra


def main():

    args = args_parser()

    (
        lines,
        gold_docs,
        gold_mentions,
        token_line
    ) = parse_gold(
        args.gold
    )

    pair_rows = defaultdict(list)

    with open(
        args.pair_scores,
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            doc = map_doc_key(
                x["doc_key"],
                gold_docs
            )

            a = tuple(
                x["global_span1"]
            )

            b = tuple(
                x["global_span2"]
            )

            if a not in gold_mentions[doc]:

                raise RuntimeError(
                    f"{doc}: span1 not gold: {a}"
                )

            if b not in gold_mentions[doc]:

                raise RuntimeError(
                    f"{doc}: span2 not gold: {b}"
                )

            if a == b:
                raise RuntimeError(
                    f"{doc}: self pair {a}"
                )

            # Ensure antecedent precedes anaphor.
            if a < b:

                ant = a
                ana = b

            else:

                ant = b
                ana = a

            pair_rows[doc].append(
                (
                    ant,
                    ana,
                    float(x["p_coref"])
                )
            )

    all_clusters = {}

    total_links = 0
    total_clusters = 0
    singleton_clusters = 0

    for doc in gold_docs:

        mentions = sorted(
            gold_mentions[doc]
        )

        uf = UnionFind(
            mentions
        )

        # Best antecedent for each anaphor.
        best = {}

        for ant, ana, prob in pair_rows[
            doc
        ]:

            old = best.get(ana)

            if (
                old is None
                or prob > old[0]
                or (
                    prob == old[0]
                    and ant > old[1]
                )
            ):

                best[ana] = (
                    prob,
                    ant
                )

        for ana, (
            prob,
            ant
        ) in best.items():

            if prob >= args.threshold:

                uf.union(
                    ant,
                    ana
                )

                total_links += 1

        grouped = defaultdict(list)

        for mention in mentions:

            grouped[
                uf.find(mention)
            ].append(
                mention
            )

        clusters = sorted(
            [
                sorted(v)
                for v
                in grouped.values()
            ],
            key=lambda xs: xs[0]
        )

        all_clusters[
            doc
        ] = clusters

        total_clusters += len(
            clusters
        )

        singleton_clusters += sum(
            len(c) == 1
            for c in clusters
        )

    # Build CoNLL markers.
    marker_map = defaultdict(
        lambda: {
            "single": [],
            "open": [],
            "close": []
        }
    )

    for doc, clusters in all_clusters.items():

        for cid, cluster in enumerate(
            clusters,
            start=1
        ):

            for start, end in cluster:

                key_start = (
                    doc,
                    start
                )

                key_end = (
                    doc,
                    end
                )

                if start == end:

                    marker_map[
                        key_start
                    ][
                        "single"
                    ].append(cid)

                else:

                    marker_map[
                        key_start
                    ][
                        "open"
                    ].append(
                        (
                            end,
                            cid
                        )
                    )

                    marker_map[
                        key_end
                    ][
                        "close"
                    ].append(
                        (
                            start,
                            cid
                        )
                    )

    output_lines = list(lines)

    for (
        doc,
        token_idx
    ), line_no in token_line.items():

        ev = marker_map[
            (
                doc,
                token_idx
            )
        ]

        marks = []

        # Longer spans open first.
        for end, cid in sorted(
            ev["open"],
            key=lambda x: (
                -x[0],
                x[1]
            )
        ):

            marks.append(
                f"({cid}"
            )

        for cid in sorted(
            ev["single"]
        ):

            marks.append(
                f"({cid})"
            )

        # Inner spans close first.
        for start, cid in sorted(
            ev["close"],
            key=lambda x: (
                -x[0],
                x[1]
            )
        ):

            marks.append(
                f"{cid})"
            )

        coref = (
            "|".join(marks)
            if marks
            else "-"
        )

        old = output_lines[
            line_no
        ].rstrip("\n")

        prefix, _old_coref = (
            old.rsplit(
                None,
                1
            )
        )

        output_lines[
            line_no
        ] = (
            prefix
            + " "
            + coref
            + "\n"
        )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.write_text(
        "".join(output_lines),
        encoding="utf-8"
    )

    summary = {
        "pair_scores":
            args.pair_scores,

        "gold":
            args.gold,

        "threshold":
            args.threshold,

        "documents":
            len(gold_docs),

        "gold_mentions":
            sum(
                len(v)
                for v
                in gold_mentions.values()
            ),

        "selected_best_antecedent_links":
            total_links,

        "predicted_clusters":
            total_clusters,

        "singleton_clusters":
            singleton_clusters
    }

    with open(
        str(output)
        + ".summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "[PASS] pair scores -> CoNLL complete"
    )


if __name__ == "__main__":
    main()
