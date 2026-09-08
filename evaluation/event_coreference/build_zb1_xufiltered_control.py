#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re

from itertools import combinations
from pathlib import Path


def read_jsonl(path):
    rows = []

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(line)
                )

    return rows


def normalize_doc_key(doc_key):

    m = re.match(
        r"^(.*)_([0-9]+)$",
        doc_key
    )

    if not m:
        raise RuntimeError(
            f"bad doc_key: {doc_key}"
        )

    return m.group(1)


def build_sentence_starts(
    gold_path
):

    starts = {}

    doc_id = None
    sentence_id = 1
    global_index = 0
    in_sentence = False

    begin_re = re.compile(
        r"#begin document \((.+?)\);"
    )

    with open(
        gold_path,
        encoding="utf-8"
    ) as f:

        for raw in f:

            line = raw.rstrip("\n")

            if line.startswith(
                "#begin document"
            ):

                m = begin_re.search(
                    line
                )

                if not m:
                    raise RuntimeError(
                        "cannot parse: "
                        + line
                    )

                doc_id = m.group(1)

                sentence_id = 1
                global_index = 0
                in_sentence = False

                continue

            if line.startswith(
                "#end document"
            ):

                doc_id = None
                continue

            if doc_id is None:
                continue

            if not line.strip():

                if in_sentence:

                    sentence_id += 1
                    in_sentence = False

                continue

            if line.startswith("#"):
                continue

            if not in_sentence:

                starts[
                    (
                        doc_id,
                        sentence_id
                    )
                ] = global_index

                in_sentence = True

            global_index += 1

    return starts


def load_event_spans(
    type_map,
    sentence_starts
):

    result = {}

    with open(
        type_map,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        for r in csv.DictReader(f):

            if (
                r["mode"] != "with_zero"
                or r["split"] != "test"
            ):
                continue

            mention_id = (
                r["mention_id"]
            )

            doc_id = r["doc_id"]

            sentence_id = int(
                r["sentence_id"]
            )

            token_start = int(
                r["token_start"]
            )

            token_end = int(
                r["token_end"]
            )

            key = (
                doc_id,
                sentence_id
            )

            if key not in sentence_starts:

                raise RuntimeError(
                    "missing sentence in gold: "
                    f"{key}"
                )

            base = (
                sentence_starts[key]
            )

            span = (
                base + token_start,
                base + token_end
            )

            result[mention_id] = {
                "doc_id":
                    doc_id,
                "span":
                    span,
                "mention_type":
                    r["mention_type"],
                "referent_semantics":
                    r[
                        "referent_semantics"
                    ],
            }

    return result


def get_cluster_objects(
    clusters
):

    if isinstance(
        clusters,
        list
    ):
        return clusters

    if isinstance(
        clusters,
        dict
    ):
        return list(
            clusters.values()
        )

    raise RuntimeError(
        "unknown clusters structure: "
        f"{type(clusters)}"
    )


def collect_event_ids(
    obj,
    valid_ids
):

    found = set()

    if isinstance(
        obj,
        str
    ):

        if obj in valid_ids:
            found.add(obj)

    elif isinstance(
        obj,
        dict
    ):

        for v in obj.values():

            found |= collect_event_ids(
                v,
                valid_ids
            )

    elif isinstance(
        obj,
        (list, tuple)
    ):

        for v in obj:

            found |= collect_event_ids(
                v,
                valid_ids
            )

    return found


def build_gold_cluster_map(
    doc
):

    events = doc["events"]

    event_ids = [
        e["event_id"]
        for e in events
    ]

    valid_ids = set(
        event_ids
    )

    cluster_map = {}

    cluster_objects = (
        get_cluster_objects(
            doc["clusters"]
        )
    )

    next_cluster = 0

    for c in cluster_objects:

        ids = collect_event_ids(
            c,
            valid_ids
        )

        # Fallback only when the cluster
        # is explicitly represented as
        # a list of event indexes.
        if (
            not ids
            and isinstance(c, list)
            and c
            and all(
                isinstance(x, int)
                and 0 <= x < len(events)
                for x in c
            )
        ):

            ids = {
                event_ids[x]
                for x in c
            }

        if not ids:
            continue

        for event_id in ids:

            if event_id in cluster_map:

                raise RuntimeError(
                    "event belongs to "
                    "multiple clusters: "
                    + event_id
                )

            cluster_map[
                event_id
            ] = next_cluster

        next_cluster += 1

    # Explicit singleton events may not
    # appear in a cluster list.
    for event_id in event_ids:

        if event_id not in cluster_map:

            cluster_map[
                event_id
            ] = next_cluster

            next_cluster += 1

    return cluster_map


def load_zb1_scores(
    path
):

    scores = {}

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            r = json.loads(
                line
            )

            doc_id = (
                normalize_doc_key(
                    r["doc_key"]
                )
            )

            s1 = tuple(
                r["global_span1"]
            )

            s2 = tuple(
                r["global_span2"]
            )

            pair = tuple(
                sorted(
                    [s1, s2]
                )
            )

            key = (
                doc_id,
                pair[0],
                pair[1]
            )

            if key in scores:

                raise RuntimeError(
                    "duplicate ZB1 pair: "
                    f"{key}"
                )

            scores[key] = float(
                r["p_coref"]
            )

    return scores


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--xu-filtered",
        required=True
    )

    p.add_argument(
        "--type-map",
        required=True
    )

    p.add_argument(
        "--gold-conll",
        required=True
    )

    p.add_argument(
        "--zb1-scores",
        required=True
    )

    p.add_argument(
        "--threshold",
        type=float,
        default=0.50
    )

    p.add_argument(
        "--output-pred",
        required=True
    )

    p.add_argument(
        "--output-metrics",
        required=True
    )

    p.add_argument(
        "--output-missing",
        required=True
    )

    args = p.parse_args()

    xu_docs = read_jsonl(
        args.xu_filtered
    )

    sentence_starts = (
        build_sentence_starts(
            args.gold_conll
        )
    )

    mention_map = (
        load_event_spans(
            args.type_map,
            sentence_starts
        )
    )

    zb1_scores = (
        load_zb1_scores(
            args.zb1_scores
        )
    )

    total_docs = len(
        xu_docs
    )

    total_events = sum(
        len(d["events"])
        for d in xu_docs
    )

    total_pairs = sum(
        len(d["events"])
        * (
            len(d["events"]) - 1
        )
        // 2
        for d in xu_docs
    )

    print(
        "DOCS =",
        total_docs
    )

    print(
        "EVENTS =",
        total_events
    )

    print(
        "PAIRS =",
        total_pairs
    )

    if total_docs != 436:
        raise RuntimeError(
            f"expected 436 docs, "
            f"got {total_docs}"
        )

    if total_events != 1502:
        raise RuntimeError(
            f"expected 1502 events, "
            f"got {total_events}"
        )

    if total_pairs != 2296:
        raise RuntimeError(
            f"expected 2296 pairs, "
            f"got {total_pairs}"
        )

    tp = fp = fn = tn = 0

    gold_positive = 0

    matched_pairs = 0
    missing_pairs = 0
    missing_positive = 0
    missing_negative = 0

    output_docs = []
    missing_rows = []

    mapped_event_ids = set()

    for doc in xu_docs:

        doc_id = doc["doc_id"]

        events = doc["events"]

        cluster_map = (
            build_gold_cluster_map(
                doc
            )
        )

        event_spans = {}

        for e in events:

            event_id = (
                e["event_id"]
            )

            if event_id not in mention_map:

                raise RuntimeError(
                    "Xu event missing from "
                    "type map: "
                    + event_id
                )

            info = (
                mention_map[
                    event_id
                ]
            )

            if (
                info["doc_id"]
                != doc_id
            ):

                raise RuntimeError(
                    "doc mismatch for "
                    + event_id
                )

            if (
                info["mention_type"]
                != "事件提及"
            ):

                raise RuntimeError(
                    "Xu Explicit Event "
                    "is not 事件提及: "
                    + event_id
                )

            event_spans[
                event_id
            ] = info["span"]

            mapped_event_ids.add(
                event_id
            )

        pred_labels = []
        pred_probs = []

        for i, j in combinations(
            range(len(events)),
            2
        ):

            e1 = events[i]
            e2 = events[j]

            id1 = e1["event_id"]
            id2 = e2["event_id"]

            s1 = event_spans[id1]
            s2 = event_spans[id2]

            pair = tuple(
                sorted(
                    [s1, s2]
                )
            )

            key = (
                doc_id,
                pair[0],
                pair[1]
            )

            gold = int(
                cluster_map[id1]
                == cluster_map[id2]
            )

            gold_positive += gold

            if key in zb1_scores:

                prob = (
                    zb1_scores[key]
                )

                matched_pairs += 1

            else:

                # Candidate pair absent
                # from ZB1's fixed
                # 300-token pair universe.
                # It stays in the Xu test
                # universe and is treated
                # as non-coreferent.
                prob = 0.0

                missing_pairs += 1

                if gold:
                    missing_positive += 1
                else:
                    missing_negative += 1

                missing_rows.append({
                    "doc_id":
                        doc_id,
                    "event1":
                        id1,
                    "event2":
                        id2,
                    "span1":
                        list(s1),
                    "span2":
                        list(s2),
                    "gold":
                        gold,
                })

            pred = int(
                prob >= args.threshold
            )

            pred_labels.append(
                pred
            )

            pred_probs.append(
                prob
            )

            if gold == 1 and pred == 1:
                tp += 1
            elif gold == 0 and pred == 1:
                fp += 1
            elif gold == 1 and pred == 0:
                fn += 1
            else:
                tn += 1

        out_events = []

        for e in events:

            out_events.append({
                "start":
                    e["start"],
                "end":
                    e["start"]
                    + len(
                        e["trigger"]
                    )
                    - 1,
                "trigger":
                    e["trigger"]
            })

        output_docs.append({
            "doc_id":
                doc_id,
            "document":
                doc["document"],
            "events":
                out_events,
            "pred_label":
                pred_labels,
            "pred_prob":
                pred_probs,
        })

    if len(mapped_event_ids) != 1502:

        raise RuntimeError(
            "event mapping coverage "
            f"{len(mapped_event_ids)} "
            "!= 1502"
        )

    # Official Xu TEST showed
    # positive support = 1064.
    # This is an important independent
    # gold-universe consistency gate.
    if gold_positive != 1064:

        raise RuntimeError(
            "gold positive pair count "
            f"{gold_positive} != 1064; "
            "STOP before evaluation"
        )

    predicted_positive = (
        tp + fp
    )

    precision = (
        tp / predicted_positive
        if predicted_positive
        else 0.0
    )

    recall = (
        tp / (
            tp + fn
        )
        if (
            tp + fn
        )
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        )
        else 0.0
    )

    accuracy = (
        (
            tp + tn
        )
        / total_pairs
    )

    metrics = {
        "model":
            "ZB1-General",
        "track":
            "Explicit_Event_XuFiltered",
        "threshold":
            args.threshold,
        "documents":
            total_docs,
        "events":
            total_events,
        "pairs":
            total_pairs,
        "gold_positive_pairs":
            gold_positive,
        "gold_negative_pairs":
            (
                total_pairs
                - gold_positive
            ),
        "matched_zb1_pairs":
            matched_pairs,
        "missing_zb1_pairs":
            missing_pairs,
        "missing_positive_pairs":
            missing_positive,
        "missing_negative_pairs":
            missing_negative,
        "pair_coverage":
            matched_pairs
            / total_pairs,
        "tp":
            tp,
        "fp":
            fp,
        "fn":
            fn,
        "tn":
            tn,
        "precision":
            precision,
        "recall":
            recall,
        "f1":
            f1,
        "accuracy":
            accuracy,
    }

    Path(
        args.output_pred
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        args.output_pred,
        "w",
        encoding="utf-8"
    ) as f:

        for d in output_docs:

            f.write(
                json.dumps(
                    d,
                    ensure_ascii=False
                )
                + "\n"
            )

    with open(
        args.output_metrics,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )

        f.write("\n")

    with open(
        args.output_missing,
        "w",
        encoding="utf-8"
    ) as f:

        for r in missing_rows:

            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False
                )
                + "\n"
            )

    print()
    print(
        "===== ZB1 SAME-UNIVERSE ====="
    )

    print(
        "EVENT_MAPPING =",
        len(mapped_event_ids),
        "/",
        total_events
    )

    print(
        "MATCHED_ZB1_PAIRS =",
        matched_pairs
    )

    print(
        "MISSING_ZB1_PAIRS =",
        missing_pairs
    )

    print(
        "PAIR_COVERAGE =",
        round(
            matched_pairs
            / total_pairs,
            6
        )
    )

    print(
        "GOLD_POSITIVE =",
        gold_positive
    )

    print(
        "TP =",
        tp,
        "FP =",
        fp,
        "FN =",
        fn,
        "TN =",
        tn
    )

    print(
        "PAIR_P =",
        round(
            precision * 100,
            4
        )
    )

    print(
        "PAIR_R =",
        round(
            recall * 100,
            4
        )
    )

    print(
        "PAIR_F1 =",
        round(
            f1 * 100,
            4
        )
    )

    print(
        "PAIR_ACC =",
        round(
            accuracy * 100,
            4
        )
    )

    print()
    print(
        "[PASS] ZB1 Xu-filtered "
        "same-universe prediction complete"
    )


if __name__ == "__main__":
    main()
