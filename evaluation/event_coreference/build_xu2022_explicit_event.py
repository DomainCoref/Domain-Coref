#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from collections import Counter


EXPECTED = {
    "train": 5178,
    "dev": 738,
    "test": 1507,
}


def locate_sentences(document, sentences):
    offsets = []
    cursor = 0

    for sid, sent in enumerate(sentences):
        pos = document.find(sent, cursor)

        if pos < 0:
            raise RuntimeError(
                f"sentence not found: sid={sid}, "
                f"cursor={cursor}, sent={sent[:80]!r}"
            )

        offsets.append(pos)
        cursor = pos + len(sent)

    return offsets


def load_docmap(path):
    rows = []

    with open(
        path,
        encoding="utf-8-sig",
        newline=""
    ) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    if len(rows) != 2241:
        raise RuntimeError(
            f"doc_map rows != 2241: {len(rows)}"
        )

    gids = [
        int(r["global_doc_index"])
        for r in rows
    ]

    if len(set(gids)) != len(gids):
        raise RuntimeError(
            "duplicate global_doc_index"
        )

    docids = [
        r["doc_id"]
        for r in rows
    ]

    if len(set(docids)) != len(docids):
        raise RuntimeError(
            "duplicate doc_id"
        )

    return rows


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--frozen", required=True)
    p.add_argument("--doc-map", required=True)
    p.add_argument("--output-dir", required=True)

    a = p.parse_args()

    with open(
        a.frozen,
        encoding="utf-8"
    ) as f:
        frozen = json.load(f)

    docmap = load_docmap(a.doc_map)

    if len(frozen) != 2241:
        raise RuntimeError(
            f"frozen docs != 2241: {len(frozen)}"
        )

    outdir = Path(a.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {
        s: open(
            outdir / f"{s}.json",
            "w",
            encoding="utf-8"
        )
        for s in EXPECTED
    }

    stats = {
        s: Counter()
        for s in EXPECTED
    }

    doc_sets = {
        s: set()
        for s in EXPECTED
    }

    try:
        for r in docmap:
            split = r["split"]

            if split not in EXPECTED:
                continue

            idx = int(
                r["global_doc_index"]
            )

            doc_id = r["doc_id"]
            doc = frozen[idx]

            if (
                doc["domain"] != r["domain"]
                or
                doc["category_primary"]
                != r["category_primary"]
            ):
                raise RuntimeError(
                    f"doc_map mismatch: {doc_id}"
                )

            document = doc["text"]
            sentences = doc["sentences"]

            sent_offsets = locate_sentences(
                document,
                sentences
            )

            sentence_objs = [
                {
                    "start": sent_offsets[i],
                    "text": s
                }
                for i, s in enumerate(sentences)
            ]

            events = []
            clusters = []

            seen_spans = set()
            seen_event_ids = set()

            for c_pos, chain in enumerate(
                doc["coreference_chains"]
            ):
                cluster_event_ids = []

                chain_index = chain.get(
                    "index",
                    c_pos + 1
                )

                for m_pos, m in enumerate(
                    chain["mentions"]
                ):
                    if not (
                        m.get("mention_type")
                        == "事件提及"
                        and
                        m.get(
                            "referent_semantics"
                        )
                        == "EVENT"
                    ):
                        continue

                    sid = int(
                        m["sentence_id"]
                    ) - 1

                    if not (
                        0 <= sid
                        < len(sentences)
                    ):
                        raise RuntimeError(
                            f"bad sentence_id: "
                            f"{doc_id}, {m}"
                        )

                    sent = sentences[sid]

                    cs = int(
                        m["char_start"]
                    )

                    ce = int(
                        m["char_end"]
                    )

                    trigger = m["text"]

                    if (
                        cs < 0
                        or ce < cs
                        or ce >= len(sent)
                    ):
                        raise RuntimeError(
                            f"bad char span: "
                            f"{doc_id}, {m}"
                        )

                    actual = sent[
                        cs:ce + 1
                    ]

                    if actual != trigger:
                        raise RuntimeError(
                            f"sentence offset mismatch: "
                            f"{doc_id}, expected={trigger!r}, "
                            f"actual={actual!r}"
                        )

                    global_start = (
                        sent_offsets[sid]
                        + cs
                    )

                    global_end = (
                        sent_offsets[sid]
                        + ce
                    )

                    if (
                        document[
                            global_start:
                            global_end + 1
                        ]
                        != trigger
                    ):
                        raise RuntimeError(
                            f"document offset mismatch: "
                            f"{doc_id}, {trigger!r}"
                        )

                    span_key = (
                        global_start,
                        global_end
                    )

                    if span_key in seen_spans:
                        # Keep the frozen Domain-Coref annotation unchanged in the raw adapter.
                        # Xu2022 official filter_events() is applied afterward.
                        pass
                    else:
                        seen_spans.add(span_key)

                    event_id = (
                        f"{doc_id}_c"
                        f"{chain_index}_m"
                        f"{m_pos + 1}"
                    )

                    if event_id in seen_event_ids:
                        raise RuntimeError(
                            f"duplicate event id: "
                            f"{event_id}"
                        )

                    seen_event_ids.add(
                        event_id
                    )

                    events.append({
                        "event_id": event_id,

                        # Xu2022 document-level
                        # inclusive character start
                        "start": global_start,

                        # zero-based sentence index
                        "sent_idx": sid,

                        # sentence-relative,
                        # inclusive start
                        "sent_start": cs,

                        "trigger": trigger,

                        # compatibility field only;
                        # normal BertForPairwiseEC
                        # does not consume subtype
                        "subtype": "other"
                    })

                    cluster_event_ids.append(
                        event_id
                    )

                if cluster_event_ids:
                    clusters.append({
                        "hopper_id":
                            f"{doc_id}_h"
                            f"{chain_index}",

                        "events":
                            cluster_event_ids
                    })

            if not events:
                continue

            events.sort(
                key=lambda x:
                (x["start"], x["event_id"])
            )

            sample = {
                "doc_id": doc_id,
                "document": document,
                "sentences": sentence_objs,
                "events": events,
                "clusters": clusters
            }

            outputs[split].write(
                json.dumps(
                    sample,
                    ensure_ascii=False
                )
                + "\n"
            )

            doc_sets[split].add(
                doc_id
            )

            stats[split]["docs"] += 1
            stats[split]["events"] += (
                len(events)
            )
            stats[split]["clusters"] += (
                len(clusters)
            )

            # Number of all event pairs
            n = len(events)

            stats[split]["pairs"] += (
                n * (n - 1) // 2
            )

            event_to_cluster = {}

            for c in clusters:
                for e in c["events"]:
                    event_to_cluster[e] = (
                        c["hopper_id"]
                    )

            for i in range(n - 1):
                for j in range(
                    i + 1,
                    n
                ):
                    e1 = events[i]["event_id"]
                    e2 = events[j]["event_id"]

                    if (
                        event_to_cluster[e1]
                        ==
                        event_to_cluster[e2]
                    ):
                        stats[split][
                            "positive_pairs"
                        ] += 1
                    else:
                        stats[split][
                            "negative_pairs"
                        ] += 1

    finally:
        for f in outputs.values():
            f.close()

    for split in [
        "train",
        "dev",
        "test"
    ]:
        print()
        print(
            "==========",
            split,
            "=========="
        )

        for k in [
            "docs",
            "events",
            "clusters",
            "pairs",
            "positive_pairs",
            "negative_pairs",
        ]:
            print(
                f"{k.upper()} =",
                stats[split][k]
            )

        if (
            stats[split]["events"]
            != EXPECTED[split]
        ):
            raise RuntimeError(
                f"{split}: event count "
                f"{stats[split]['events']} "
                f"!= expected "
                f"{EXPECTED[split]}"
            )

    print()
    print(
        "TRAIN_DEV_DOC_OVERLAP =",
        len(
            doc_sets["train"]
            &
            doc_sets["dev"]
        )
    )

    print(
        "TRAIN_TEST_DOC_OVERLAP =",
        len(
            doc_sets["train"]
            &
            doc_sets["test"]
        )
    )

    print(
        "DEV_TEST_DOC_OVERLAP =",
        len(
            doc_sets["dev"]
            &
            doc_sets["test"]
        )
    )

    assert not (
        doc_sets["train"]
        &
        doc_sets["dev"]
    )

    assert not (
        doc_sets["train"]
        &
        doc_sets["test"]
    )

    assert not (
        doc_sets["dev"]
        &
        doc_sets["test"]
    )

    summary = {
        split: dict(stats[split])
        for split in stats
    }

    with open(
        outdir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        "[PASS] Xu2022 Explicit Event "
        "conversion complete"
    )


if __name__ == "__main__":
    main()
