#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path


def load_jsonl(path):
    docs = []

    with open(
        path,
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                docs.append(
                    json.loads(line)
                )

    return docs


def write_jsonl(path, docs):
    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:
        for doc in docs:
            f.write(
                json.dumps(
                    doc,
                    ensure_ascii=False
                )
                + "\n"
            )


def load_xu_utils(xu_root):
    path = (
        Path(xu_root)
        / "data"
        / "utils.py"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "xu2022_data_utils",
            path
        )
    )

    mod = (
        importlib.util
        .module_from_spec(spec)
    )

    spec.loader.exec_module(mod)

    return mod


def find_conflicts(docs):
    conflicts = []

    for doc in docs:
        events = sorted(
            doc["events"],
            key=lambda x: x["start"]
        )

        for a, b in zip(
            events,
            events[1:]
        ):
            same = (
                a["start"] == b["start"]
                and
                a["trigger"] == b["trigger"]
            )

            overlap = (
                a["start"]
                + len(a["trigger"])
                > b["start"]
            )

            if same or overlap:
                conflicts.append({
                    "doc_id":
                        doc["doc_id"],
                    "a_event":
                        a["event_id"],
                    "a_start":
                        a["start"],
                    "a_trigger":
                        a["trigger"],
                    "b_event":
                        b["event_id"],
                    "b_start":
                        b["start"],
                    "b_trigger":
                        b["trigger"],
                    "reason":
                        (
                            "SAME"
                            if same
                            else "OVERLAP"
                        )
                })

    return conflicts


def validate_clusters(docs):
    for doc in docs:

        event_ids = {
            e["event_id"]
            for e in doc["events"]
        }

        membership = {}

        for cluster in doc["clusters"]:
            for event_id in cluster["events"]:

                if event_id not in event_ids:
                    raise RuntimeError(
                        "cluster contains "
                        "unknown event: "
                        f"{doc['doc_id']} "
                        f"{event_id}"
                    )

                membership.setdefault(
                    event_id,
                    []
                ).append(
                    cluster["hopper_id"]
                )

        for event_id in event_ids:
            n = len(
                membership.get(
                    event_id,
                    []
                )
            )

            if n != 1:
                raise RuntimeError(
                    f"{doc['doc_id']} "
                    f"{event_id}: "
                    f"cluster_membership={n}"
                )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--xu-root",
        required=True
    )

    p.add_argument(
        "--data-dir",
        required=True
    )

    p.add_argument(
        "--audit-dir",
        required=True
    )

    a = p.parse_args()

    data_dir = Path(a.data_dir)

    audit_dir = Path(a.audit_dir)

    audit_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    xu_utils = load_xu_utils(
        a.xu_root
    )

    removed_rows = []

    summary = {}

    for split in [
        "train",
        "dev",
        "test"
    ]:

        raw_path = (
            data_dir
            / f"{split}.json"
        )

        filtered_path = (
            data_dir
            / f"{split}_filtered.json"
        )

        raw_docs = load_jsonl(
            raw_path
        )

        before = copy.deepcopy(
            raw_docs
        )

        filtered = (
            xu_utils.filter_events(
                copy.deepcopy(raw_docs),
                split
            )
        )

        # Run author's own conflict audit.
        xu_utils.check_event_conflict(
            filtered
        )

        conflicts = find_conflicts(
            filtered
        )

        if conflicts:
            raise RuntimeError(
                f"{split}: "
                f"{len(conflicts)} "
                "conflicts remain after "
                "Xu2022 official filtering"
            )

        validate_clusters(
            filtered
        )

        before_map = {
            doc["doc_id"]: {
                e["event_id"]: e
                for e in doc["events"]
            }
            for doc in before
        }

        after_map = {
            doc["doc_id"]: {
                e["event_id"]: e
                for e in doc["events"]
            }
            for doc in filtered
        }

        removed = 0

        for doc_id, events in before_map.items():

            after_ids = set(
                after_map.get(
                    doc_id,
                    {}
                )
            )

            missing = [
                e
                for event_id, e
                in events.items()
                if event_id
                not in after_ids
            ]

            missing.sort(
                key=lambda e: (
                    e["start"],
                    e["event_id"]
                )
            )

            for e in missing:

                removed += 1

                removed_rows.append({
                    "split": split,
                    "doc_id": doc_id,
                    "event_id":
                        e["event_id"],
                    "start":
                        e["start"],
                    "trigger":
                        e["trigger"],
                    "reason":
                        "XU2022_SAME_OR_OVERLAP"
                })

        raw_events = sum(
            len(d["events"])
            for d in before
        )

        filtered_events = sum(
            len(d["events"])
            for d in filtered
        )

        raw_clusters = sum(
            len(d["clusters"])
            for d in before
        )

        filtered_clusters = sum(
            len(d["clusters"])
            for d in filtered
        )

        summary[split] = {
            "docs":
                len(filtered),
            "raw_events":
                raw_events,
            "filtered_events":
                filtered_events,
            "removed_events":
                removed,
            "raw_clusters":
                raw_clusters,
            "filtered_clusters":
                filtered_clusters,
            "remaining_conflicts":
                len(conflicts),
        }

        write_jsonl(
            filtered_path,
            filtered
        )

        print()
        print(
            "==========",
            split,
            "=========="
        )

        for k, v in (
            summary[split]
        ).items():
            print(
                k.upper(),
                "=",
                v
            )

    audit_tsv = (
        audit_dir
        / "xu2022_removed_events.tsv"
    )

    with open(
        audit_tsv,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        fields = [
            "split",
            "doc_id",
            "event_id",
            "start",
            "trigger",
            "reason",
        ]

        w = csv.DictWriter(
            f,
            fieldnames=fields,
            delimiter="\t"
        )

        w.writeheader()
        w.writerows(
            removed_rows
        )

    with open(
        audit_dir
        / "xu2022_filter_summary.json",
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
        "[PASS] Xu2022 official "
        "filter complete"
    )


if __name__ == "__main__":
    main()
