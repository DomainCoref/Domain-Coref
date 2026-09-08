#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys


BEGIN_RE = re.compile(r'^#begin document \(([^)]+)\)')


def load_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.append(x)

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate doc IDs found in: {}".format(path))

    return ids


def read_conll_docs(path):
    docs = {}
    order = []

    current_id = None
    current_lines = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            m = BEGIN_RE.match(line)

            if m:
                if current_id is not None:
                    raise ValueError(
                        "New document begins before previous document ends: {}"
                        .format(current_id)
                    )

                current_id = m.group(1)
                current_lines = [line]
                continue

            if current_id is not None:
                current_lines.append(line)

                if line.startswith("#end document"):
                    if current_id in docs:
                        raise ValueError(
                            "Duplicate document in CoNLL file: {}"
                            .format(current_id)
                        )

                    docs[current_id] = list(current_lines)
                    order.append(current_id)

                    current_id = None
                    current_lines = []

    if current_id is not None:
        raise ValueError(
            "Unclosed document at end of file: {}".format(current_id)
        )

    return docs, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--doc-ids", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    wanted = load_ids(args.doc_ids)
    docs, order = read_conll_docs(args.input)

    missing = [x for x in wanted if x not in docs]

    if missing:
        sys.stderr.write(
            "ERROR: {} requested documents are missing.\n".format(len(missing))
        )
        for x in missing[:20]:
            sys.stderr.write("  {}\n".format(x))
        sys.exit(2)

    wanted_set = set(wanted)

    # Preserve original CoNLL document order.
    selected = [doc_id for doc_id in order if doc_id in wanted_set]

    if len(selected) != len(wanted):
        raise RuntimeError(
            "Selected {} docs but expected {}."
            .format(len(selected), len(wanted))
        )

    with open(args.output, "w", encoding="utf-8") as out:
        for doc_id in selected:
            out.writelines(docs[doc_id])

    print("input_docs={}".format(len(order)))
    print("requested_docs={}".format(len(wanted)))
    print("written_docs={}".format(len(selected)))
    print("output={}".format(args.output))


if __name__ == "__main__":
    main()
