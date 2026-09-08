#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path


def norm_text(s):
    return "".join(
        str(s).split()
    ).lower()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True
    )

    ap.add_argument(
        "--output",
        required=True
    )

    args = ap.parse_args()

    out_path = Path(args.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    n = 0
    same = 0

    with open(
        args.input,
        encoding="utf-8"
    ) as fin, open(
        args.output,
        "w",
        encoding="utf-8"
    ) as fout:

        for line in fin:

            if not line.strip():
                continue

            x = json.loads(line)

            meta = x["_meta"]
            target = x["target"]

            t1 = norm_text(
                target["span1_text"]
            )

            t2 = norm_text(
                target["span2_text"]
            )

            pred = (
                1.0
                if t1 == t2
                else 0.0
            )

            if pred == 1.0:
                same += 1

            y = {
                "doc_key":
                    meta["doc_key"],

                "global_span1":
                    meta["global_span1"],

                "global_span2":
                    meta["global_span2"],

                "p_coref":
                    pred
            }

            fout.write(
                json.dumps(
                    y,
                    ensure_ascii=False
                )
                + "\n"
            )

            n += 1

    print(
        "ROWS =",
        n
    )

    print(
        "EXACT_STRING_POSITIVE_PAIRS =",
        same
    )

    print(
        "[PASS] exact-string scores built"
    )


if __name__ == "__main__":
    main()
