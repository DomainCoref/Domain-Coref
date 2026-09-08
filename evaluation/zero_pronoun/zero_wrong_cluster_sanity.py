#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import importlib.util
from collections import defaultdict
from pathlib import Path


def load_module(path):
    spec = importlib.util.spec_from_file_location("strict_zero", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--strict-eval", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--mention-map", required=True)
    ap.add_argument("--token-map", required=True)

    ap.add_argument(
        "--semantics",
        default="ALL",
        choices=["ALL", "ENTITY", "EVENT"]
    )

    ap.add_argument("--expect-zero", type=int)

    args = ap.parse_args()

    zmod = load_module(args.strict_eval)

    gold = zmod.parse_conll(args.gold)

    selected_zero, all_zero = zmod.load_zero_rows(
        args.mention_map,
        args.token_map,
        "test",
        args.semantics
    )

    if (
        args.expect_zero is not None
        and len(selected_zero) != args.expect_zero
    ):
        raise RuntimeError(
            f"Zero count mismatch: "
            f"{len(selected_zero)} != {args.expect_zero}"
        )

    # --------------------------------------------------------
    # Gold copy -> corrupted prediction structure
    # --------------------------------------------------------

    pred = {
        "tokens": dict(gold["tokens"]),
        "docs": list(gold["docs"]),
        "clusters": {
            k: set(v)
            for k, v in gold["clusters"].items()
        },
        "m2c": dict(gold["m2c"])
    }

    # 每篇文档可用的 non-zero clusters。
    doc_nonzero_clusters = defaultdict(list)

    for (doc, cid), members in pred["clusters"].items():

        nonzero = [
            m for m in members
            if m not in all_zero
        ]

        if nonzero:
            doc_nonzero_clusters[doc].append(
                (cid, set(nonzero))
            )

    moved = 0
    skipped = 0

    # --------------------------------------------------------
    # 对每个 Zero：
    # 从原 gold cluster 移除；
    # 放到同文档另一个含 non-zero mention 的错误 cluster。
    # 所有 span 本身仍然保留。
    # --------------------------------------------------------

    for z in sorted(selected_zero):

        doc = z[0]
        old_cid = pred["m2c"][z]

        candidates = [
            (cid, members)
            for cid, members
            in doc_nonzero_clusters[doc]
            if cid != old_cid
        ]

        if not candidates:
            skipped += 1
            continue

        # deterministic:
        # 取排序后第一个不同 cluster。
        candidates.sort(
            key=lambda x: str(x[0])
        )

        new_cid = candidates[0][0]

        pred["clusters"][(doc, old_cid)].discard(z)
        pred["clusters"][(doc, new_cid)].add(z)
        pred["m2c"][z] = new_cid

        moved += 1

    result = zmod.evaluate(
        gold,
        pred,
        selected_zero,
        all_zero
    )

    print("=" * 72)
    print("SANITY = WRONG_CLUSTER")
    print("ZERO =", len(selected_zero))
    print("MOVED_ZERO =", moved)
    print("SKIPPED_ZERO =", skipped)

    for k in [
        "gold_zero",
        "gold_resolvable_zero",
        "covered_zero",
        "cluster_coverage",
        "pred_resolved_zero",
        "correct_resolved_zero",
        "resolution_precision",
        "resolution_recall",
        "resolution_f1",
        "gold_zero_pairs",
        "pred_zero_pairs",
        "correct_zero_pairs",
        "pair_precision",
        "pair_recall",
        "pair_f1"
    ]:
        print(f"{k} = {result[k]}")


if __name__ == "__main__":
    main()
