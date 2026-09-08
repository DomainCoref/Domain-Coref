#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
from pathlib import Path


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--strict-eval", required=True)
    ap.add_argument("--normalized-eval", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--mention-map", required=True)
    ap.add_argument("--token-map", required=True)

    args = ap.parse_args()

    strict = load_module(
        args.strict_eval,
        "strict_zero"
    )

    norm = load_module(
        args.normalized_eval,
        "norm_zero"
    )

    gold = strict.parse_conll(
        args.gold
    )

    selected, all_zero = (
        strict.load_zero_rows(
            args.mention_map,
            args.token_map,
            "test",
            "ALL"
        )
    )

    anchors = norm.structural_zero_anchors(
        args.token_map,
        "test"
    )

    # --------------------------------------------------------
    # 构造 pairized prediction：
    # gold cluster structure 不变，
    # 但所有 atomic Zero key (t,t)
    # 改成 (t,t+1)。
    # --------------------------------------------------------

    pclusters = {}
    pm2c = {}

    changed = 0

    for (doc, cid), members in gold["clusters"].items():

        new_members = set()

        for m in members:

            d, s, st, en = m

            new_m = m

            if (
                m in all_zero
                and (d, s, st) in anchors
                and gold["tokens"].get(
                    (d, s, st)
                ) == "Ø"
                and gold["tokens"].get(
                    (d, s, st + 1)
                ) == "Ø"
            ):
                new_m = (
                    d,
                    s,
                    st,
                    st + 1
                )
                changed += 1

            new_members.add(new_m)
            pm2c[new_m] = str(cid)

        pclusters[(doc, str(cid))] = (
            new_members
        )

    pairized = {
        "tokens": dict(gold["tokens"]),
        "docs": list(gold["docs"]),
        "clusters": pclusters,
        "m2c": pm2c
    }

    strict_result = strict.evaluate(
        gold,
        pairized,
        selected,
        all_zero
    )

    normalized_pred, changed_rows, _ = (
        norm.normalize_prediction(
            pairized,
            anchors
        )
    )

    norm_result = strict.evaluate(
        gold,
        normalized_pred,
        selected,
        all_zero
    )

    print("=" * 72)
    print("PAIRIZED ZERO =", changed)

    print()
    print("STRICT")
    print(
        "covered_zero =",
        strict_result["covered_zero"]
    )
    print(
        "resolution_f1 =",
        strict_result["resolution_f1"]
    )
    print(
        "pair_f1 =",
        strict_result["pair_f1"]
    )

    print()
    print("NORMALIZED")
    print(
        "normalized_mentions =",
        len(changed_rows)
    )
    print(
        "covered_zero =",
        norm_result["covered_zero"]
    )
    print(
        "resolution_f1 =",
        norm_result["resolution_f1"]
    )
    print(
        "pair_f1 =",
        norm_result["pair_f1"]
    )


if __name__ == "__main__":
    main()
