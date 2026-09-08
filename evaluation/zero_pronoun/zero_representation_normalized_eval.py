#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import importlib.util
import json
from pathlib import Path


TRUE = {"1", "true", "yes", "y"}


def yes(v):
    return str(v).strip().lower() in TRUE


def load_module(path):
    spec = importlib.util.spec_from_file_location(
        "strict_zero",
        path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def structural_zero_anchors(token_map, split="test"):
    anchors = set()

    with open(
        token_map,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        for r in csv.DictReader(f):

            if r.get("mode", "").strip() != "with_zero":
                continue

            if r.get("split", "").strip().lower() != split:
                continue

            if not r.get("zero_mention_id", "").strip():
                continue

            anchors.add((
                r["doc_id"].strip(),
                int(r["sentence_id"]),
                int(r["token_id"])
            ))

    return anchors


def normalize_prediction(
    pred,
    structural_anchors
):
    """
    唯一允许的 normalization：

      Pred mention = (doc,sent,t,t+1)
      token[t] == Ø
      token[t+1] == Ø
      (doc,sent,t) 是 structural Zero anchor

    则将 mention key canonicalize 为：
      (doc,sent,t,t)

    不改变 cluster ID。
    不处理任意更长 span。
    """

    new_clusters = {}
    new_m2c = {}

    normalized = []
    collisions = []

    for (doc, cid), members in pred["clusters"].items():

        out_members = set()

        for m in members:

            d, s, st, en = m
            new_m = m

            anchor = (d, s, st)

            if (
                en == st + 1
                and anchor in structural_anchors
                and pred["tokens"].get(
                    (d, s, st)
                ) == "Ø"
                and pred["tokens"].get(
                    (d, s, en)
                ) == "Ø"
            ):
                new_m = (
                    d,
                    s,
                    st,
                    st
                )

                normalized.append(
                    (m, new_m, cid)
                )

            if new_m in new_m2c:
                old_cid = new_m2c[new_m]

                if old_cid != str(cid):
                    collisions.append(
                        (
                            new_m,
                            old_cid,
                            str(cid)
                        )
                    )
                    raise RuntimeError(
                        "canonicalization collision: "
                        f"{new_m} "
                        f"{old_cid} vs {cid}"
                    )

            new_m2c[new_m] = str(cid)
            out_members.add(new_m)

        new_clusters[(doc, str(cid))] = out_members

    out = {
        "tokens": dict(pred["tokens"]),
        "docs": list(pred["docs"]),
        "clusters": new_clusters,
        "m2c": new_m2c
    }

    return out, normalized, collisions


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--strict-eval",
        required=True
    )

    ap.add_argument(
        "--gold",
        required=True
    )

    ap.add_argument(
        "--response",
        required=True
    )

    ap.add_argument(
        "--mention-map",
        required=True
    )

    ap.add_argument(
        "--token-map",
        required=True
    )

    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"]
    )

    ap.add_argument(
        "--semantics",
        default="ALL",
        choices=["ALL", "ENTITY", "EVENT"]
    )

    ap.add_argument(
        "--model",
        default="model"
    )

    ap.add_argument(
        "--expect-zero",
        type=int
    )

    ap.add_argument(
        "--out-json"
    )

    ap.add_argument(
        "--out-csv"
    )

    args = ap.parse_args()

    zmod = load_module(
        args.strict_eval
    )

    gold = zmod.parse_conll(
        args.gold
    )

    pred_raw = zmod.parse_conll(
        args.response
    )

    zmod.check_alignment(
        gold,
        pred_raw
    )

    selected_zero, all_zero = (
        zmod.load_zero_rows(
            args.mention_map,
            args.token_map,
            args.split,
            args.semantics
        )
    )

    if (
        args.expect_zero is not None
        and len(selected_zero)
        != args.expect_zero
    ):
        raise RuntimeError(
            f"unexpected Zero count: "
            f"{len(selected_zero)} "
            f"!= {args.expect_zero}"
        )

    anchors = structural_zero_anchors(
        args.token_map,
        args.split
    )

    pred_norm, normalized, collisions = (
        normalize_prediction(
            pred_raw,
            anchors
        )
    )

    result = zmod.evaluate(
        gold,
        pred_norm,
        selected_zero,
        all_zero
    )

    result["normalization"] = {
        "rule":
            "exact_pred_span_[t,t+1]_with_tokens_ØØ"
            "_and_t_structural_zero_anchor_to_[t,t]",
        "normalized_prediction_mentions":
            len(normalized),
        "canonicalization_collisions":
            len(collisions)
    }

    print(
        "[PASS] CoNLL document/token alignment"
    )

    print(
        "[PASS] Zero metadata/token-map cross-check"
    )

    print(
        "[PASS] representation normalization"
    )

    print("=" * 72)

    print(
        f"MODEL = {args.model}"
    )

    print(
        f"ZERO_SEMANTICS = {args.semantics}"
    )

    print(
        "NORMALIZED_PRED_MENTIONS = "
        f"{len(normalized)}"
    )

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
        print(
            f"{k} = {result[k]}"
        )

    if args.out_json:

        out = Path(args.out_json)
        out.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        save = dict(result)
        save.pop("details", None)

        with out.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                save,
                f,
                ensure_ascii=False,
                indent=2
            )

    if args.out_csv:
        zmod.write_csv(
            args.out_csv,
            result["details"]
        )


if __name__ == "__main__":
    main()
