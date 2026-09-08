#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

TRUE = {"1", "true", "yes", "y"}

BEGIN_RE = re.compile(
    r"^#begin document\s+\((.+?)\);\s*part\s+(\d+)"
)


def yes(v):
    return str(v).strip().lower() in TRUE


def div(a, b):
    return 0.0 if b == 0 else a / float(b)


def f1(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def mkey(doc, sent, start, end):
    return (
        str(doc),
        int(sent),
        int(start),
        int(end)
    )


def register(clusters, m2c, doc, cid, key):
    cid = str(cid)

    clusters[(doc, cid)].add(key)

    if key in m2c and m2c[key] != cid:
        raise ValueError(
            "mention belongs to multiple clusters: "
            f"{key}: {m2c[key]} vs {cid}"
        )

    m2c[key] = cid


def parse_conll(path):
    tokens = {}
    clusters = defaultdict(set)
    m2c = {}
    docs = []

    doc = None
    sent = 1
    sent_has_token = False
    opened = defaultdict(list)

    with open(path, encoding="utf-8") as f:

        for lineno, raw in enumerate(f, 1):

            line = raw.rstrip("\n")

            if line.startswith("#begin document"):

                m = BEGIN_RE.match(line)

                if not m:
                    raise ValueError(
                        f"bad begin line {lineno}: {line}"
                    )

                doc = m.group(1)
                docs.append(doc)

                sent = 1
                sent_has_token = False
                opened = defaultdict(list)
                continue

            if line.startswith("#end document"):

                remain = {
                    k: v
                    for k, v in opened.items()
                    if v
                }

                if remain:
                    raise ValueError(
                        f"unclosed spans in {doc}: {remain}"
                    )

                doc = None
                sent_has_token = False
                continue

            if not line.strip():

                if doc is not None and sent_has_token:
                    sent += 1
                    sent_has_token = False

                continue

            if line.startswith("#"):
                continue

            if doc is None:
                raise ValueError(
                    f"token outside document at line {lineno}"
                )

            cols = line.split()

            if len(cols) < 5:
                raise ValueError(
                    f"bad token line {lineno}: {line}"
                )

            try:
                tok = int(cols[2])
            except ValueError:
                raise ValueError(
                    f"bad token id line {lineno}: {cols[2]}"
                )

            word = cols[3]
            coref = cols[-1]

            tk = (doc, sent, tok)

            if tk in tokens:
                raise ValueError(
                    f"duplicate token key: {tk}"
                )

            tokens[tk] = word
            sent_has_token = True

            if coref in {"-", "_", "*"}:
                continue

            for atom in coref.split("|"):

                atom = atom.strip()

                if not atom or atom in {"-", "_", "*"}:
                    continue

                if atom.startswith("(") and atom.endswith(")"):

                    cid = atom[1:-1]

                    key = mkey(
                        doc, sent, tok, tok
                    )

                    register(
                        clusters, m2c,
                        doc, cid, key
                    )

                elif atom.startswith("("):

                    cid = atom[1:]

                    opened[cid].append(
                        (sent, tok)
                    )

                elif atom.endswith(")"):

                    cid = atom[:-1]

                    if not opened[cid]:
                        raise ValueError(
                            "close without open: "
                            f"line={lineno} cid={cid}"
                        )

                    start_sent, start_tok = (
                        opened[cid].pop()
                    )

                    if start_sent != sent:
                        raise ValueError(
                            "cross-sentence mention: "
                            f"doc={doc} cid={cid}"
                        )

                    key = mkey(
                        doc,
                        sent,
                        start_tok,
                        tok
                    )

                    register(
                        clusters, m2c,
                        doc, cid, key
                    )

                else:

                    raise ValueError(
                        "unknown coref atom: "
                        f"line={lineno} atom={atom}"
                    )

    return {
        "tokens": tokens,
        "clusters": dict(clusters),
        "m2c": m2c,
        "docs": docs
    }


def check_alignment(gold, pred):

    if gold["docs"] != pred["docs"]:
        raise ValueError(
            "gold/pred document order mismatch"
        )

    g = gold["tokens"]
    p = pred["tokens"]

    if set(g) != set(p):

        missing = sorted(
            set(g) - set(p)
        )[:10]

        extra = sorted(
            set(p) - set(g)
        )[:10]

        raise ValueError(
            "gold/pred token-key mismatch: "
            f"missing={missing} extra={extra}"
        )

    wrong = []

    for key in g:

        if g[key] != p[key]:

            wrong.append(
                (key, g[key], p[key])
            )

            if len(wrong) >= 10:
                break

    if wrong:
        raise ValueError(
            f"gold/pred token-text mismatch: {wrong}"
        )


def load_zero_rows(
    mention_map,
    token_map,
    split,
    semantics
):

    semantics = semantics.upper()
    split = split.lower()

    selected = {}
    all_zero = {}

    with open(
        mention_map,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for r in reader:

            if r.get("mode", "").strip() != "with_zero":
                continue

            if r.get("split", "").strip().lower() != split:
                continue

            if not yes(r.get("is_exported", "")):
                continue

            if not yes(r.get("is_zero", "")):
                continue

            key = mkey(
                r["doc_id"],
                r["sentence_id"],
                r["token_start"],
                r["token_end"]
            )

            sem = (
                r.get(
                    "referent_semantics", ""
                )
                .strip()
                .upper()
            )

            info = {
                "mention_id":
                    r["mention_id"].strip(),
                "gold_chain_id":
                    r["conll_chain_id"].strip(),
                "semantics":
                    sem
            }

            all_zero[key] = info

            if (
                semantics == "ALL"
                or sem == semantics
            ):
                selected[key] = info

    structural = set()

    with open(
        token_map,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for r in reader:

            if r.get("mode", "").strip() != "with_zero":
                continue

            if r.get("split", "").strip().lower() != split:
                continue

            if not r.get(
                "zero_mention_id", ""
            ).strip():
                continue

            structural.add(
                mkey(
                    r["doc_id"],
                    r["sentence_id"],
                    r["token_id"],
                    r["token_id"]
                )
            )

    if set(all_zero) != structural:

        missing = sorted(
            set(all_zero) - structural
        )[:10]

        extra = sorted(
            structural - set(all_zero)
        )[:10]

        raise ValueError(
            "mention_map/token_map Zero mismatch: "
            f"missing={missing} extra={extra}"
        )

    return selected, set(all_zero)


def evaluate(
    gold,
    pred,
    selected_zero,
    all_zero_keys
):

    gclusters = gold["clusters"]
    gm2c = gold["m2c"]

    pclusters = pred["clusters"]
    pm2c = pred["m2c"]

    zero_keys = set(selected_zero)

    for z in zero_keys:

        if z not in gm2c:
            raise ValueError(
                f"Zero missing from Gold: {z}"
            )

        meta_cid = (
            selected_zero[z][
                "gold_chain_id"
            ]
        )

        if gm2c[z] != meta_cid:
            raise ValueError(
                "metadata/Gold chain mismatch: "
                f"{z}: {meta_cid} vs {gm2c[z]}"
            )

    covered = 0
    pred_resolved = 0
    correct_resolved = 0
    gold_resolvable = 0

    gold_pairs = set()
    pred_pairs = set()

    rows = []

    for z in sorted(zero_keys):

        doc = z[0]

        gcid = gm2c[z]

        gcluster = set(
            gclusters[(doc, gcid)]
        )

        # Resolution is evaluated against
        # non-Zero coreferent partners.
        gpartners = {
            m for m in gcluster
            if m != z
            and m not in all_zero_keys
        }

        if gpartners:
            gold_resolvable += 1

        for m in gpartners:
            gold_pairs.add((z, m))

        is_covered = z in pm2c

        if is_covered:

            covered += 1

            pcid = pm2c[z]

            pcluster = set(
                pclusters[(doc, pcid)]
            )

            ppartners = {
                m for m in pcluster
                if m != z
                and m not in all_zero_keys
            }

        else:

            pcid = ""
            ppartners = set()

        is_resolved = bool(ppartners)

        if is_resolved:
            pred_resolved += 1

        correct_partners = (
            gpartners & ppartners
        )

        is_correct = bool(
            correct_partners
        )

        if is_correct:
            correct_resolved += 1

        for m in ppartners:
            pred_pairs.add((z, m))

        rows.append({
            "doc_id": z[0],
            "sentence_id": z[1],
            "token_start": z[2],
            "token_end": z[3],
            "semantics":
                selected_zero[z][
                    "semantics"
                ],
            "gold_chain_id":
                gcid,
            "pred_chain_id":
                pcid,
            "covered":
                int(is_covered),
            "pred_resolved":
                int(is_resolved),
            "correct_resolved":
                int(is_correct),
            "gold_partner_count":
                len(gpartners),
            "pred_partner_count":
                len(ppartners),
            "correct_partner_count":
                len(correct_partners)
        })

    rp = div(
        correct_resolved,
        pred_resolved
    )

    rr = div(
        correct_resolved,
        gold_resolvable
    )

    rf = f1(rp, rr)

    pair_tp = len(
        gold_pairs & pred_pairs
    )

    pp = div(
        pair_tp,
        len(pred_pairs)
    )

    pr = div(
        pair_tp,
        len(gold_pairs)
    )

    pf = f1(pp, pr)

    return {
        "gold_zero":
            len(zero_keys),
        "gold_resolvable_zero":
            gold_resolvable,
        "covered_zero":
            covered,
        "cluster_coverage":
            div(
                covered,
                len(zero_keys)
            ),
        "pred_resolved_zero":
            pred_resolved,
        "correct_resolved_zero":
            correct_resolved,
        "resolution_precision":
            rp,
        "resolution_recall":
            rr,
        "resolution_f1":
            rf,
        "gold_zero_pairs":
            len(gold_pairs),
        "pred_zero_pairs":
            len(pred_pairs),
        "correct_zero_pairs":
            pair_tp,
        "pair_precision":
            pp,
        "pair_recall":
            pr,
        "pair_f1":
            pf,
        "details":
            rows
    }


def write_csv(path, rows):

    if not rows:
        return

    out = Path(path)

    out.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with out.open(
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(rows)


def main():

    ap = argparse.ArgumentParser()

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
        choices=[
            "train",
            "dev",
            "test"
        ]
    )

    ap.add_argument(
        "--semantics",
        default="ALL",
        choices=[
            "ALL",
            "ENTITY",
            "EVENT"
        ]
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

    gold = parse_conll(
        args.gold
    )

    pred = parse_conll(
        args.response
    )

    check_alignment(
        gold,
        pred
    )

    selected, all_zero = (
        load_zero_rows(
            args.mention_map,
            args.token_map,
            args.split,
            args.semantics
        )
    )

    if (
        args.expect_zero is not None
        and len(selected)
            != args.expect_zero
    ):

        raise ValueError(
            "unexpected Zero count: "
            f"expected={args.expect_zero}, "
            f"actual={len(selected)}"
        )

    result = evaluate(
        gold,
        pred,
        selected,
        all_zero
    )

    print(
        "[PASS] CoNLL document/token alignment"
    )

    print(
        "[PASS] Zero metadata/token-map cross-check"
    )

    print("=" * 72)

    print(
        f"MODEL = {args.model}"
    )

    print(
        f"SPLIT = {args.split}"
    )

    print(
        "ZERO_SEMANTICS = "
        f"{args.semantics}"
    )

    print(
        "Gold Zero = "
        f"{result['gold_zero']}"
    )

    print(
        "Gold Resolvable Zero = "
        f"{result['gold_resolvable_zero']}"
    )

    print(
        "Covered Zero = "
        f"{result['covered_zero']}"
    )

    print(
        "Cluster Coverage = "
        f"{100*result['cluster_coverage']:.2f}%"
    )

    print(
        "Pred Resolved = "
        f"{result['pred_resolved_zero']}"
    )

    print(
        "Correct Resolved = "
        f"{result['correct_resolved_zero']}"
    )

    print(
        "Resolution P/R/F1 = "
        f"{100*result['resolution_precision']:.2f} / "
        f"{100*result['resolution_recall']:.2f} / "
        f"{100*result['resolution_f1']:.2f}"
    )

    print(
        "Gold Zero Pairs = "
        f"{result['gold_zero_pairs']}"
    )

    print(
        "Pred Zero Pairs = "
        f"{result['pred_zero_pairs']}"
    )

    print(
        "Correct Zero Pairs = "
        f"{result['correct_zero_pairs']}"
    )

    print(
        "Pair P/R/F1 = "
        f"{100*result['pair_precision']:.2f} / "
        f"{100*result['pair_recall']:.2f} / "
        f"{100*result['pair_f1']:.2f}"
    )

    print("=" * 72)

    summary = {
        "model":
            args.model,
        "split":
            args.split,
        "zero_semantics":
            args.semantics
    }

    for k, v in result.items():

        if k != "details":
            summary[k] = v

    if args.out_json:

        out = Path(
            args.out_json
        )

        out.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with out.open(
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                summary,
                f,
                ensure_ascii=False,
                indent=2
            )

    if args.out_csv:

        write_csv(
            args.out_csv,
            result["details"]
        )


if __name__ == "__main__":

    try:

        main()

    except Exception as e:

        print(
            f"[FAIL] "
            f"{type(e).__name__}: {e}",
            file=sys.stderr
        )

        sys.exit(1)
