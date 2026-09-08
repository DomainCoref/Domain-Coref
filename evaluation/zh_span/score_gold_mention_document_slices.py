#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import subprocess
from pathlib import Path
from collections import defaultdict


BEGIN = re.compile(
    r'^#begin document \((.+?)\); part (\d+)'
)


DOMAIN_MAP = {
    "新闻与公共事务": "PA",
    "商业与金融": "BF",
    "科技与数字治理": "TDG",
    "教育与学术": "EDU",
    "社会民生与文体": "SLSC",
}


CATEGORY_MAP = {
    "人称代词": "Pronoun",
    "指示代词": "Demonstrative",
    "有定描述": "DefiniteNP",
    "零指代": "Zero",
    "事件指代": "Event",
}


def read_conll(path):

    docs = {}
    current_id = None
    lines = []

    with open(
        path,
        encoding="utf-8"
    ) as f:

        for raw in f:

            m = BEGIN.match(raw)

            if m:

                if current_id is not None:
                    raise RuntimeError(
                        "nested #begin"
                    )

                current_id = m.group(1)
                lines = [raw]
                continue

            if current_id is not None:

                lines.append(raw)

                if raw.startswith(
                    "#end document"
                ):

                    if current_id in docs:
                        raise RuntimeError(
                            "duplicate doc: "
                            + current_id
                        )

                    docs[current_id] = list(lines)

                    current_id = None
                    lines = []

    if current_id is not None:
        raise RuntimeError(
            "unterminated document"
        )

    return docs


def write_subset(
    docs,
    ids,
    path
):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for doc_id in ids:

            if doc_id not in docs:
                raise RuntimeError(
                    "missing doc: "
                    + doc_id
                )

            f.writelines(
                docs[doc_id]
            )


def scorer_f1(
    scorer,
    metric,
    gold,
    pred,
    log
):

    p = subprocess.run(
        [
            "perl",
            scorer,
            metric,
            str(gold),
            str(pred),
            "none",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )

    log.write_text(
        p.stdout,
        encoding="utf-8"
    )

    vals = []

    for line in p.stdout.splitlines():

        if (
            "Coreference:" in line
            and "F1:" in line
        ):

            m = re.search(
                r'F1:\s*([0-9.]+)%',
                line
            )

            if m:
                vals.append(
                    float(m.group(1))
                )

    if not vals:
        raise RuntimeError(
            "cannot parse scorer: "
            + str(log)
        )

    return vals[-1]


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--gold",
        required=True
    )

    ap.add_argument(
        "--pred",
        required=True
    )

    ap.add_argument(
        "--meta",
        required=True
    )

    ap.add_argument(
        "--scorer",
        required=True
    )

    ap.add_argument(
        "--out-dir",
        required=True
    )

    ap.add_argument(
        "--setting",
        required=True
    )

    a = ap.parse_args()

    out = Path(a.out_dir)

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    gold_docs = read_conll(
        a.gold
    )

    pred_docs = read_conll(
        a.pred
    )

    if set(gold_docs) != set(pred_docs):

        raise RuntimeError(
            "gold/pred doc set mismatch"
        )

    if len(gold_docs) != 448:

        raise RuntimeError(
            f"expected 448 docs, got "
            f"{len(gold_docs)}"
        )

    domains = defaultdict(list)
    categories = defaultdict(list)

    with open(
        a.meta,
        encoding="utf-8-sig",
        newline=""
    ) as f:

        rows = list(
            csv.DictReader(f)
        )

    meta = {
        r["doc_id"]: r
        for r in rows
    }

    for doc_id in gold_docs:

        if doc_id not in meta:

            raise RuntimeError(
                "metadata missing: "
                + doc_id
            )

        r = meta[doc_id]

        if r["split"] != "test":

            raise RuntimeError(
                "non-test metadata: "
                + doc_id
            )

        d = r["domain"]
        c = r["category_primary"]

        if d not in DOMAIN_MAP:
            raise RuntimeError(
                "unknown domain: "
                + repr(d)
            )

        if c not in CATEGORY_MAP:
            raise RuntimeError(
                "unknown category: "
                + repr(c)
            )

        domains[
            DOMAIN_MAP[d]
        ].append(doc_id)

        categories[
            CATEGORY_MAP[c]
        ].append(doc_id)

    print(
        "SETTING =",
        a.setting
    )

    print(
        "TEST_DOCS =",
        len(gold_docs)
    )

    for k in [
        "PA", "BF", "TDG",
        "EDU", "SLSC"
    ]:
        print(
            "DOMAIN",
            k,
            len(domains[k])
        )

    for k in [
        "Pronoun",
        "Demonstrative",
        "DefiniteNP",
        "Zero",
        "Event",
    ]:
        print(
            "PHENOMENON",
            k,
            len(categories[k])
        )

    if sum(
        len(x)
        for x in domains.values()
    ) != 448:
        raise RuntimeError(
            "domain total != 448"
        )

    if sum(
        len(x)
        for x in categories.values()
    ) != 448:
        raise RuntimeError(
            "category total != 448"
        )

    def evaluate_group(
        group_name,
        groups,
        order
    ):

        root = out / group_name

        root.mkdir(
            parents=True,
            exist_ok=True
        )

        result_rows = []

        for name in order:

            ids = groups[name]

            sdir = root / name

            sdir.mkdir(
                parents=True,
                exist_ok=True
            )

            gold_file = (
                sdir / "gold.conll"
            )

            pred_file = (
                sdir / "pred.conll"
            )

            write_subset(
                gold_docs,
                ids,
                gold_file
            )

            write_subset(
                pred_docs,
                ids,
                pred_file
            )

            scores = {}

            for metric in [
                "muc",
                "bcub",
                "ceafe"
            ]:

                scores[metric] = scorer_f1(
                    a.scorer,
                    metric,
                    gold_file,
                    pred_file,
                    sdir / f"{metric}.log"
                )

            conll = (
                scores["muc"]
                + scores["bcub"]
                + scores["ceafe"]
            ) / 3.0

            result_rows.append({
                "Setting":
                    a.setting,

                "Slice":
                    name,

                "Docs":
                    len(ids),

                "MUC_F1":
                    f'{scores["muc"]:.4f}',

                "B3_F1":
                    f'{scores["bcub"]:.4f}',

                "CEAFe_F1":
                    f'{scores["ceafe"]:.4f}',

                "CoNLL_F1":
                    f"{conll:.4f}",
            })

            print(
                group_name,
                name,
                len(ids),
                f"CoNLL={conll:.4f}"
            )

        result_file = (
            out
            / f"{group_name}_results.tsv"
        )

        with open(
            result_file,
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            fields = [
                "Setting",
                "Slice",
                "Docs",
                "MUC_F1",
                "B3_F1",
                "CEAFe_F1",
                "CoNLL_F1",
            ]

            w = csv.DictWriter(
                f,
                fieldnames=fields,
                delimiter="\t"
            )

            w.writeheader()
            w.writerows(
                result_rows
            )

    evaluate_group(
        "domain",
        domains,
        [
            "PA",
            "BF",
            "TDG",
            "EDU",
            "SLSC",
        ]
    )

    evaluate_group(
        "phenomenon_primary",
        categories,
        [
            "Pronoun",
            "Demonstrative",
            "DefiniteNP",
            "Zero",
            "Event",
        ]
    )

    print(
        "[PASS] document slice evaluation complete"
    )


if __name__ == "__main__":
    main()
