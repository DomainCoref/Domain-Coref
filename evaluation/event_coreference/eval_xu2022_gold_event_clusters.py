#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


COREF_RESULTS_REGEX = re.compile(
    r".*Coreference: Recall: "
    r"\([0-9.]+ / [0-9.]+\) ([0-9.]+)%\t"
    r"Precision: "
    r"\([0-9.]+ / [0-9.]+\) ([0-9.]+)%\t"
    r"F1: ([0-9.]+)%.*",
    re.DOTALL
)

BLANC_RESULTS_REGEX = re.compile(
    r".*BLANC: Recall: "
    r"\([0-9.]+ / [0-9.]+\) ([0-9.]+)%\t"
    r"Precision: "
    r"\([0-9.]+ / [0-9.]+\) ([0-9.]+)%\t"
    r"F1: ([0-9.]+)%.*",
    re.DOTALL
)


def run_scorer(
    scorer,
    metric,
    gold,
    pred,
    output_dir
):
    cmd = [
        "perl",
        scorer,
        metric,
        str(gold),
        str(pred),
        "none"
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True
    )

    stdout = proc.stdout

    score_file = (
        output_dir
        / f"{metric}.scorer.txt"
    )

    score_file.write_text(
        stdout,
        encoding="utf-8"
    )

    regex = (
        BLANC_RESULTS_REGEX
        if metric == "blanc"
        else COREF_RESULTS_REGEX
    )

    m = re.match(
        regex,
        stdout
    )

    if m is None:
        raise RuntimeError(
            f"cannot parse scorer output "
            f"for metric={metric}; "
            f"see {score_file}"
        )

    return {
        "recall": float(m.group(1)),
        "precision": float(m.group(2)),
        "f1": float(m.group(3)),
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--xu-root",
        required=True
    )

    parser.add_argument(
        "--gold",
        required=True
    )

    parser.add_argument(
        "--pred",
        required=True
    )

    parser.add_argument(
        "--scorer",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()

    sys.path.insert(
        0,
        args.xu_root
    )

    from src.clustering.utils import (
        create_golden_conll_file,
        get_pred_coref_results,
        create_pred_conll_file
    )

    from src.clustering.cluster import (
        clustering
    )

    outdir = Path(
        args.output_dir
    )

    if outdir.exists():

        if any(
            outdir.iterdir()
        ):
            raise RuntimeError(
                "output directory is not empty: "
                f"{outdir}"
            )

    else:
        outdir.mkdir(
            parents=True
        )

    gold_conll = (
        outdir
        / "gold.event.conll"
    )

    pred_conll = (
        outdir
        / "pred.event.conll"
    )

    print(
        "Creating Gold Event CoNLL..."
    )

    create_golden_conll_file(
        args.gold,
        str(gold_conll)
    )

    pred_results = (
        get_pred_coref_results(
            args.pred
        )
    )

    print(
        "PRED_DOCS =",
        len(pred_results)
    )

    cluster_dict = {}

    print(
        "Running Xu2022 greedy clustering..."
    )

    for doc_id, result in (
        pred_results.items()
    ):

        cluster_dict[doc_id] = (
            clustering(
                result["events"],
                result["pred_labels"],
                mode="greedy"
            )
        )

    create_pred_conll_file(
        cluster_dict,
        str(gold_conll),
        str(pred_conll)
    )

    metrics = {}

    print(
        "Running official scorer..."
    )

    for metric in [
        "muc",
        "bcub",
        "ceafe",
        "blanc"
    ]:

        metrics[metric] = (
            run_scorer(
                args.scorer,
                metric,
                gold_conll,
                pred_conll,
                outdir
            )
        )

    conll_f1 = sum(
        metrics[m]["f1"]
        for m in [
            "muc",
            "bcub",
            "ceafe"
        ]
    ) / 3.0

    xu_avg_f4 = sum(
        metrics[m]["f1"]
        for m in [
            "muc",
            "bcub",
            "ceafe",
            "blanc"
        ]
    ) / 4.0

    metrics["conll_f1"] = (
        conll_f1
    )

    metrics["xu2022_avg_f4"] = (
        xu_avg_f4
    )

    result_file = (
        outdir
        / "evaluate_results.json"
    )

    result_file.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2
        )
        + "\n",
        encoding="utf-8"
    )

    print()
    print(
        "========== FINAL EVENT RESULTS =========="
    )

    for metric in [
        "muc",
        "bcub",
        "ceafe",
        "blanc"
    ]:

        m = metrics[metric]

        print(
            metric.upper(),
            "P=",
            round(
                m["precision"],
                4
            ),
            "R=",
            round(
                m["recall"],
                4
            ),
            "F1=",
            round(
                m["f1"],
                4
            )
        )

    print()
    print(
        "CoNLL_F1 =",
        round(
            conll_f1,
            4
        )
    )

    print(
        "Xu2022_AVG_F4 =",
        round(
            xu_avg_f4,
            4
        )
    )

    print()
    print(
        "[PASS] Event cluster evaluation complete"
    )


if __name__ == "__main__":
    main()
