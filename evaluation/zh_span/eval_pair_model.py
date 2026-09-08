#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bert-dir", required=True)
    p.add_argument("--output", required=True)

    p.add_argument(
        "--split",
        choices=["dev", "test"],
        required=True
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    p.add_argument(
        "--max-seq-len",
        type=int,
        default=300
    )

    p.add_argument(
        "--device",
        default="cuda:0"
    )

    return p.parse_args()


args_cli = parse_args()

# The upstream repository parses sys.argv at import time.
# Prevent our evaluation-only arguments from leaking into it.
sys.argv = [sys.argv[0]]

import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from preprocess import (
    CRProcessor,
    convert_examples_to_features
)

import dataset
import CRModel


def main():

    data_path = Path(args_cli.data)
    ckpt_path = Path(args_cli.checkpoint)
    out_path = Path(args_cli.output)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        args_cli.device
    )

    processor = CRProcessor()

    raw_lines = list(
        processor.read_json(
            str(data_path)
        )
    )

    records = [
        json.loads(x)
        for x in raw_lines
    ]

    examples = processor.get_examples(
        raw_lines,
        args_cli.split
    )

    features = convert_examples_to_features(
        examples,
        max_seq_len=args_cli.max_seq_len,
        bert_dir=args_cli.bert_dir
    )

    if not (
        len(records)
        == len(examples)
        == len(features)
    ):
        raise RuntimeError(
            "records/examples/features length mismatch"
        )

    ds = dataset.CRDataset(
        features
    )

    loader = DataLoader(
        ds,
        batch_size=args_cli.batch_size,
        shuffle=False,
        num_workers=2
    )

    model_args = SimpleNamespace(
        bert_dir=args_cli.bert_dir,
        dropout_prob=0.1,
        model_type="2d",
        en_cn="cn"
    )

    model = (
        CRModel
        .CorefernceResolutionModel(
            model_args
        )
        .to(device)
    )

    state = torch.load(
        ckpt_path,
        map_location=device
    )

    model.load_state_dict(
        state,
        strict=True
    )

    model.eval()

    cursor = 0

    tp = fp = fn = tn = 0

    with open(
        out_path,
        "w",
        encoding="utf-8"
    ) as fout:

        with torch.no_grad():

            for batch in loader:

                for key in batch:
                    batch[key] = (
                        batch[key]
                        .to(device)
                    )

                logits = model(
                    batch["token_ids"],
                    batch["attention_masks"],
                    batch["token_type_ids"],
                    batch["span1_ids"],
                    batch["span2_ids"]
                )

                probs = torch.softmax(
                    logits,
                    dim=-1
                )

                p0 = (
                    probs[:, 0]
                    .detach()
                    .cpu()
                    .tolist()
                )

                p1 = (
                    probs[:, 1]
                    .detach()
                    .cpu()
                    .tolist()
                )

                gold = (
                    batch["label"]
                    .detach()
                    .cpu()
                    .tolist()
                )

                n = len(gold)

                for j in range(n):

                    src = records[
                        cursor + j
                    ]

                    meta = src["_meta"]

                    pred = (
                        1
                        if p1[j] >= 0.5
                        else 0
                    )

                    y = int(gold[j])

                    if pred == 1 and y == 1:
                        tp += 1
                    elif pred == 1 and y == 0:
                        fp += 1
                    elif pred == 0 and y == 1:
                        fn += 1
                    else:
                        tn += 1

                    out = {
                        "idx":
                            cursor + j,

                        "doc_key":
                            meta["doc_key"],

                        "global_span1":
                            meta["global_span1"],

                        "global_span2":
                            meta["global_span2"],

                        "cluster1":
                            meta.get("cluster1"),

                        "cluster2":
                            meta.get("cluster2"),

                        "gold_label":
                            y,

                        "p_noncoref":
                            float(p0[j]),

                        "p_coref":
                            float(p1[j]),

                        "pred_0p5":
                            pred
                    }

                    fout.write(
                        json.dumps(
                            out,
                            ensure_ascii=False
                        )
                        + "\n"
                    )

                cursor += n

    if cursor != len(records):
        raise RuntimeError(
            f"output rows mismatch: "
            f"{cursor} != {len(records)}"
        )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    accuracy = (
        (tp + tn)
        / (tp + fp + fn + tn)
    )

    summary = {
        "split":
            args_cli.split,

        "rows":
            cursor,

        "threshold":
            0.5,

        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,

        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1
    }

    summary_path = Path(
        str(out_path)
        + ".summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False
        )
    )

    print(
        "[PASS] pair inference complete"
    )


if __name__ == "__main__":
    main()
