#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoConfig, AutoTokenizer


def load_jsonl(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(json.loads(line))
    return docs


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--xu-root", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--output-file", required=True)

    a = p.parse_args()

    import sys
    sys.path.insert(0, a.xu_root)

    from src.local_event_coref.data import (
        KBPCorefPair,
        get_dataLoader
    )
    from src.local_event_coref.modeling import (
        BertForPairwiseEC
    )

    args = SimpleNamespace(
        max_seq_length=512,
        batch_size=4,
        num_labels=2,
        softmax_loss="ce",
        matching_style="multi",
        device="cuda"
    )

    dataset = KBPCorefPair(a.test_file)

    config = AutoConfig.from_pretrained(
        a.model,
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        a.model,
        local_files_only=True
    )

    loader = get_dataLoader(
        args,
        dataset,
        tokenizer,
        batch_size=4,
        shuffle=False
    )

    model = BertForPairwiseEC.from_pretrained(
        a.model,
        config=config,
        args=args,
        local_files_only=True
    ).cuda()

    state = torch.load(
        a.checkpoint,
        map_location="cuda"
    )

    model.load_state_dict(state)
    model.eval()

    predictions = []
    positive_probs = []
    gold_labels = []

    with torch.no_grad():

        for batch in loader:

            batch_inputs = {
                k: v.cuda()
                for k, v
                in batch["batch_inputs"].items()
            }

            e1 = torch.tensor(
                batch["batch_e1_idx"],
                device="cuda"
            )

            e2 = torch.tensor(
                batch["batch_e2_idx"],
                device="cuda"
            )

            labels = torch.tensor(
                batch["labels"],
                device="cuda"
            )

            outputs = model(
                batch_inputs=batch_inputs,
                batch_e1_idx=e1,
                batch_e2_idx=e2,
                labels=labels
            )

            logits = outputs[1]

            probs = torch.softmax(
                logits,
                dim=-1
            )

            pred = logits.argmax(
                dim=-1
            )

            predictions.extend(
                pred.cpu().tolist()
            )

            positive_probs.extend(
                probs[:, 1].cpu().tolist()
            )

            gold_labels.extend(
                labels.cpu().tolist()
            )

    if len(predictions) != len(dataset):
        raise RuntimeError(
            f"prediction count mismatch: "
            f"{len(predictions)} != {len(dataset)}"
        )

    p_, r_, f1_, support_ = (
        precision_recall_fscore_support(
            gold_labels,
            predictions,
            labels=[1],
            average=None,
            zero_division=0
        )
    )

    print("PAIR_COUNT =", len(predictions))
    print("PAIR_P =", round(float(p_[0]) * 100, 4))
    print("PAIR_R =", round(float(r_[0]) * 100, 4))
    print("PAIR_F1 =", round(float(f1_[0]) * 100, 4))
    print("POS_SUPPORT =", int(support_[0]))

    docs = load_jsonl(a.test_file)

    cursor = 0
    output = []

    for doc in docs:

        events = doc["events"]
        n = len(events)
        npairs = n * (n - 1) // 2

        doc_pred = predictions[
            cursor:cursor + npairs
        ]

        doc_prob = positive_probs[
            cursor:cursor + npairs
        ]

        if len(doc_pred) != npairs:
            raise RuntimeError(
                f"{doc['doc_id']}: "
                "pair slice mismatch"
            )

        out_events = []

        for e in events:
            out_events.append({
                "start": e["start"],
                "end":
                    e["start"]
                    + len(e["trigger"]) - 1,
                "trigger": e["trigger"]
            })

        output.append({
            "doc_id": doc["doc_id"],
            "document": doc["document"],
            "events": out_events,
            "pred_label": doc_pred,
            "pred_prob": doc_prob
        })

        cursor += npairs

    if cursor != len(predictions):
        raise RuntimeError(
            f"final cursor mismatch: "
            f"{cursor} != {len(predictions)}"
        )

    Path(a.output_file).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        a.output_file,
        "w",
        encoding="utf-8"
    ) as f:

        for row in output:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("DOCS =", len(output))
    print("[PASS] Gold-event pair predictions generated")


if __name__ == "__main__":
    main()
