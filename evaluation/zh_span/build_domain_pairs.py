#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path


def flatten_tokens(doc):
    return [
        token
        for sent in doc["sentences"]
        for token in sent
    ]



def normalize_e2e_tokens(
    raw_tokens,
    raw_clusters,
    doc_key
):
    """
    Remove BERT structural [CLS]/[SEP] tokens
    inherited from e2e/minimize and remap all
    inclusive mention offsets.
    """

    special = {
        "[CLS]",
        "[SEP]"
    }

    keep_old_indexes = [
        i
        for i, token in enumerate(raw_tokens)
        if token not in special
    ]

    old_to_new = {
        old: new
        for new, old
        in enumerate(keep_old_indexes)
    }

    tokens = [
        raw_tokens[i]
        for i in keep_old_indexes
    ]

    clusters = []

    for cid, cluster in enumerate(
        raw_clusters
    ):

        new_cluster = []

        for mention in cluster:

            start = int(mention[0])
            end = int(mention[1])

            if not (
                0 <= start <= end
                < len(raw_tokens)
            ):
                raise ValueError(
                    f"{doc_key}: bad raw span "
                    f"{(start, end)}"
                )

            old_positions = list(
                range(
                    start,
                    end + 1
                )
            )

            if any(
                raw_tokens[i] in special
                for i in old_positions
            ):
                raise ValueError(
                    f"{doc_key}: gold mention "
                    f"touches structural token: "
                    f"{(start, end)} "
                    f"{raw_tokens[start:end+1]}"
                )

            mapped = [
                old_to_new[i]
                for i in old_positions
            ]

            expected = list(
                range(
                    mapped[0],
                    mapped[-1] + 1
                )
            )

            if mapped != expected:
                raise ValueError(
                    f"{doc_key}: remapped mention "
                    f"is not contiguous: "
                    f"{(start, end)} -> {mapped}"
                )

            new_cluster.append(
                [
                    mapped[0],
                    mapped[-1]
                ]
            )

        clusters.append(
            new_cluster
        )

    removed = (
        len(raw_tokens)
        - len(tokens)
    )

    return (
        tokens,
        clusters,
        removed
    )


def build_cluster_map(clusters, n_tokens, doc_key):
    """
    span: inclusive [start, end]
    return:
        cluster_of[(start, end)] = cluster_id
    """
    cluster_of = {}

    for cid, cluster in enumerate(clusters):

        for mention in cluster:

            if len(mention) != 2:
                raise ValueError(
                    f"{doc_key}: illegal mention {mention}"
                )

            start = int(mention[0])
            end = int(mention[1])

            if not (
                0 <= start <= end < n_tokens
            ):
                raise ValueError(
                    f"{doc_key}: span out of range "
                    f"{(start, end)} / {n_tokens}"
                )

            span = (start, end)

            if (
                span in cluster_of
                and cluster_of[span] != cid
            ):
                raise ValueError(
                    f"{doc_key}: same span appears "
                    f"in multiple clusters: {span}"
                )

            cluster_of[span] = cid

    return cluster_of


def centered_window(tokens, span1, span2, max_tokens):
    """
    Construct a pair-centered window.

    span1/span2 use document-global inclusive indexes.
    max_tokens excludes [CLS] and [SEP].
    """

    left = min(
        span1[0],
        span2[0]
    )

    right = max(
        span1[1],
        span2[1]
    )

    pair_width = right - left + 1

    if pair_width > max_tokens:
        return None

    spare = max_tokens - pair_width

    start = max(
        0,
        left - spare // 2
    )

    end = min(
        len(tokens),
        start + max_tokens
    )

    # If right boundary reaches document end,
    # move the window left as much as possible.
    if end - start < max_tokens:
        start = max(
            0,
            end - max_tokens
        )

    end = min(
        len(tokens),
        start + max_tokens
    )

    if not (
        start <= left
        and right < end
    ):
        raise RuntimeError(
            "pair-centered window construction failed"
        )

    local1 = (
        span1[0] - start,
        span1[1] - start
    )

    local2 = (
        span2[0] - start,
        span2[1] - start
    )

    return {
        "start": start,
        "end": end,
        "tokens": tokens[start:end],
        "local1": local1,
        "local2": local2,
        "pair_width": pair_width,
    }


def make_record(
    doc_key,
    tokens,
    span1,
    span2,
    cid1,
    cid2,
    window,
    representation,
    split
):
    wtokens = window["tokens"]

    local1 = window["local1"]
    local2 = window["local2"]

    span1_tokens = tokens[
        span1[0]:span1[1] + 1
    ]

    span2_tokens = tokens[
        span2[0]:span2[1] + 1
    ]

    label = (
        "true"
        if cid1 == cid2
        else "false"
    )

    return {
        "idx": -1,

        "text": " ".join(wtokens),

        "target": {
            "span1_text":
                " ".join(span1_tokens),

            "span1_index":
                local1[0],

            "span2_text":
                " ".join(span2_tokens),

            "span2_index":
                local2[0],
        },

        "label": label,

        "_meta": {
            "doc_key": doc_key,
            "representation": representation,
            "split": split,

            "global_span1": list(span1),
            "global_span2": list(span2),

            "local_span1": list(local1),
            "local_span2": list(local2),

            "cluster1": cid1,
            "cluster2": cid2,

            "window_start": window["start"],
            "window_end_exclusive": window["end"],

            "pair_width": window["pair_width"],
        }
    }


def write_jsonl(path, records):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for idx, record in enumerate(records):

            record = dict(record)
            record["idx"] = idx

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


def process_split(
    input_path,
    output_dir,
    representation,
    split,
    max_seq_len,
    seed,
    train_neg_ratio
):

    max_tokens = max_seq_len - 2

    positives = []
    negatives = []
    rejected = []

    stats = Counter()

    doc_keys_with_output = set()

    with open(
        input_path,
        encoding="utf-8"
    ) as f:

        for line_no, line in enumerate(f, 1):

            if not line.strip():
                continue

            doc = json.loads(line)

            doc_key = doc["doc_key"]

            raw_tokens = flatten_tokens(doc)

            raw_clusters = doc.get(
                "clusters",
                []
            )

            (
                tokens,
                clusters,
                removed_special
            ) = normalize_e2e_tokens(
                raw_tokens,
                raw_clusters,
                doc_key
            )

            stats["source_docs"] += 1

            stats[
                "raw_source_tokens"
            ] += len(raw_tokens)

            stats[
                "removed_special_tokens"
            ] += removed_special

            stats[
                "source_tokens"
            ] += len(tokens)

            # preprocess.py uses text.split(' '),
            # so a source token must not itself contain a blank.
            bad_tokens = [
                t
                for t in tokens
                if " " in t
            ]

            if bad_tokens:
                raise ValueError(
                    f"{doc_key}: token contains spaces: "
                    f"{bad_tokens[:10]}"
                )

            stats["source_chains"] += len(
                clusters
            )

            cluster_of = build_cluster_map(
                clusters,
                len(tokens),
                doc_key
            )

            mentions = sorted(
                cluster_of.keys(),
                key=lambda x: (
                    x[0],
                    x[1]
                )
            )

            stats["source_mentions"] += len(
                mentions
            )

            # combinations => i < j
            # so no self-pair is possible.
            for span1, span2 in combinations(
                mentions,
                2
            ):

                stats["candidate_pairs"] += 1

                cid1 = cluster_of[span1]
                cid2 = cluster_of[span2]

                is_positive = (
                    cid1 == cid2
                )

                if is_positive:
                    stats[
                        "candidate_positive"
                    ] += 1
                else:
                    stats[
                        "candidate_negative"
                    ] += 1

                window = centered_window(
                    tokens,
                    span1,
                    span2,
                    max_tokens
                )

                if window is None:

                    key = (
                        "dropped_long_positive"
                        if is_positive
                        else
                        "dropped_long_negative"
                    )

                    stats[key] += 1

                    rejected.append({
                        "doc_key": doc_key,
                        "span1": list(span1),
                        "span2": list(span2),
                        "label":
                            "true"
                            if is_positive
                            else "false"
                    })

                    continue

                record = make_record(
                    doc_key=doc_key,
                    tokens=tokens,
                    span1=span1,
                    span2=span2,
                    cid1=cid1,
                    cid2=cid2,
                    window=window,
                    representation=representation,
                    split=split
                )

                doc_keys_with_output.add(
                    doc_key
                )

                if is_positive:
                    positives.append(
                        record
                    )
                else:
                    negatives.append(
                        record
                    )

    stats["valid_positive"] = len(
        positives
    )

    stats["valid_negative"] = len(
        negatives
    )

    rng = random.Random(
        seed
    )

    if split == "train":

        target_neg = int(
            round(
                len(positives)
                * train_neg_ratio
            )
        )

        target_neg = min(
            target_neg,
            len(negatives)
        )

        sampled_negative = rng.sample(
            negatives,
            target_neg
        )

        records = (
            positives
            + sampled_negative
        )

        rng.shuffle(records)

    else:

        # DEV/TEST:
        # never use gold labels to remove
        # valid candidate pairs.
        records = (
            positives
            + negatives
        )

    stats["written_total"] = len(
        records
    )

    stats["written_positive"] = sum(
        x["label"] == "true"
        for x in records
    )

    stats["written_negative"] = sum(
        x["label"] == "false"
        for x in records
    )

    stats[
        "docs_with_output"
    ] = len(doc_keys_with_output)

    output_path = (
        output_dir
        / f"{split}.json"
    )

    write_jsonl(
        output_path,
        records
    )

    reject_path = (
        output_dir
        / f"{split}.rejected_long.jsonl"
    )

    write_jsonl(
        reject_path,
        rejected
    )

    summary = {
        "representation": representation,
        "split": split,

        "input":
            str(input_path),

        "output":
            str(output_path),

        "max_seq_len":
            max_seq_len,

        "max_document_tokens_per_pair":
            max_tokens,

        "seed":
            seed,

        "train_negative_ratio":
            train_neg_ratio,

        **dict(stats)
    }

    summary_path = (
        output_dir
        / f"{split}.summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    return summary


def main():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input-root",
        required=True
    )

    p.add_argument(
        "--output-root",
        required=True
    )

    p.add_argument(
        "--representations",
        nargs="+",
        default=[
            "with_zero",
            "surface_only"
        ]
    )

    p.add_argument(
        "--source-size",
        type=int,
        default=256
    )

    p.add_argument(
        "--max-seq-len",
        type=int,
        default=300
    )

    p.add_argument(
        "--seed",
        type=int,
        default=123
    )

    p.add_argument(
        "--train-neg-ratio",
        type=float,
        default=1.0
    )

    args = p.parse_args()

    input_root = Path(
        args.input_root
    )

    output_root = Path(
        args.output_root
    )

    all_summaries = []

    for rep in args.representations:

        out = (
            output_root
            / rep
        )

        out.mkdir(
            parents=True,
            exist_ok=True
        )

        for split in [
            "train",
            "dev",
            "test"
        ]:

            inp = (
                input_root
                / rep
                / (
                    f"{split}.chinese."
                    f"{args.source_size}.jsonlines"
                )
            )

            if not inp.is_file():
                raise FileNotFoundError(
                    inp
                )

            summary = process_split(
                input_path=inp,
                output_dir=out,
                representation=rep,
                split=split,
                max_seq_len=args.max_seq_len,
                seed=args.seed,
                train_neg_ratio=args.train_neg_ratio
            )

            all_summaries.append(
                summary
            )

            print()
            print(
                "=====",
                rep,
                split,
                "====="
            )

            for k, v in summary.items():
                if k not in {
                    "input",
                    "output"
                }:
                    print(
                        f"{k} = {v}"
                    )

    with open(
        output_root
        / "adapter_summary.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_summaries,
            f,
            ensure_ascii=False,
            indent=2
        )


if __name__ == "__main__":
    main()
