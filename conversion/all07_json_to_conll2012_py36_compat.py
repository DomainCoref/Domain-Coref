# -*- coding: utf-8 -*-
"""
ALL07: Chinese Coreference JSON -> CoNLL-2012 compatible format.

Outputs two scorer-ready variants:
1) conll_char_with_zero: character-level tokens with pseudo-token Ø for zero pronouns.
2) conll_char_surface_only: character-level tokens without zero-pronoun mentions.

Recommended command:
python F:\CoreNLP\Fil\ALL07\all07_json_to_conll2012.py ^
  --frozen_json "F:\CoreNLP\Datasets\Adomain\freeze\Domain_ALL_v1.0_20260623\data\Domain_ALL.frozen.json" ^
  --split_doc_ids_csv "F:\CoreNLP\Fil\ALL05\splits_decontamination_v2\split_doc_ids.csv" ^
  --out_dir "F:\CoreNLP\Fil\ALL07_conll2012" ^
  --char_end_mode inclusive ^
  --error_policy fail
"""


import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SPLITS = ["train", "dev", "test"]
MODES = ["with_zero", "surface_only"]

DOC_MAP_FIELDS = [
    "doc_id", "split", "row_index", "global_doc_index", "domain", "category_primary",
    "source_model", "source_file", "group_id", "dup_cluster_id", "n_sentences", "n_chains", "n_mentions",
]
CHAIN_MAP_FIELDS = [
    "mode", "doc_id", "split", "conll_chain_id", "original_chain_index", "chain_type",
    "n_mentions_original", "n_mentions_exported", "dropped", "drop_reason",
]
MENTION_MAP_FIELDS = [
    "mode", "doc_id", "split", "mention_id", "conll_chain_id", "original_chain_index", "chain_type",
    "sentence_id", "mention_text", "role", "mention_type", "referent_semantics", "char_start", "char_end",
    "token_start", "token_end", "is_zero", "is_exported", "drop_reason",
]
TOKEN_MAP_FIELDS = [
    "mode", "doc_id", "split", "sentence_id", "token_id", "token_text", "char_start", "char_end",
    "is_zero_token", "zero_mention_id", "char_index_before",
]
ERROR_FIELDS = [
    "severity", "mode", "split", "doc_id", "row_index", "chain_index", "mention_id", "sentence_id",
    "error_type", "message", "first_sentence",
]
TYPE_SLICE_FIELDS = [
    "mode", "doc_id", "split", "domain", "category_primary", "conll_chain_id", "chain_type",
    "mention_id", "mention_type", "referent_semantics", "is_zero", "sentence_id", "token_start", "token_end",
]


def read_text_any_encoding(path: Path) -> str:
    for enc in ["utf-8-sig", "utf-8", "gb18030", "gbk"]:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def write_text_utf8(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_json(path: Path) -> Any:
    return json.loads(read_text_any_encoding(path))


def write_json(path: Path, obj: Any) -> None:
    write_text_utf8(path, json.dumps(obj, ensure_ascii=False, indent=2))


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    text = read_text_any_encoding(path)
    lines = text.splitlines()
    if not lines:
        return []
    return [{k: (v if v is not None else "") for k, v in row.items()} for row in csv.DictReader(lines)]


def serialize_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, dict, tuple, set)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def write_csv_dicts(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: serialize_cell(row.get(k, "")) for k in fieldnames})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def is_blank(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "na"}


def safe_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        return int(float(str(x).strip()))
    except Exception:
        return None


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def normalize_space(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def sanitize_doc_id(doc_id: str) -> str:
    s = str(doc_id or "").strip() or "UNKNOWN_DOC"
    s = re.sub(r"\s+", "_", s)
    return s.replace("(", "_").replace(")", "_").replace(";", "_")


def first_sentence(doc: Dict[str, Any]) -> str:
    sents = doc.get("sentences")
    if isinstance(sents, list) and sents:
        return safe_str(sents[0])[:180]
    text = safe_str(doc.get("text", ""))
    parts = re.split(r"(?<=[。！？!?])", text, maxsplit=1)
    return (parts[0] if parts else text)[:180]


def count_doc_mentions(doc: Dict[str, Any]) -> int:
    n = 0
    for ch in doc.get("coreference_chains", []) or []:
        if isinstance(ch, dict):
            n += len(ch.get("mentions", []) or [])
    return n


def get_sentences(doc: Dict[str, Any]) -> List[str]:
    sents = doc.get("sentences")
    if isinstance(sents, list):
        return [safe_str(s) for s in sents]
    text = safe_str(doc.get("text", ""))
    parts = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", text)
    return [p for p in parts if p]


def load_docs_from_frozen_and_split(frozen_json: Path, split_doc_ids_csv: Path) -> List[Dict[str, Any]]:
    all_docs = read_json(frozen_json)
    if not isinstance(all_docs, list):
        raise ValueError("frozen_json 顶层必须是 list")
    split_rows = read_csv_dicts(split_doc_ids_csv)
    out = []
    for r in split_rows:
        idx = safe_int(r.get("row_index"))
        if idx is None:
            idx = safe_int(r.get("global_doc_index"))
        if idx is None or idx < 0 or idx >= len(all_docs):
            raise ValueError(f"split_doc_ids.csv 中 row_index/global_doc_index 无法定位 frozen doc：{r}")
        doc = all_docs[idx]
        if not isinstance(doc, dict):
            raise ValueError(f"frozen_json 第 {idx} 条不是 object")
        split = r.get("split", "")
        if split not in SPLITS:
            raise ValueError(f"非法 split={split} row={r}")
        out.append({
            "doc_id": sanitize_doc_id(r.get("doc_id") or f"DALL_{idx+1:06d}"),
            "split": split,
            "row_index": idx,
            "global_doc_index": r.get("global_doc_index", str(idx)),
            "domain": r.get("domain", doc.get("domain", "")),
            "category_primary": r.get("category_primary", doc.get("category_primary", "")),
            "source_model": r.get("source_model", ""),
            "source_file": r.get("source_file", ""),
            "group_id": r.get("group_id", ""),
            "dup_cluster_id": r.get("dup_cluster_id", ""),
            "doc": doc,
        })
    return out


def load_docs_from_split_jsons(train_json: Path, dev_json: Path, test_json: Path) -> List[Dict[str, Any]]:
    out = []
    for split, path in [("train", train_json), ("dev", dev_json), ("test", test_json)]:
        docs = read_json(path)
        if not isinstance(docs, list):
            raise ValueError(f"{path} 顶层必须是 list")
        for i, doc in enumerate(docs):
            if not isinstance(doc, dict):
                raise ValueError(f"{path} 第 {i} 条不是 object")
            out.append({
                "doc_id": sanitize_doc_id(doc.get("doc_id") or doc.get("_doc_id") or f"{split.upper()}_{i+1:06d}"),
                "split": split,
                "row_index": i,
                "global_doc_index": doc.get("global_doc_index", ""),
                "domain": doc.get("domain", ""),
                "category_primary": doc.get("category_primary", ""),
                "source_model": doc.get("source_model", ""),
                "source_file": doc.get("source_file", ""),
                "group_id": doc.get("group_id", ""),
                "dup_cluster_id": doc.get("dup_cluster_id", ""),
                "doc": doc,
            })
    return out


def is_zero_mention(m: Dict[str, Any]) -> bool:
    text = safe_str(m.get("text", "")).strip()
    mt = safe_str(m.get("mention_type", "")).strip()
    return text == "Ø" or text.startswith("Ø") or mt == "零指代"


def add_issue(rows: List[Dict[str, Any]], severity: str, mode: str, split: str, doc_id: str, row_index: Any,
              chain_index: Any, mention_id: Any, sentence_id: Any, error_type: str, message: str, doc: Dict[str, Any]) -> None:
    rows.append({
        "severity": severity, "mode": mode, "split": split, "doc_id": doc_id, "row_index": row_index,
        "chain_index": chain_index, "mention_id": mention_id, "sentence_id": sentence_id,
        "error_type": error_type, "message": message, "first_sentence": first_sentence(doc),
    })


def normalize_sentence_index(sentence_id: Any, n_sentences: int, allow_zero_based: bool,
                             warnings: List[Dict[str, Any]], base: Dict[str, Any], doc: Dict[str, Any]) -> Optional[int]:
    sid = safe_int(sentence_id)
    if sid is None:
        add_issue(warnings, "WARN", base["mode"], base["split"], base["doc_id"], base["row_index"], base["chain_index"], base["mention_id"], sentence_id, "bad_sentence_id", f"sentence_id 不是整数：{sentence_id}", doc)
        return None
    if 1 <= sid <= n_sentences:
        return sid - 1
    if allow_zero_based and 0 <= sid < n_sentences:
        add_issue(warnings, "WARN", base["mode"], base["split"], base["doc_id"], base["row_index"], base["chain_index"], base["mention_id"], sentence_id, "zero_based_sentence_id_used", f"sentence_id={sid} 被按 0-based 解释", doc)
        return sid
    add_issue(warnings, "WARN", base["mode"], base["split"], base["doc_id"], base["row_index"], base["chain_index"], base["mention_id"], sentence_id, "sentence_id_out_of_range", f"sentence_id={sid} 超出句子范围", doc)
    return None


def collect_zero_mentions_by_sentence(doc_entry: Dict[str, Any], mode: str, allow_zero_based_sentence_id: bool,
                                      warnings: List[Dict[str, Any]]) -> Dict[int, Dict[int, List[Dict[str, Any]]]]:
    if mode != "with_zero":
        return {}
    doc = doc_entry["doc"]
    doc_id = doc_entry["doc_id"]
    split = doc_entry["split"]
    row_index = doc_entry["row_index"]
    sents = get_sentences(doc)
    by_sent = defaultdict(lambda: defaultdict(list))
    for cpos, ch in enumerate(doc.get("coreference_chains", []) or []):
        if not isinstance(ch, dict):
            continue
        chain_index = ch.get("index", cpos + 1)
        for mpos, m in enumerate(ch.get("mentions", []) or []):
            if not isinstance(m, dict) or not is_zero_mention(m):
                continue
            mention_id = f"{doc_id}_c{chain_index}_m{mpos+1}"
            base = {"mode": mode, "split": split, "doc_id": doc_id, "row_index": row_index, "chain_index": chain_index, "mention_id": mention_id}
            si = normalize_sentence_index(m.get("sentence_id"), len(sents), allow_zero_based_sentence_id, warnings, base, doc)
            if si is None:
                continue
            cs = safe_int(m.get("char_start"))
            if cs is None:
                cs = 0
                add_issue(warnings, "WARN", mode, split, doc_id, row_index, chain_index, mention_id, m.get("sentence_id"), "zero_bad_char_start", "零指代 char_start 非整数，按 0 处理", doc)
            cs = max(0, min(cs, len(sents[si])))
            by_sent[si][cs].append({"mention_id": mention_id, "chain_index": chain_index, "mention_pos": mpos, "mention": m})
    for si in list(by_sent.keys()):
        for cs in list(by_sent[si].keys()):
            by_sent[si][cs].sort(key=lambda x: (safe_int(x.get("chain_index")) or 0, x.get("mention_pos", 0), x.get("mention_id", "")))
    return by_sent


def tokenize_sentence_char_level(sent: str, zero_at_pos: Dict[int, List[Dict[str, Any]]], mode: str,
                                 skip_whitespace_tokens: bool) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[str, int]]:
    tokens = []
    char_to_token: Dict[int, int] = {}
    zero_mention_to_token: Dict[str, int] = {}

    def add_token(token_text: str, char_start: Any, char_end: Any, is_zero: int, zero_mention_id: str, char_index_before: Any) -> int:
        token_id = len(tokens)
        tokens.append({
            "token_id": token_id, "token_text": token_text, "char_start": char_start, "char_end": char_end,
            "is_zero_token": is_zero, "zero_mention_id": zero_mention_id, "char_index_before": char_index_before,
        })
        return token_id

    for pos in range(len(sent) + 1):
        if mode == "with_zero" and pos in zero_at_pos:
            for z in zero_at_pos[pos]:
                tid = add_token("Ø", "", "", 1, z["mention_id"], pos)
                zero_mention_to_token[z["mention_id"]] = tid
        if pos == len(sent):
            continue
        ch = sent[pos]
        if skip_whitespace_tokens and ch.isspace():
            continue
        tid = add_token(ch, pos, pos, 0, "", "")
        char_to_token[pos] = tid
    return tokens, char_to_token, zero_mention_to_token


def map_char_span_to_token_span(char_to_token: Dict[int, int], char_start: int, char_end: int) -> Optional[Tuple[int, int]]:
    ids = [char_to_token[pos] for pos in range(char_start, char_end + 1) if pos in char_to_token]
    if not ids:
        return None
    return min(ids), max(ids)


def get_char_end(m: Dict[str, Any], char_end_mode: str) -> Optional[int]:
    ce = safe_int(m.get("char_end"))
    if ce is None:
        return None
    return ce - 1 if char_end_mode == "exclusive" else ce


def make_mention_row(mode: str, doc_id: str, split: str, mention_id: str, conll_chain_id: Any,
                     original_chain_index: Any, chain_type: str, m: Dict[str, Any], token_start: Any, token_end: Any,
                     is_zero: Any, is_exported: int, drop_reason: str) -> Dict[str, Any]:
    return {
        "mode": mode, "doc_id": doc_id, "split": split, "mention_id": mention_id,
        "conll_chain_id": conll_chain_id, "original_chain_index": original_chain_index,
        "chain_type": chain_type, "sentence_id": m.get("sentence_id", ""), "mention_text": m.get("text", ""),
        "role": m.get("role", ""), "mention_type": m.get("mention_type", ""),
        "referent_semantics": m.get("referent_semantics", ""), "char_start": m.get("char_start", ""),
        "char_end": m.get("char_end", ""), "token_start": token_start, "token_end": token_end,
        "is_zero": int(bool(is_zero)), "is_exported": int(is_exported), "drop_reason": drop_reason,
    }


def validate_nonzero_mention_text(sent: str, m: Dict[str, Any], char_start: int, char_end: int) -> Tuple[bool, str, str, bool]:
    if char_start < 0 or char_end < char_start or char_end >= len(sent):
        return False, "", safe_str(m.get("text", "")), False
    observed = sent[char_start:char_end + 1]
    expected = safe_str(m.get("text", ""))
    if observed == expected:
        return True, observed, expected, False
    if normalize_space(observed) == normalize_space(expected):
        return True, observed, expected, True
    return False, observed, expected, False


def build_doc_conll(doc_entry: Dict[str, Any], mode: str, char_end_mode: str, skip_whitespace_tokens: bool,
                    allow_zero_based_sentence_id: bool, drop_singleton_chains: bool) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    doc = doc_entry["doc"]
    doc_id = sanitize_doc_id(doc_entry["doc_id"])
    split = doc_entry["split"]
    row_index = doc_entry["row_index"]
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    chain_rows: List[Dict[str, Any]] = []
    mention_rows: List[Dict[str, Any]] = []
    token_rows: List[Dict[str, Any]] = []

    sents = get_sentences(doc)
    text = safe_str(doc.get("text", ""))
    if text and normalize_space(text) != normalize_space("".join(sents)):
        add_issue(warnings, "WARN", mode, split, doc_id, row_index, "", "", "", "text_sentences_mismatch", "text 与 ''.join(sentences) 去空白后不一致；转换以 sentences 为准", doc)

    doc_rows = [{
        "doc_id": doc_id, "split": split, "row_index": row_index,
        "global_doc_index": doc_entry.get("global_doc_index", ""),
        "domain": doc_entry.get("domain", doc.get("domain", "")),
        "category_primary": doc_entry.get("category_primary", doc.get("category_primary", "")),
        "source_model": doc_entry.get("source_model", ""), "source_file": doc_entry.get("source_file", ""),
        "group_id": doc_entry.get("group_id", ""), "dup_cluster_id": doc_entry.get("dup_cluster_id", ""),
        "n_sentences": len(sents), "n_chains": len(doc.get("coreference_chains", []) or []), "n_mentions": count_doc_mentions(doc),
    }]

    zero_by_sent = collect_zero_mentions_by_sentence(doc_entry, mode, allow_zero_based_sentence_id, warnings)
    sentence_tokens = []
    sentence_char_maps = []
    zero_maps = []
    for si, sent in enumerate(sents):
        toks, cmap, zmap = tokenize_sentence_char_level(sent, zero_by_sent.get(si, {}), mode, skip_whitespace_tokens)
        sentence_tokens.append(toks)
        sentence_char_maps.append(cmap)
        zero_maps.append(zmap)
        for t in toks:
            token_rows.append({
                "mode": mode, "doc_id": doc_id, "split": split, "sentence_id": si + 1,
                "token_id": t["token_id"], "token_text": t["token_text"], "char_start": t["char_start"],
                "char_end": t["char_end"], "is_zero_token": t["is_zero_token"],
                "zero_mention_id": t["zero_mention_id"], "char_index_before": t["char_index_before"],
            })

    start_events = defaultdict(lambda: defaultdict(list))
    end_events = defaultdict(lambda: defaultdict(list))
    single_events = defaultdict(lambda: defaultdict(list))
    next_chain_id = 1

    for cpos, ch in enumerate(doc.get("coreference_chains", []) or []):
        if not isinstance(ch, dict):
            add_issue(warnings, "WARN", mode, split, doc_id, row_index, cpos + 1, "", "", "bad_chain_object", "coreference_chains 中存在非 object，已跳过", doc)
            continue
        original_chain_index = safe_int(ch.get("index")) or (cpos + 1)
        chain_type = safe_str(ch.get("type", ""))
        raw_mentions = [m for m in (ch.get("mentions", []) or []) if isinstance(m, dict)]
        exported = []

        for mpos, m in enumerate(raw_mentions):
            mention_id = f"{doc_id}_c{original_chain_index}_m{mpos+1}"
            base = {"mode": mode, "split": split, "doc_id": doc_id, "row_index": row_index, "chain_index": original_chain_index, "mention_id": mention_id}
            si = normalize_sentence_index(m.get("sentence_id"), len(sents), allow_zero_based_sentence_id, warnings, base, doc)
            z = is_zero_mention(m)
            if si is None:
                mention_rows.append(make_mention_row(mode, doc_id, split, mention_id, "", original_chain_index, chain_type, m, "", "", z, 0, "bad_sentence_id"))
                continue
            if z and mode == "surface_only":
                mention_rows.append(make_mention_row(mode, doc_id, split, mention_id, "", original_chain_index, chain_type, m, "", "", z, 0, "surface_only_drop_zero"))
                continue
            token_start = token_end = None
            if z:
                if mention_id not in zero_maps[si]:
                    add_issue(errors, "ERROR", mode, split, doc_id, row_index, original_chain_index, mention_id, m.get("sentence_id"), "zero_token_not_found", "零指代未找到对应 Ø 伪 token", doc)
                    mention_rows.append(make_mention_row(mode, doc_id, split, mention_id, "", original_chain_index, chain_type, m, "", "", z, 0, "zero_token_not_found"))
                    continue
                token_start = token_end = zero_maps[si][mention_id]
            else:
                cs = safe_int(m.get("char_start"))
                ce = get_char_end(m, char_end_mode)
                if cs is None or ce is None:
                    add_issue(errors, "ERROR", mode, split, doc_id, row_index, original_chain_index, mention_id, m.get("sentence_id"), "bad_char_offset", "非零 mention 缺少整数 char_start/char_end", doc)
                    mention_rows.append(make_mention_row(mode, doc_id, split, mention_id, "", original_chain_index, chain_type, m, "", "", z, 0, "bad_char_offset"))
                    continue
                ok, observed, expected, space_only = validate_nonzero_mention_text(sents[si], m, cs, ce)
                if not ok:
                    msg = f"mention.text 与 offset 回切不一致；observed={observed!r}; expected={expected!r}; char_start={cs}; char_end={ce}; char_end_mode={char_end_mode}"
                    add_issue(errors, "ERROR", mode, split, doc_id, row_index, original_chain_index, mention_id, m.get("sentence_id"), "offset_text_mismatch", msg, doc)
                    mention_rows.append(make_mention_row(mode, doc_id, split, mention_id, "", original_chain_index, chain_type, m, "", "", z, 0, "offset_text_mismatch"))
                    continue
                if space_only:
                    add_issue(warnings, "WARN", mode, split, doc_id, row_index, original_chain_index, mention_id, m.get("sentence_id"), "offset_text_space_normalized", f"offset 回切只在去空白后匹配：observed={observed!r}; expected={expected!r}", doc)
                span = map_char_span_to_token_span(sentence_char_maps[si], cs, ce)
                if span is None:
                    add_issue(errors, "ERROR", mode, split, doc_id, row_index, original_chain_index, mention_id, m.get("sentence_id"), "char_span_to_token_failed", f"char span 无法映射到 token span：char_start={cs}; char_end={ce}", doc)
                    mention_rows.append(make_mention_row(mode, doc_id, split, mention_id, "", original_chain_index, chain_type, m, "", "", z, 0, "char_span_to_token_failed"))
                    continue
                token_start, token_end = span
            exported.append({"mention_id": mention_id, "mention": m, "sentence_index": si, "token_start": int(token_start), "token_end": int(token_end), "is_zero": int(z)})

        dropped = False
        drop_reason = ""
        if drop_singleton_chains and len(exported) < 2:
            dropped = True
            drop_reason = "singleton_after_filter"
            add_issue(warnings, "WARN", mode, split, doc_id, row_index, original_chain_index, "", "", "drop_singleton_chain", f"导出 mention 数={len(exported)}，少于 2，未写入 CoNLL coref column", doc)
        conll_chain_id = ""
        if not dropped:
            conll_chain_id = next_chain_id
            next_chain_id += 1
            for em in exported:
                si = em["sentence_index"]
                ts = em["token_start"]
                te = em["token_end"]
                span_len = te - ts + 1
                ev = {"cid": conll_chain_id, "token_start": ts, "token_end": te, "span_len": span_len}
                if ts == te:
                    single_events[si][ts].append(ev)
                else:
                    start_events[si][ts].append(ev)
                    end_events[si][te].append(ev)
        chain_rows.append({
            "mode": mode, "doc_id": doc_id, "split": split, "conll_chain_id": conll_chain_id,
            "original_chain_index": original_chain_index, "chain_type": chain_type,
            "n_mentions_original": len(raw_mentions), "n_mentions_exported": 0 if dropped else len(exported),
            "dropped": int(dropped), "drop_reason": drop_reason,
        })
        exported_by_mid = {em["mention_id"]: em for em in exported}
        for mpos, m in enumerate(raw_mentions):
            mid = f"{doc_id}_c{original_chain_index}_m{mpos+1}"
            if mid in exported_by_mid:
                em = exported_by_mid[mid]
                mention_rows.append(make_mention_row(mode, doc_id, split, mid, conll_chain_id if not dropped else "", original_chain_index, chain_type, m, em["token_start"], em["token_end"], em["is_zero"], 0 if dropped else 1, drop_reason))

    conll_lines = [f"#begin document ({doc_id}); part 000"]
    for si, toks in enumerate(sentence_tokens):
        for t in toks:
            tid = t["token_id"]
            markers = []
            for ev in sorted(start_events[si].get(tid, []), key=lambda x: (-x["span_len"], x["cid"])):
                markers.append(f"({ev['cid']}")
            for ev in sorted(single_events[si].get(tid, []), key=lambda x: (x["cid"], x["span_len"])):
                markers.append(f"({ev['cid']})")
            for ev in sorted(end_events[si].get(tid, []), key=lambda x: (x["span_len"], x["cid"])):
                markers.append(f"{ev['cid']})")
            coref = "|".join(markers) if markers else "-"
            cols = [doc_id, "000", str(tid), t["token_text"], "-", "*", "-", "-", "-", "-", "*", coref]
            conll_lines.append(" ".join(cols))
        conll_lines.append("")
    conll_lines.append("#end document")
    conll_lines.append("")
    return conll_lines, doc_rows, chain_rows, mention_rows, token_rows, errors, warnings


def write_empty_response_from_gold(gold_path: Path, out_path: Path) -> None:
    lines = []
    for line in read_text_any_encoding(gold_path).splitlines():
        if not line.strip() or line.startswith("#"):
            lines.append(line)
            continue
        cols = line.split()
        if len(cols) >= 12:
            cols[-1] = "-"
            lines.append(" ".join(cols))
        else:
            lines.append(line)
    write_text_utf8(out_path, "\n".join(lines) + "\n")


def make_type_slice_rows(doc_rows: List[Dict[str, Any]], chain_rows: List[Dict[str, Any]], mention_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    doc_meta = {(r["doc_id"], r["split"]): r for r in doc_rows}
    chain_meta = {(r["mode"], r["doc_id"], str(r["conll_chain_id"])): r for r in chain_rows if str(r.get("conll_chain_id", ""))}
    rows = []
    for m in mention_rows:
        if str(m.get("is_exported", "0")) != "1":
            continue
        d = doc_meta.get((m["doc_id"], m["split"]), {})
        c = chain_meta.get((m["mode"], m["doc_id"], str(m.get("conll_chain_id", ""))), {})
        rows.append({
            "mode": m.get("mode", ""), "doc_id": m.get("doc_id", ""), "split": m.get("split", ""),
            "domain": d.get("domain", ""), "category_primary": d.get("category_primary", ""),
            "conll_chain_id": m.get("conll_chain_id", ""), "chain_type": c.get("chain_type", m.get("chain_type", "")),
            "mention_id": m.get("mention_id", ""), "mention_type": m.get("mention_type", ""),
            "referent_semantics": m.get("referent_semantics", ""), "is_zero": m.get("is_zero", ""),
            "sentence_id": m.get("sentence_id", ""), "token_start": m.get("token_start", ""), "token_end": m.get("token_end", ""),
        })
    return rows


def summarize(doc_rows, chain_rows, mention_rows, token_rows, errors, warnings, output_files) -> Dict[str, Any]:
    return {
        "total_docs": len(doc_rows),
        "docs_by_split": dict(Counter(r.get("split", "") for r in doc_rows)),
        "docs_by_domain": dict(Counter(r.get("domain", "") for r in doc_rows)),
        "docs_by_category_primary": dict(Counter(r.get("category_primary", "") for r in doc_rows)),
        "modes": MODES,
        "chains_total_rows": len(chain_rows),
        "chains_exported": sum(1 for r in chain_rows if str(r.get("dropped", "0")) == "0"),
        "chains_dropped": sum(1 for r in chain_rows if str(r.get("dropped", "0")) == "1"),
        "mentions_total_rows": len(mention_rows),
        "mentions_exported": sum(1 for r in mention_rows if str(r.get("is_exported", "0")) == "1"),
        "mentions_dropped": sum(1 for r in mention_rows if str(r.get("is_exported", "0")) != "1"),
        "exported_mentions_by_mode": dict(Counter(r.get("mode", "") for r in mention_rows if str(r.get("is_exported", "0")) == "1")),
        "exported_zero_mentions_by_mode": dict(Counter(r.get("mode", "") for r in mention_rows if str(r.get("is_exported", "0")) == "1" and str(r.get("is_zero", "0")) == "1")),
        "token_rows": len(token_rows),
        "zero_tokens": sum(1 for r in token_rows if str(r.get("is_zero_token", "0")) == "1"),
        "errors": len(errors),
        "warnings": len(warnings),
        "error_types": dict(Counter(r.get("error_type", "") for r in errors)),
        "warning_types": dict(Counter(r.get("error_type", "") for r in warnings)),
        "output_files": output_files,
        "conversion_pass": len(errors) == 0,
    }


def scorer_command_text(out_dir: Path) -> str:
    return f"""# CoNLL scorer 自检命令示例
# 下载官方 scorer: https://github.com/conll/reference-coreference-scorers
# 假设 scorer.pl 路径为：F:\\CoreNLP\\reference-coreference-scorers\\scorer.pl

set SCORER=F:\\CoreNLP\\reference-coreference-scorers\\scorer.pl

perl %SCORER% all "{out_dir}\\conll_char_with_zero\\test.gold.conll" "{out_dir}\\conll_char_with_zero\\test.gold.conll" none
perl %SCORER% all "{out_dir}\\scorer_ready\\key.test.with_zero.conll" "{out_dir}\\scorer_ready\\response.identity.test.with_zero.conll" none
perl %SCORER% all "{out_dir}\\scorer_ready\\key.test.with_zero.conll" "{out_dir}\\scorer_ready\\response.empty.test.with_zero.conll" none
perl %SCORER% all "{out_dir}\\conll_char_surface_only\\test.gold.conll" "{out_dir}\\conll_char_surface_only\\test.gold.conll" none
"""


def main():
    ap = argparse.ArgumentParser(description="ALL07 Chinese coreference JSON to CoNLL-2012 converter")
    ap.add_argument("--frozen_json", default="")
    ap.add_argument("--split_doc_ids_csv", default="")
    ap.add_argument("--train_json", default="")
    ap.add_argument("--dev_json", default="")
    ap.add_argument("--test_json", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--char_end_mode", choices=["inclusive", "exclusive"], default="inclusive")
    ap.add_argument("--skip_whitespace_tokens", type=int, default=1)
    ap.add_argument("--allow_zero_based_sentence_id", type=int, default=1)
    ap.add_argument("--drop_singleton_chains", type=int, default=1)
    ap.add_argument("--error_policy", choices=["fail", "skip"], default="fail")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.frozen_json and args.split_doc_ids_csv:
        docs = load_docs_from_frozen_and_split(Path(args.frozen_json), Path(args.split_doc_ids_csv))
        input_mode = "frozen_json_plus_split_doc_ids"
    elif args.train_json and args.dev_json and args.test_json:
        docs = load_docs_from_split_jsons(Path(args.train_json), Path(args.dev_json), Path(args.test_json))
        input_mode = "split_jsons"
    else:
        raise SystemExit("必须提供 --frozen_json + --split_doc_ids_csv，或者同时提供 --train_json --dev_json --test_json")

    conll_dirs = {"with_zero": out_dir / "conll_char_with_zero", "surface_only": out_dir / "conll_char_surface_only"}
    metadata_dir = out_dir / "metadata"
    validation_dir = out_dir / "validation"
    scorer_dir = out_dir / "scorer_ready"
    for p in list(conll_dirs.values()) + [metadata_dir, validation_dir, scorer_dir]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ALL07 JSON → CoNLL-2012 转换")
    print("=" * 80)
    print("input_mode:", input_mode)
    print("docs:", len(docs))
    print("out_dir:", out_dir)
    print("char_end_mode:", args.char_end_mode)
    print("=" * 80)

    all_doc_rows, all_chain_rows, all_mention_rows, all_token_rows = [], [], [], []
    all_errors, all_warnings = [], []
    conll_by_mode_split = {m: {s: [] for s in SPLITS} for m in MODES}

    for mode in MODES:
        for entry in docs:
            lines, doc_rows, chain_rows, mention_rows, token_rows, errors, warnings = build_doc_conll(
                entry, mode, args.char_end_mode, bool(args.skip_whitespace_tokens),
                bool(args.allow_zero_based_sentence_id), bool(args.drop_singleton_chains)
            )
            conll_by_mode_split[mode][entry["split"]].extend(lines)
            if mode == "with_zero":
                all_doc_rows.extend(doc_rows)
            all_chain_rows.extend(chain_rows)
            all_mention_rows.extend(mention_rows)
            all_token_rows.extend(token_rows)
            all_errors.extend(errors)
            all_warnings.extend(warnings)

    if all_errors and args.error_policy == "fail":
        write_csv_dicts(validation_dir / "conversion_errors.csv", all_errors, ERROR_FIELDS)
        write_csv_dicts(validation_dir / "conversion_warnings.csv", all_warnings, ERROR_FIELDS)
        summary = summarize(all_doc_rows, all_chain_rows, all_mention_rows, all_token_rows, all_errors, all_warnings, {})
        write_json(validation_dir / "conversion_summary.json", summary)
        print("[FAILED] 存在转换 ERROR，未写正式 CoNLL。请查看 conversion_errors.csv")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    output_files = {}
    for mode in MODES:
        all_lines = []
        for split in SPLITS:
            path = conll_dirs[mode] / f"{split}.gold.conll"
            text = "\n".join(conll_by_mode_split[mode][split])
            if not text.endswith("\n"):
                text += "\n"
            write_text_utf8(path, text)
            output_files[f"{mode}.{split}.gold.conll"] = str(path)
            all_lines.extend(conll_by_mode_split[mode][split])
        all_path = conll_dirs[mode] / "all.gold.conll"
        text = "\n".join(all_lines)
        if not text.endswith("\n"):
            text += "\n"
        write_text_utf8(all_path, text)
        output_files[f"{mode}.all.gold.conll"] = str(all_path)

    write_csv_dicts(metadata_dir / "doc_map.csv", all_doc_rows, DOC_MAP_FIELDS)
    write_csv_dicts(metadata_dir / "chain_map.csv", all_chain_rows, CHAIN_MAP_FIELDS)
    write_csv_dicts(metadata_dir / "mention_map.csv", all_mention_rows, MENTION_MAP_FIELDS)
    write_csv_dicts(metadata_dir / "token_map.csv", all_token_rows, TOKEN_MAP_FIELDS)
    write_csv_dicts(metadata_dir / "type_slice_map.csv", make_type_slice_rows(all_doc_rows, all_chain_rows, all_mention_rows), TYPE_SLICE_FIELDS)
    write_csv_dicts(validation_dir / "conversion_errors.csv", all_errors, ERROR_FIELDS)
    write_csv_dicts(validation_dir / "conversion_warnings.csv", all_warnings, ERROR_FIELDS)

    key_test = scorer_dir / "key.test.with_zero.conll"
    identity_test = scorer_dir / "response.identity.test.with_zero.conll"
    empty_test = scorer_dir / "response.empty.test.with_zero.conll"
    shutil.copyfile(conll_dirs["with_zero"] / "test.gold.conll", key_test)
    shutil.copyfile(conll_dirs["with_zero"] / "test.gold.conll", identity_test)
    write_empty_response_from_gold(conll_dirs["with_zero"] / "test.gold.conll", empty_test)
    write_text_utf8(scorer_dir / "scorer_command.txt", scorer_command_text(out_dir))
    output_files["scorer_ready.key.test.with_zero.conll"] = str(key_test)
    output_files["scorer_ready.response.identity.test.with_zero.conll"] = str(identity_test)
    output_files["scorer_ready.response.empty.test.with_zero.conll"] = str(empty_test)

    summary = summarize(all_doc_rows, all_chain_rows, all_mention_rows, all_token_rows, all_errors, all_warnings, output_files)
    summary["config"] = {
        "input_mode": input_mode,
        "char_end_mode": args.char_end_mode,
        "skip_whitespace_tokens": bool(args.skip_whitespace_tokens),
        "allow_zero_based_sentence_id": bool(args.allow_zero_based_sentence_id),
        "drop_singleton_chains": bool(args.drop_singleton_chains),
        "error_policy": args.error_policy,
        "tokenization": "character-level",
        "zero_pronoun_mode": {"with_zero": "insert pseudo-token Ø", "surface_only": "drop zero-pronoun mentions"},
    }
    if args.frozen_json:
        summary["input_frozen_json"] = args.frozen_json
        try:
            summary["input_frozen_json_sha256"] = sha256_file(Path(args.frozen_json))
        except Exception:
            summary["input_frozen_json_sha256"] = ""
    if args.split_doc_ids_csv:
        summary["input_split_doc_ids_csv"] = args.split_doc_ids_csv
        try:
            summary["input_split_doc_ids_csv_sha256"] = sha256_file(Path(args.split_doc_ids_csv))
        except Exception:
            summary["input_split_doc_ids_csv_sha256"] = ""
    write_json(validation_dir / "conversion_summary.json", summary)

    manifest = {
        "artifact": "ALL07 CoNLL-2012 conversion",
        "version": "v1_char_with_zero_and_surface_only",
        "output_dir": str(out_dir),
        "summary_file": str(validation_dir / "conversion_summary.json"),
        "main_gold_files": {
            "with_zero_train": str(conll_dirs["with_zero"] / "train.gold.conll"),
            "with_zero_dev": str(conll_dirs["with_zero"] / "dev.gold.conll"),
            "with_zero_test": str(conll_dirs["with_zero"] / "test.gold.conll"),
            "surface_only_train": str(conll_dirs["surface_only"] / "train.gold.conll"),
            "surface_only_dev": str(conll_dirs["surface_only"] / "dev.gold.conll"),
            "surface_only_test": str(conll_dirs["surface_only"] / "test.gold.conll"),
        },
        "sidecar_metadata": {
            "doc_map": str(metadata_dir / "doc_map.csv"),
            "chain_map": str(metadata_dir / "chain_map.csv"),
            "mention_map": str(metadata_dir / "mention_map.csv"),
            "token_map": str(metadata_dir / "token_map.csv"),
            "type_slice_map": str(metadata_dir / "type_slice_map.csv"),
        },
        "scorer_ready": {
            "key_test": str(key_test),
            "identity_response": str(identity_test),
            "empty_response": str(empty_test),
            "commands": str(scorer_dir / "scorer_command.txt"),
        },
    }
    write_json(out_dir / "input_manifest.json", manifest)
    write_json(out_dir / "conversion_config.json", summary["config"])

    print("=" * 80)
    print("[DONE] CoNLL-2012 conversion completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)
    print("with_zero:", conll_dirs["with_zero"])
    print("surface_only:", conll_dirs["surface_only"])
    print("metadata:", metadata_dir)
    print("validation:", validation_dir)
    print("scorer_ready:", scorer_dir)


if __name__ == "__main__":
    main()
