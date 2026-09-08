import csv, json, sys, copy

src, docmap, out = sys.argv[1:4]

with open(src, encoding="utf-8") as f:
    docs = json.load(f)

with open(docmap, encoding="utf-8-sig", newline="") as f:
    idx = {
        r["doc_id"].strip(): int(r["global_doc_index"])
        for r in csv.DictReader(f)
    }

affected = set()

def doc(did):
    affected.add(did)
    return docs[idx[did]]

def chain(d, ci):
    x = [c for c in d["coreference_chains"]
         if str(c.get("index")) == str(ci)]
    assert len(x) == 1, (ci, len(x))
    return x[0]

def mention(c, sid, text):
    x = [m for m in c["mentions"]
         if m.get("sentence_id") == sid
         and m.get("text") == text]
    assert len(x) == 1, (sid, text, len(x))
    return x[0]

def sentence(d, sid):
    x = d["sentences"][sid - 1]
    return x["text"] if isinstance(x, dict) else x

def set_sentence(d, sid, text):
    x = d["sentences"][sid - 1]
    if isinstance(x, dict):
        x["text"] = text
    else:
        d["sentences"][sid - 1] = text

def entity_chain(did, ci):
    c = chain(doc(did), ci)
    for m in c["mentions"]:
        m["referent_semantics"] = "ENTITY"
        if m["text"] == "Ø":
            m["mention_type"] = "零指代"
        elif m["text"] == "其":
            m["mention_type"] = "人称代词"
        else:
            m["mention_type"] = "有定描述"
    return c

# DALL_000507:
# EVENT phrase -> ENTITY NP "可验证的溯源机制"
d = doc("DALL_000507")
c = chain(d, 2)
m = mention(c, 2, "嵌入可验证的溯源机制")
assert m["char_start"] == 49 and m["char_end"] == 58
m["text"] = "可验证的溯源机制"
m["char_start"] = 51
m["mention_type"] = "有定描述"
m["referent_semantics"] = "ENTITY"

m = mention(c, 3, "该机制")
m["mention_type"] = "有定描述"
m["referent_semantics"] = "ENTITY"

m = mention(c, 4, "Ø")
m["mention_type"] = "零指代"
m["referent_semantics"] = "ENTITY"

c["mentions"] = [
    m for m in c["mentions"]
    if not (
        m.get("sentence_id") == 5
        and m.get("text") == "这项技术性规定"
    )
]

# Straight ENTITY-chain corrections
for did, ci in [
    ("DALL_000576", 1),
    ("DALL_001411", 3),
    ("DALL_001420", 3),
    ("DALL_001664", 4),
]:
    entity_chain(did, ci)

# DALL_001380:
# "建立分级访问机制" -> "分级访问机制"
d = doc("DALL_001380")
c = chain(d, 3)

m = mention(c, 4, "建立分级访问机制")
assert m["char_start"] == 5 and m["char_end"] == 12
m["text"] = "分级访问机制"
m["char_start"] = 7
m["mention_type"] = "有定描述"
m["referent_semantics"] = "ENTITY"

m = mention(c, 4, "Ø")
m["mention_type"] = "零指代"
m["referent_semantics"] = "ENTITY"

def remove_zero_char(did, ci, sid):
    d = doc(did)
    c = chain(d, ci)

    zz = [
        m for m in c["mentions"]
        if m.get("sentence_id") == sid
        and m.get("text") == "Ø"
    ]
    assert len(zz) == 1, (did, ci, sid, len(zz))

    z = zz[0]
    pos = z["char_start"]
    assert z["char_end"] == pos

    old = sentence(d, sid)
    assert old[pos] == "Ø", (did, sid, pos, old)

    c["mentions"].remove(z)

    new = old[:pos] + old[pos + 1:]
    set_sentence(d, sid, new)

    # all later mention offsets in this sentence shift left by one
    for cc in d["coreference_chains"]:
        for m in cc["mentions"]:
            if m.get("sentence_id") != sid:
                continue
            if m.get("char_start", -1) > pos:
                m["char_start"] -= 1
                m["char_end"] -= 1

    # keep top-level text synchronized
    if old in d.get("text", ""):
        d["text"] = d["text"].replace(old, new, 1)
    else:
        raise AssertionError(
            ("sentence_not_found_in_text", did, sid)
        )

    return copy.deepcopy(z)

# DALL_000575 s5:
# malformed "他强调，Ø，..." -> remove Ø
remove_zero_char("DALL_000575", 5, 5)

# DALL_001906 chain 3:
# "Ø决定" -> overt event mention "决定"
template = remove_zero_char("DALL_001906", 3, 7)

d = doc("DALL_001906")
c = chain(d, 3)
s = sentence(d, 7)

p = s.find("决定")
assert p >= 0

template["text"] = "决定"
template["char_start"] = p
template["char_end"] = p + 1
template["mention_type"] = "事件提及"
template["referent_semantics"] = "EVENT"
template["role"] = "anaphor"

c["mentions"].append(template)

# Recompute chain-level derived labels
priority = [
    ("零指代", "零指代"),
    ("人称代词", "人称代词"),
    ("指示代词", "指示代词"),
    ("事件提及", "事件指代"),
    ("有定描述", "有定描述"),
]

doc_priority = [
    "零指代",
    "事件指代",
    "人称代词",
    "指示代词",
    "有定描述",
]

for did in affected:
    d = docs[idx[did]]

    for c in d["coreference_chains"]:
        assert len(c["mentions"]) >= 2

        mts = {m.get("mention_type") for m in c["mentions"]}

        c["type"] = next(
            ct for mt, ct in priority
            if mt in mts
        )

        sids = [
            m["sentence_id"]
            for m in c["mentions"]
        ]
        c["start_sentence"] = min(sids)
        c["end_sentence"] = max(sids)

    counts = {
        t: sum(
            1 for c in d["coreference_chains"]
            if c.get("type") == t
        )
        for t in doc_priority
    }

    mx = max(counts.values())

    d["category_primary"] = next(
        t for t in doc_priority
        if counts[t] == mx
    )

with open(out, "w", encoding="utf-8") as f:
    json.dump(
        docs,
        f,
        ensure_ascii=False,
        indent=2
    )

print("affected_docs =", len(affected))
print("output =", out)
print("[PASS] semantic-zero correction candidate created")
