import csv, json, sys
from collections import Counter

domain, docmap, testids, mmap, out = sys.argv[1:6]

with open(testids, encoding="utf-8-sig") as f:
    tids = {x.strip() for x in f if x.strip()}

assert len(tids) == 448, len(tids)

with open(docmap, encoding="utf-8-sig", newline="") as f:
    idx = {
        r["doc_id"].strip(): int(r["global_doc_index"])
        for r in csv.DictReader(f)
    }

with open(domain, encoding="utf-8") as f:
    docs = json.load(f)

j = Counter()

for did in tids:
    for c in docs[idx[did]].get("coreference_chains", []):
        for m in c.get("mentions", []):
            mt = str(m.get("mention_type"))
            sem = str(m.get("referent_semantics"))
            txt = str(m.get("text"))

            if mt == "零指代":
                j["type_zero"] += 1
                j["type_zero_sem:" + sem] += 1

            if txt == "Ø":
                j["text_O"] += 1
                j["text_O_type:" + mt] += 1
                j["text_O_sem:" + sem] += 1

m = Counter()

with open(mmap, encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("split", "").strip() != "test":
            continue

        mode = r.get("mode", "").strip()
        mt = r.get("mention_type", "").strip()
        sem = r.get("referent_semantics", "").strip()
        txt = r.get("mention_text", "")
        iz = r.get("is_zero", "").strip() == "1"
        ex = r.get("is_exported", "").strip() == "1"

        m[mode + ":rows"] += 1

        if iz:
            m[mode + ":is_zero"] += 1
            m[mode + ":zero_sem:" + sem] += 1
            m[mode + ":zero_type:" + mt] += 1

        if iz and ex:
            m[mode + ":exported_zero"] += 1

        if mt == "零指代":
            m[mode + ":type_zero"] += 1

        if txt == "Ø":
            m[mode + ":text_O"] += 1

result = {
    "test_documents": len(tids),
    "frozen_json": dict(sorted(j.items())),
    "mention_map": dict(sorted(m.items()))
}

with open(out, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))
print("[PASS] zero count reconciliation")
