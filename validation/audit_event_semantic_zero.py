import csv
import json
import sys

domain, docmap, testids, out = sys.argv[1:5]

with open(testids, encoding="utf-8-sig") as f:
    tids = {x.strip() for x in f if x.strip()}

assert len(tids) == 448

with open(docmap, encoding="utf-8-sig", newline="") as f:
    idx = {
        r["doc_id"].strip(): int(r["global_doc_index"])
        for r in csv.DictReader(f)
    }

with open(domain, encoding="utf-8") as f:
    docs = json.load(f)

rows = []

for did in sorted(tids):

    doc = docs[idx[did]]
    sents = doc.get("sentences", [])

    for chain in doc.get("coreference_chains", []):

        mentions = chain.get("mentions", [])

        bad = [
            m for m in mentions
            if m.get("mention_type") == "零指代"
            and m.get("referent_semantics") == "EVENT"
        ]

        for z in bad:

            sid = z.get("sentence_id")
            sent = ""

            if (
                isinstance(sents, list)
                and isinstance(sid, int)
                and 1 <= sid <= len(sents)
            ):
                item = sents[sid - 1]

                if isinstance(item, dict):
                    sent = str(item.get("text", item))
                else:
                    sent = str(item)

            chain_view = []

            for m in mentions:
                chain_view.append(
                    "%s|%s|%s|s%s" % (
                        m.get("text"),
                        m.get("mention_type"),
                        m.get("referent_semantics"),
                        m.get("sentence_id")
                    )
                )

            rows.append({
                "doc_id": did,
                "domain": doc.get("domain"),
                "category_primary":
                    doc.get("category_primary"),
                "chain_index":
                    chain.get("index"),
                "chain_type":
                    chain.get("type"),
                "sentence_id": sid,
                "sentence": sent,
                "zero_role":
                    z.get("role"),
                "chain_mentions":
                    " || ".join(chain_view)
            })

assert len(rows) == 19, len(rows)

fields = [
    "doc_id",
    "domain",
    "category_primary",
    "chain_index",
    "chain_type",
    "sentence_id",
    "sentence",
    "zero_role",
    "chain_mentions"
]

with open(out, "w", encoding="utf-8", newline="") as f:

    w = csv.DictWriter(
        f,
        fieldnames=fields,
        delimiter="\t"
    )

    w.writeheader()
    w.writerows(rows)

print("EVENT-semantic Zero =", len(rows))
print(out)
print("[PASS]")
