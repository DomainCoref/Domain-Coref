import csv
import json
import sys
from collections import Counter

domain_path = sys.argv[1]
docmap_path = sys.argv[2]
testids_path = sys.argv[3]
out_path = sys.argv[4]

with open(domain_path, encoding="utf-8") as f:
    docs = json.load(f)

with open(docmap_path, encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    if "doc_id" not in reader.fieldnames:
        raise SystemExit("[FAIL] doc_id missing from doc_map")

    if "global_doc_index" not in reader.fieldnames:
        raise SystemExit("[FAIL] global_doc_index missing from doc_map")

    id_to_index = {
        row["doc_id"].strip():
        int(row["global_doc_index"])
        for row in reader
    }

with open(testids_path, encoding="utf-8-sig") as f:
    test_ids = [
        line.strip()
        for line in f
        if line.strip()
    ]

print("test_documents =", len(test_ids))

if len(test_ids) != 448:
    raise SystemExit("[FAIL] expected 448 test documents")

missing = [
    d for d in test_ids
    if d not in id_to_index
]

if missing:
    print("[FAIL] missing doc mappings:", missing[:20])
    raise SystemExit(1)

stats = Counter()
types = Counter()

for doc_id in test_ids:

    doc = docs[id_to_index[doc_id]]

    stats["documents"] += 1

    for chain in doc.get("coreference_chains", []):

        mentions = chain.get("mentions", [])

        stats["full_chains"] += 1
        stats["full_mentions"] += len(mentions)

        entity = []
        event = []
        overt_entity = []
        zero_entity = []

        semantics = set()

        for m in mentions:

            sem = m.get("referent_semantics")
            mtype = m.get("mention_type", "MISSING")

            if sem:
                semantics.add(sem)

            if sem == "ENTITY":

                entity.append(m)
                stats["entity_mentions"] += 1
                types[mtype] += 1

                if mtype == "零指代":
                    zero_entity.append(m)
                    stats["zero_entity_mentions"] += 1
                else:
                    overt_entity.append(m)
                    stats["overt_entity_mentions"] += 1

            elif sem == "EVENT":

                event.append(m)
                stats["event_mentions"] += 1

        if len(semantics) > 1:
            stats["mixed_semantics_chains"] += 1

        if len(entity) >= 2:
            stats["entity_chains"] += 1

        if len(overt_entity) >= 2:

            stats["matched_overt_entity_chains"] += 1
            stats["matched_overt_entity_mentions"] += len(
                overt_entity
            )

        if zero_entity:
            stats["chains_with_zero"] += 1

        if len(event) >= 2:
            stats["event_chains"] += 1

result = {
    **dict(stats),
    "entity_mention_types": dict(types)
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(
        result,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    json.dumps(
        result,
        ensure_ascii=False,
        indent=2
    )
)

if stats["mixed_semantics_chains"] != 0:
    raise SystemExit(
        "[FAIL] mixed ENTITY/EVENT chains detected"
    )

print()
print("[PASS] official Domain-Coref test views audited")
