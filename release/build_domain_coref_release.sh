#!/usr/bin/env bash
# Domain-Coref publication release builder
# Run with: bash build_domain_coref_release.sh
# IMPORTANT: do NOT source this script.

set -Eeuo pipefail
trap 'rc=$?; echo "[ERROR] build stopped at line ${LINENO}, exit=${rc}. Your E-Shell should remain open."; exit $rc' ERR

BASE="${DOMAIN_COREF_ROOT:-$(pwd)}"

# Keep the build process outside the publication directory so that
# replacing Domain-Coref cannot invalidate the current working directory.
cd "$BASE"

FROZEN="${DOMAIN_COREF_FROZEN_JSON:?Set DOMAIN_COREF_FROZEN_JSON}"
GOLD="${DOMAIN_COREF_GOLD_DIR:?Set DOMAIN_COREF_GOLD_DIR}"
SPLIT="${DOMAIN_COREF_SPLIT_DIR:?Set DOMAIN_COREF_SPLIT_DIR}"
DECON="${DOMAIN_COREF_DECONTAM_DIR:-}"

RELEASE_ROOT="${DOMAIN_COREF_RELEASE_DIR:-$BASE/release}"
FINAL="$RELEASE_ROOT/Domain-Coref"
BUILD="$RELEASE_ROOT/.Domain-Coref.build"

echo "[1] Preflight"
for p in "$FROZEN" "$GOLD" "$SPLIT"; do
  test -e "$p" || { echo "[FAIL] missing: $p"; exit 20; }
done

python - "$FROZEN" <<'PY'
import json,sys
p=sys.argv[1]
with open(p,encoding="utf-8") as f: D=json.load(f)
docs=len(D)
sents=sum(len(d.get("sentences",[])) for d in D)
mentions=sum(len(c.get("mentions",[])) for d in D for c in d.get("coreference_chains",[]))
chains=sum(len(d.get("coreference_chains",[])) for d in D)
print(dict(documents=docs,sentences=sents,mentions=mentions,chains=chains))
assert (docs,sents,mentions,chains)==(2241,15959,22510,9659)
PY

python - "$SPLIT/train_doc_ids.txt" "$SPLIT/dev_doc_ids.txt" "$SPLIT/test_doc_ids.txt" <<'PY'
import sys
for p,n in zip(sys.argv[1:],[1567,226,448]):
    with open(p,encoding="utf-8") as f:
        ids=[x.strip() for x in f if x.strip()]
    print(p.split("/")[-1],len(ids))
    assert len(ids)==n
PY

echo "[2] Clean staging"
mkdir -p "$RELEASE_ROOT"
rm -rf "$BUILD"
mkdir -p \
  "$BUILD/data/json" \
  "$BUILD/data/conll/retained_zero" \
  "$BUILD/data/conll/surface_only" \
  "$BUILD/data/splits" \
  "$BUILD/metadata" \
  "$BUILD/validation" \
  "$BUILD/docs"

echo "[3] Canonical JSON"
cp "$FROZEN" "$BUILD/data/json/Domain-Coref.json"
test "$(sha256sum "$FROZEN" | awk '{print $1}')" = \
     "$(sha256sum "$BUILD/data/json/Domain-Coref.json" | awk '{print $1}')"

echo "[4] CoNLL"
for split in train dev test all; do
  cp -L "$GOLD/conll_char_with_zero/${split}.gold.conll" \
        "$BUILD/data/conll/retained_zero/${split}.gold.conll"
  cp -L "$GOLD/conll_char_surface_only/${split}.gold.conll" \
        "$BUILD/data/conll/surface_only/${split}.gold.conll"
done

echo "[5] Official split files"
for f in \
  split_manifest.json \
  train_doc_ids.txt dev_doc_ids.txt test_doc_ids.txt \
  test.cluster_controlled_doc_ids.csv \
  test.source_balanced_doc_ids.csv
do
  test -f "$SPLIT/$f" && cp "$SPLIT/$f" "$BUILD/data/splits/$f"
done

for f in cluster_split_validation.csv split_leakage_report.csv; do
  test -f "$SPLIT/$f" && cp "$SPLIT/$f" "$BUILD/validation/$f"
done

echo "[6] Current mapping metadata"
for f in doc_map.csv chain_map.csv mention_map.csv token_map.csv type_slice_map.csv; do
  test -f "$GOLD/metadata/$f" && cp "$GOLD/metadata/$f" "$BUILD/metadata/$f"
done

echo "[7] Decontamination summaries only"
test -f "$DECON/reports/decontamination_summary.json" && \
  cp "$DECON/reports/decontamination_summary.json" \
     "$BUILD/validation/ontonotes_decontamination_summary.json"
test -f "$DECON/reports/decontamination_report.md" && \
  cp "$DECON/reports/decontamination_report.md" \
     "$BUILD/validation/ontonotes_decontamination_report.md"

echo "[8] Documentation"
cat > "$BUILD/README.md" <<'EOF'
# Domain-Coref

Domain-Coref is a document-level Chinese identity-coreference benchmark
covering five discourse domains and five anaphoric phenomena:
personal pronouns, demonstratives, definite descriptions, zero pronouns,
and event mentions.

Dataset statistics:
- 2,241 documents
- 15,959 sentences
- 558,149 tokens
- 22,510 mentions
- 9,659 identity-coreference chains

Official splits:
- train: 1,567 documents
- development: 226 documents
- test: 448 documents

The archive contains the canonical frozen JSON corpus, two official
CoNLL representations (retained-zero and surface-only), fixed split
metadata, mapping metadata, and validation/audit summaries.

Model checkpoints, predictions, run logs, caches, server paths, account
information, and experimental run directories are intentionally excluded.
EOF

cat > "$BUILD/docs/DATA_DICTIONARY.md" <<'EOF'
# Domain-Coref Data Dictionary

Document fields:
- category_primary: document-level primary anaphoric phenomenon
- domain: discourse domain
- text: complete document text
- sentences: sentence segmentation
- coreference_chains: identity-coreference chains
- entities: auxiliary entity inventory

Mention fields:
- sentence_id: 1-based sentence identifier
- text: overt mention text or annotated zero symbol
- char_start / char_end: inclusive character offsets
- role: mention role
- mention_type: personal pronoun, demonstrative, definite description,
  zero pronoun, or event mention
- referent_semantics: ENTITY or EVENT

Only identity coreference is annotated. Bridging, causal, part-whole,
temporal, and other non-identity relations are excluded.
EOF

cat > "$BUILD/docs/ACCESS_AND_LICENSE.md" <<'EOF'
# Access and License

This package is prepared for journal peer review.
The final public reuse license will be confirmed before public release
after verification of all applicable data-generation and redistribution terms.
EOF

echo "[9] Release validation"
python - "$BUILD/data/json/Domain-Coref.json" \
  "$BUILD/validation/release_validation.json" <<'PY'
import json,hashlib,sys
src,out=sys.argv[1],sys.argv[2]
raw=open(src,"rb").read()
D=json.loads(raw.decode("utf-8"))
r={
 "dataset":"Domain-Coref",
 "documents":len(D),
 "sentences":sum(len(d.get("sentences",[])) for d in D),
 "mentions":sum(len(c.get("mentions",[])) for d in D for c in d.get("coreference_chains",[])),
 "chains":sum(len(d.get("coreference_chains",[])) for d in D),
 "sha256":hashlib.sha256(raw).hexdigest(),
 "status":"frozen"
}
with open(out,"w",encoding="utf-8") as f:
    json.dump(r,f,ensure_ascii=False,indent=2)
print(r)
PY

echo "[9.5] Sanitizing publication-only metadata"

python - "$BUILD/data/splits/split_manifest.json" <<'PY2'
import json
import os
import sys

p = sys.argv[1]
root = os.path.dirname(p)

manifest = {
    "dataset": "Domain-Coref",
    "split_strategy": (
        "Group-stratified split over duplicate clusters, "
        "preserving domain, source-model, and "
        "primary-phenomenon structure."
    ),
    "counts": {
        "train_documents": 1567,
        "development_documents": 226,
        "test_documents": 448
    },
    "files": {
        "train_doc_ids": "train_doc_ids.txt",
        "dev_doc_ids": "dev_doc_ids.txt",
        "test_doc_ids": "test_doc_ids.txt"
    }
}

optional = [
    (
        "test_cluster_controlled_doc_ids",
        "test.cluster_controlled_doc_ids.csv"
    ),
    (
        "test_source_balanced_doc_ids",
        "test.source_balanced_doc_ids.csv"
    )
]

for key, name in optional:
    if os.path.exists(os.path.join(root, name)):
        manifest["files"][key] = name

with open(p, "w", encoding="utf-8") as f:
    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=2
    )

print(
    "[PASS] public split_manifest.json regenerated"
)
PY2

# Long internal audit narrative is excluded from the dataset package.
# The publication package keeps the structured summary only.
rm -f \
"$BUILD/validation/ontonotes_decontamination_report.md"

echo "[10] Sensitive/internal information scan"
SCAN="$RELEASE_ROOT/security_scan.txt"
: > "$SCAN"

grep -RInE --binary-files=without-match \
 '(/work/home/|/root/|\.ai_user_info|[A-Za-z]:\\)' \
 "$BUILD" >> "$SCAN" || true

grep -RInEi --binary-files=without-match \
 '(api[_-]?key[[:space:]]*[:=]|secret[_-]?key[[:space:]]*[:=]|access[_-]?token[[:space:]]*[:=]|refresh[_-]?token[[:space:]]*[:=]|bearer[[:space:]]+[A-Za-z0-9._-]+|sk-[A-Za-z0-9_-]{10,})' \
 "$BUILD" >> "$SCAN" || true

if [ -s "$SCAN" ]; then
  echo "[STOP] Possible internal/sensitive strings found:"
  cat "$SCAN"
  echo "[INFO] Nothing was finalized. Review $SCAN"
  exit 40
fi

if find "$BUILD" -type l | grep -q .; then
  echo "[STOP] symbolic links remain"
  find "$BUILD" -type l -print
  exit 41
fi

echo "[11] Checksums and manifest"
(
  cd "$BUILD"
  find . -type f ! -name SHA256SUMS.txt ! -name MANIFEST.tsv -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS.txt

  printf "path\tsize_bytes\tsha256\n" > MANIFEST.tsv
  find . -type f ! -name MANIFEST.tsv -print0 | sort -z \
    | while IFS= read -r -d '' f; do
        size=$(stat -c '%s' "$f")
        hash=$(sha256sum "$f" | awk '{print $1}')
        printf "%s\t%s\t%s\n" "${f#./}" "$size" "$hash"
      done >> MANIFEST.tsv
)

echo "[12] Finalize"
if [ -e "$FINAL" ]; then
    rm -rf "$FINAL"
fi
mv "$BUILD" "$FINAL"

ZIP="$RELEASE_ROOT/Domain-Coref.zip"
rm -f "$ZIP" "$ZIP.sha256"

python - "$FINAL" "$ZIP" <<'PY'
import os,sys,zipfile
src,out=sys.argv[1],sys.argv[2]
parent=os.path.dirname(src)
with zipfile.ZipFile(out,"w",compression=zipfile.ZIP_DEFLATED) as z:
    for base,dirs,files in os.walk(src):
        dirs.sort(); files.sort()
        for name in files:
            p=os.path.join(base,name)
            z.write(p,os.path.relpath(p,parent))
print(out)
PY

(
    cd "$RELEASE_ROOT"
    sha256sum "Domain-Coref.zip" > "Domain-Coref.zip.sha256"
)

echo "[13] Verify ZIP"
TMP=$(mktemp -d)
python - "$ZIP" "$TMP" <<'PY'
import sys,zipfile
z,out=sys.argv[1],sys.argv[2]
with zipfile.ZipFile(z) as f:
    bad=f.testzip()
    if bad:
        raise SystemExit("Corrupted member: "+bad)
    f.extractall(out)
print("[PASS] ZIP integrity")
PY
(
  cd "$TMP/Domain-Coref"
  sha256sum -c SHA256SUMS.txt
)
rm -rf "$TMP"

echo "[SUCCESS]"
echo "Directory: $FINAL"
echo "ZIP:       $ZIP"
echo "Checksum:  $ZIP.sha256"
find "$FINAL" -type f | wc -l
du -sh "$FINAL" "$ZIP"
