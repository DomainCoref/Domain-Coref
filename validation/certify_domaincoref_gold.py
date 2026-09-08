import csv,json,sys
from collections import Counter

src,docmap,testids=sys.argv[1:4]

with open(src,encoding="utf-8") as f:
    docs=json.load(f)

with open(docmap,encoding="utf-8-sig",newline="") as f:
    rows=list(csv.DictReader(f))

idx2id={
    int(r["global_doc_index"]):r["doc_id"]
    for r in rows
}

with open(testids,encoding="utf-8-sig") as f:
    tids={x.strip() for x in f if x.strip()}

cp=[
("零指代","零指代"),
("人称代词","人称代词"),
("指示代词","指示代词"),
("事件提及","事件指代"),
("有定描述","有定描述"),
]

dp=[
"零指代",
"事件指代",
"人称代词",
"指示代词",
"有定描述"
]

assert len(docs)==2241
assert len(tids)==448

A=Counter()
T=Counter()

for i,d in enumerate(docs):

    did=idx2id[i]
    istest=did in tids
    sents=d["sentences"]

    for c in d["coreference_chains"]:

        ms=c["mentions"]

        assert len(ms)>=2,(did,c["index"],"singleton")

        sems={m["referent_semantics"] for m in ms}
        assert len(sems)==1,(did,c["index"],sems)

        sids=[m["sentence_id"] for m in ms]

        assert c["start_sentence"]==min(sids)
        assert c["end_sentence"]==max(sids)

        mts={m["mention_type"] for m in ms}

        et=next(ct for mt,ct in cp if mt in mts)

        assert c["type"]==et,(
            did,c["index"],c["type"],et
        )

        A["chains"]+=1

        if istest:
            T["chains"]+=1

        for m in ms:

            sid=m["sentence_id"]
            s=sents[sid-1]

            if isinstance(s,dict):
                s=s["text"]

            a=m["char_start"]
            b=m["char_end"]
            txt=m["text"]

            assert 0<=a<=b<len(s),(
                did,sid,txt,a,b,len(s)
            )

            assert s[a:b+1]==txt,(
                did,sid,txt,s[a:b+1]
            )

            A["mentions"]+=1

            if txt=="Ø":
                assert a==b
                assert m["mention_type"]=="零指代"
                A["zero"]+=1

            if istest:

                T["mentions"]+=1

                if txt=="Ø":
                    T["zero"]+=1
                    T["zero_"+m["referent_semantics"]]+=1

    counts=Counter(
        c["type"] for c in d["coreference_chains"]
    )

    mx=max(counts.values())

    ep=next(x for x in dp if counts[x]==mx)

    assert d["category_primary"]==ep,(
        did,d["category_primary"],ep
    )

print("ALL =",dict(A))
print("TEST =",dict(T))

assert A["chains"]==9659
assert A["mentions"]==22508
assert A["zero"]==1754

assert T["chains"]==1940
assert T["mentions"]==4554
assert T["zero"]==349
assert T["zero_ENTITY"]==338
assert T["zero_EVENT"]==11

print("[PASS] FINAL DOMAIN-COREF GOLD CERTIFICATION")
