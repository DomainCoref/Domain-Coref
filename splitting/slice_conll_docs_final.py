import sys,re

src,idsf,out=sys.argv[1:4]

ids={
    x.strip()
    for x in open(idsf,encoding="utf-8")
    if x.strip()
}

BEGIN=re.compile(r"#begin document \((.*?)\)")

buf=[]
keep=False

with open(out,"w",encoding="utf-8") as fout:

    for line in open(src,encoding="utf-8"):

        if line.startswith("#begin document"):

            m=BEGIN.search(line)
            assert m,line

            did=m.group(1)

            buf=[line]
            keep=did in ids

        elif line.startswith("#end document"):

            buf.append(line)

            if keep:
                fout.write("".join(buf))

            buf=[]

        elif buf:
            buf.append(line)

print("docs =",sum(
    1 for x in open(out,encoding="utf-8")
    if x.startswith("#begin document")
))
print("[PASS]")
