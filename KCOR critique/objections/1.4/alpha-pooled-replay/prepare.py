"""Prepare only aggregate selected cohort-week counts from the historical LFS workbook."""
from pathlib import Path
import sys,subprocess,zipfile,hashlib
from collections import defaultdict
from lxml import etree
import pandas as pd,yaml
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
COMMIT="e50c23ca"
OID="aafa9d34f64f56010246e7536099d758821deaf3c4175184bedbe022387d3ad3"
def git(path):return subprocess.check_output(["git","show",f"{COMMIT}:{path}"],cwd=REPO)
CFG=yaml.safe_load(git("test/alpha/params_alpha.yaml"))
NS="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
def main():
    p=REPO/".git/lfs/objects"/OID[:2]/OID[2:4]/OID
    h=hashlib.sha256()
    with p.open("rb") as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b""):h.update(b)
    assert h.hexdigest()==OID
    A=CFG["analysis"];ages=set(A["yob_decades"]);doses=set(A["doses"])
    out=[]
    with zipfile.ZipFile(p) as z:
        strings=["".join(e.itertext()) for e in etree.fromstring(z.read("xl/sharedStrings.xml")).findall(NS+"si")]
        w=etree.fromstring(z.read("xl/workbook.xml"));rs=etree.fromstring(z.read("xl/_rels/workbook.xml.rels"));targets={r.get("Id"):r.get("Target") for r in rs}
        for sheet in A["enrollment_dates"]:
            start=sheet.replace("_","-")
            if start>"2022-12":continue
            sh=next(s for s in w.findall(".//"+NS+"sheet") if s.get("name")==sheet)
            target=targets[sh.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")];target=target.lstrip("/") if target.startswith("/") else "xl/"+target
            counts=defaultdict(lambda:[0,0])
            with z.open(target) as stream:
                for _,row in etree.iterparse(stream,events=("end",),tag=NS+"row"):
                    if row.get("r")!="1":
                        c={x.get("r")[0]:x.findtext(NS+"v") for x in row}
                        yob=int(c["B"])//10*10;dose=int(c["E"])
                        if yob in ages and dose in doses:
                            week=strings[int(c["A"])]
                            if week>=start:
                                v=counts[yob,dose,week];v[0]+=int(c["J"]);v[1]+=int(c["G"])
                    row.clear()
                    while row.getprevious() is not None:del row.getparent()[0]
            for (age,dose,week),(alive,dead) in counts.items():out.append(dict(enrollment=sheet,YearOfBirth=age,Dose=dose,ISOweekDied=week,Alive=alive,Dead=dead))
            print(sheet,len(counts),"selected histories",flush=True)
    df=pd.DataFrame(out).sort_values(["enrollment","YearOfBirth","Dose","ISOweekDied"])
    df.to_csv(HERE/"aggregate-histories.csv",index=False)
    print("Aggregated cohort-weeks",len(df),flush=True)
if __name__=="__main__":main()
