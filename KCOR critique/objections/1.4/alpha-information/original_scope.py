"""Audit the current checked-in original alpha study scope against the CMR workbook."""
from pathlib import Path
from collections import defaultdict
from datetime import date
import zipfile,gzip,sys
from lxml import etree
import numpy as np
import pandas as pd
import yaml
import diagnostic as d
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
NS="{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
CFG=yaml.safe_load((REPO/"test/alpha/params_alpha.yaml").read_text())
A=CFG["analysis"]
wave_start="2021-41";wave_end="2022-12"
years=set(A["yob_decades"]);doses=set(A["doses"])
def rows():
    path=REPO/"data/Czech/KCOR_CMR.xlsx"
    records=[]
    with zipfile.ZipFile(path) as z:
        strings=["".join(e.itertext()) for e in etree.fromstring(z.read("xl/sharedStrings.xml")).findall(NS+"si")]
        w=etree.fromstring(z.read("xl/workbook.xml"));rs=etree.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets={r.get("Id"):r.get("Target") for r in rs}
        for sheet in A["enrollment_dates"]:
            start=sheet.replace("_","-")
            # No overlap is excluded in the current alpha code. Skip loading
            # sheets whose enrollments start after the supplied wave ends.
            if start>wave_end:
                records.append(dict(enrollment=sheet,selected_people=0,wave_people=0,eligible_cohorts_by_counts=0,disposition="no-wave-overlap"))
                continue
            sh=next(s for s in w.findall(".//"+NS+"sheet") if s.get("name")==sheet)
            target=targets[sh.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")];target=target.lstrip("/") if target.startswith("/") else "xl/"+target
            groups=defaultdict(lambda:[0,0])
            with z.open(target) as stream:
                for _,row in etree.iterparse(stream,events=("end",),tag=NS+"row"):
                    if row.get("r")!="1":
                        # Read age and dose first, before expanding all cell values.
                        c={x.get("r")[0]:x.findtext(NS+"v") for x in row}
                        yob=int(c["B"])//10*10;dose=int(c["E"])
                        if yob in years and dose in doses:
                            week=strings[int(c["A"])]
                            if week>=start:
                                v=groups[yob,dose,week];v[0]+=int(c["J"]);v[1]+=int(c["G"])
                    row.clear()
                    while row.getprevious() is not None:del row.getparent()[0]
            history=defaultdict(dict)
            for (yob,dose,week),v in groups.items():history[yob,dose][week]=v
            selected=wave_people=eligible=0
            for (yob,dose),data in sorted(history.items()):
                weeks=sorted(data);values=np.array([data[w] for w in weeks])
                dates=pd.DatetimeIndex([date.fromisocalendar(*map(int,w.split("-")),1) for w in weeks])
                selection=values[0,0];selected+=selection
                mask=np.array([wave_start<=w<=wave_end and data[w][0]>0 for w in weeks])
                if not mask.any():continue
                count=values[mask,1].sum();valid=int(mask.sum()>=A["min_wave_weeks"] and count>=A["min_wave_deaths"])
                eligible+=valid;wave_people+=values[np.flatnonzero(mask)[0],0]
                if valid:
                    yield "actual Czech",sheet,str(yob),dose,dates,values[:,0],values[:,1]
            records.append(dict(enrollment=sheet,selected_people=int(selected),wave_people=int(wave_people),eligible_cohorts_by_counts=eligible,disposition="wave-overlap"))
            print("scope",records[-1],flush=True)
    pd.DataFrame(records).to_csv(HERE/"original-scope-populations.csv",index=False)

def main():
    d.old.read_czech=rows
    result=pd.DataFrame(list(d.actual_rows()))
    result.to_csv(HERE/"original-scope-cohort-information.csv",index=False)
    summary=[]
    # Retain separate enrollment groups: cohorts are repeated people.
    for (enroll,scenario),g in result.groupby(["enrollment","scenario"]):
        info=g["info"].sum();se=1/np.sqrt(info) if info>0 else np.inf
        summary.append(dict(enrollment=enroll,scenario=scenario,cohorts=len(g),people_at_modeled_origin=int(g.N.sum()),SE=se,half95=1.96*se))
    df=pd.DataFrame(summary);df.to_csv(HERE/"original-scope-information.csv",index=False)
    print(df.round(4).to_string(index=False))
if __name__=="__main__":main()
