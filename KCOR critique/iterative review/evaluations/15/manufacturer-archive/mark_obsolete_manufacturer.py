"""Replace only obsolete manufacturer worksheet parts, preserving unrelated OOXML."""
from pathlib import Path
from zipfile import ZipFile
import hashlib, json, shutil, sys, os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

ROOT=Path(sys.argv[1]).resolve()
PARENT=Path(sys.argv[2]).resolve()
EVIDENCE=Path(__file__).resolve().parent
NS="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
FILES={"KCOR.xlsx":"MFG_MP_","KCOR_CMR.xlsx":"_MFG_D"}

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def sheet_targets(z,needle):
    wb=ET.fromstring(z.read("xl/workbook.xml"))
    rels={x.attrib["Id"]:x.attrib["Target"] for x in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    out={}
    for s in wb.findall(f"{{{NS}}}sheets/{{{NS}}}sheet"):
        name=s.attrib["name"]
        if needle not in name:continue
        target=rels[s.attrib[f"{{{REL}}}id"]]
        path=target.lstrip("/") if target.startswith("/") else "xl/"+target
        assert not any(n.endswith(path.rsplit("/",1)[-1]+".rels") for n in z.namelist()), name
        out[path]=name
    assert len(out)==3,out
    return out

def notice(filename,sheet):
    if filename=="KCOR.xlsx":
        reason="Historical manufacturer ratios were superseded: they were not frailty-adjusted KCOR and used an incorrect enrollment reference. No current estimate or interval is available until regeneration."
        fields=["manufacturer_status","reason","historical_archive","KCOR_MP","CI_lower","CI_upper"]
    else:
        reason="Historical manufacturer counts do not reconcile to the ordinary fixed-enrollment dose histories. Regenerate from the corrected converter before manufacturer analysis."
        fields=["manufacturer_status","reason","historical_archive","MFG"]
    values=["UNAVAILABLE - OBSOLETE",reason,f"m173/data/Czech/{filename}; original tab: {sheet}"]+[""]*(len(fields)-3)
    def row(n,vals,style=False):
        cells=[]
        for i,v in enumerate(vals):
            col=chr(ord("A")+i)
            a=f' r="{col}{n}"'+(' s="1"' if style else '')
            if v:cells.append(f'<c{a} t="inlineStr"><is><t>{escape(v)}</t></is></c>')
            else:cells.append(f'<c{a}/>')
        return f'<row r="{n}">'+''.join(cells)+'</row>'
    # Wide status and explanation fields keep the warning readable.
    end=chr(ord("A")+len(fields)-1)
    doc=f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{NS}"><sheetPr><tabColor rgb="FFFF0000"/></sheetPr>
<dimension ref="A1:{end}2"/><sheetViews><sheetView workbookViewId="0"><selection activeCell="A1" sqref="A1"/></sheetView></sheetViews>
<sheetFormatPr defaultRowHeight="15"/><cols><col min="1" max="1" width="30" customWidth="1"/><col min="2" max="2" width="155" customWidth="1"/><col min="3" max="3" width="83" customWidth="1"/><col min="4" max="{len(fields)}" width="15" customWidth="1"/></cols>
<sheetData>{row(1,fields,True)}{row(2,values)}</sheetData></worksheet>'''
    return doc.encode("utf-8")

def streamhash(z,name):
    with z.open(name) as f:return hashlib.file_digest(f,"sha256").hexdigest()

def update(filename,needle):
    old=PARENT/"data/Czech"/filename
    path=ROOT/"data/Czech"/filename
    assert sha(old)==sha(path),"candidate not exact parent copy"
    tmp=path.with_suffix(".tmp.xlsx")
    with ZipFile(old) as src:
        targets=sheet_targets(src,needle)
        with ZipFile(tmp,"w",allowZip64=True) as dst:
            dst.comment=src.comment
            for info in src.infolist():
                if info.filename in targets:
                    dst.writestr(info,notice(filename,targets[info.filename]))
                else:
                    with src.open(info) as a,dst.open(info,"w",force_zip64=info.file_size>=2**31) as b:
                        shutil.copyfileobj(a,b,1024*1024)
    os.replace(tmp,path)
    with ZipFile(old) as a,ZipFile(path) as b:
        assert a.namelist()==b.namelist()
        assert all(streamhash(a,n)==streamhash(b,n) for n in a.namelist() if n not in targets)
        for member,name in targets.items():
            s=ET.fromstring(b.read(member))
            vals=[x.text for x in s.findall(f".//{{{NS}}}t")]
            assert "UNAVAILABLE - OBSOLETE" in vals
            assert len(s.findall(f".//{{{NS}}}row"))==2
        result=dict(file=filename,archive_sha256=sha(old),active_sha256=sha(path),changed_tabs=list(targets.values()),other_package_parts_identical=len(a.namelist())-len(targets))
    print(result,flush=True);return result

if __name__=="__main__":
    results=[update(filename,needle) for filename,needle in FILES.items()]
    (EVIDENCE/"verification.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
