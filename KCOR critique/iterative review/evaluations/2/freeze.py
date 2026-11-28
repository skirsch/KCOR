"""Materialize the two blinded editions from pinned Git objects and preserved overlays."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

RUN = Path(__file__).resolve().parent
REVIEW = RUN.parents[1]
ROOT = REVIEW.parents[1]
REV = "82f546cd5cc1af2ea99e4501a1aa92d0c9335f78"
BOOK = "f2dbb3d7a39d47a06821048de042285df46785ba"
COMMON = ("code", "config", "data", "analysis", "example", "figures", "identifiability",
          "reference_results", "scripts", "test", "validation", "requirements.txt", "Makefile",
          "documentation/preprint/figures", "documentation/preprint/refs.bib")
PAPER = "documentation/preprint/"
FILES = ["code/KCOR.py", "code/KCOR_CMR.py", PAPER+"paper.md", PAPER+"supplement.md"]

def function_source(path, name):
    text = path.read_text(encoding="utf-8-sig")
    node = next(n for n in ast.parse(text).body if getattr(n, "name", None) == name)
    return ast.get_source_segment(text, node) + "\n"

def make_overlay():
    payload = {f"{label}/{f}": (REVIEW / label / f).read_bytes() for label in ("baseline", "revised") for f in FILES}
    payload["baseline/"+PAPER+"paper.pdf"] = (REVIEW.parent/"original-review/KCOR.pdf").read_bytes()
    payload["revised/"+PAPER+"paper.pdf"] = (REVIEW/"KCOR-clean.pdf").read_bytes()
    for label in ("baseline","revised"):
        payload[label+"/README.md"] = b"KCOR\n\nThe combined manuscript, its Markdown sources and figures are in documentation/preprint. Method code is in code and scientific inputs and analyses are in the corresponding repo directories.\n"
    old = REVIEW.parent/"1.3"
    current = REVIEW.parent/"1.6"
    validation = REVIEW.parent/"1.3/objection3_corrected_validation"
    for name in ("replicates.csv","summary.json"):
        payload["revised/validation/person_bootstrap/"+name] = (validation/"final_2026-11-10/assessment"/name).read_bytes()
    payload["revised/validation/person_bootstrap/design.json"] = (validation/"design.json").read_bytes()
    payload["revised/validation/person_bootstrap/paired.json"] = (old/"paired-seed-fit/results.json").read_bytes()
    fit = old/"paired-seed-fit/compare.py"
    step = REVIEW.parent/"1.3/objection3_paper_method/paper_step1.py"
    bootstrap = (old/"table8_bootstrap.py").read_text(encoding="utf8").replace(
        'subprocess.check_output(["git", "cat-file", "blob", INPUT_BLOB], cwd=HERE)', '(HERE/"input.xlsx").read_bytes()')
    replay = (current/"replay.py").read_text(encoding="utf8").replace(
        'HERE.parent / "1.3"', "HERE").replace(
        'subprocess.check_output(["git", "cat-file", "blob", study.INPUT_BLOB], cwd=HERE)', '(HERE/"input.xlsx").read_bytes()')
    prefix = ("import importlib.util, inspect, os\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
              "import numpy as np\nfrom scipy.optimize import least_squares\n"
              "PRODUCTION = Path(__file__).resolve().parents[2]/'code/KCOR.py'\nadapter=SimpleNamespace()\n"
              "os.environ['DATASET']='Czech'\n")
    compare = prefix + function_source(step,"fit_step1") + function_source(fit,"initialize")
    # Only the revised mode is used by replay. Exclude the historical comparison arm.
    i, j = compare.index('    if weekly_update == "archived":'), compare.index('    original = "theta_seed')
    compare = compare[:i] + '    assert weekly_update == "revised"\n' + compare[j:]
    for name, data in {"compare.py":compare.encode(), "table8_bootstrap.py":bootstrap.encode(),
                       "replay.py":replay.encode(),"table8_bootstrap.json":(current/"table8_bootstrap.json").read_bytes(),
                       "input.xlsx":subprocess.check_output(["git","cat-file","blob",BOOK],cwd=ROOT)}.items():
        payload["revised/analysis/table8/"+name]=data
    with zipfile.ZipFile(RUN/"edition-overlays.zip","w",zipfile.ZIP_DEFLATED) as z:
        for name, data in payload.items(): z.writestr(name,data)

def materialize(destination):
    base, revised = destination/"m17", destination/"m42"
    base.mkdir(parents=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git","archive",REV,*COMMON],cwd=ROOT,stdout=archive,check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar: tar.extractall(base,filter="data")
    shutil.copytree(base,revised,copy_function=os.link)
    finish_materialization(base,revised)

def finish_materialization(base,revised):
    if not (RUN/"edition-overlays.zip").exists(): make_overlay()
    mapping = {"baseline":base,"revised":revised}
    with zipfile.ZipFile(RUN/"edition-overlays.zip") as z:
        for entry in z.namelist():
            label, rel = entry.split("/",1)
            target = mapping[label]/rel
            target.parent.mkdir(parents=True,exist_ok=True)
            target.unlink(missing_ok=True)
            target.write_bytes(z.read(entry))
    paths = FILES + [p.relative_to(revised).as_posix() for d in ("analysis/table8","validation/person_bootstrap") for p in sorted((revised/d).glob("*"))]
    diff = []
    for rel in paths:
        if rel.endswith((".xlsx",".pdf")): continue
        before, after = base/rel, revised/rel
        a = before.read_text(encoding="utf8").splitlines(True) if before.exists() else []
        diff.extend(difflib.unified_diff(a,after.read_text(encoding="utf8").splitlines(True),
                    fromfile="a/"+rel if before.exists() else "/dev/null",tofile="b/"+rel))
    verify_or_record("changes.diff","".join(diff))
    tables = {}
    for label,directory in mapping.items():
        tables[label]={p.relative_to(directory).as_posix():hashlib.file_digest(p.open("rb"),"sha256").hexdigest()
                       for p in sorted(directory.rglob("*")) if p.is_file()}
    verify_or_record("inputs.json",json.dumps(dict(commit=REV,common_paths=COMMON,editions=tables),indent=2))

def verify_or_record(name,text):
    target=RUN/name
    if target.exists():
        if target.read_text(encoding="utf8")!=text:
            raise ValueError(f"Reconstructed {name} differs from the recorded evaluation input")
    else:
        target.write_text(text,encoding="utf8")

if __name__ == "__main__":
    materialize(Path(sys.argv[1]))
