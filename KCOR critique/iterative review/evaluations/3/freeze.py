"""Reconstruct series 3 from the hashed series 2 editions and sealed overlays."""
import argparse
import difflib
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parents[1]
PRIOR = HERE.with_name("2")
PAPER = "documentation/preprint/"
LABELS = {"original": "m17", "evaluated": "m42", "checkpoint": "m61", "current": "m83"}


def digest(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def make_overlays():
    entries = {}
    for label, base in [("checkpoint", REVIEW/"checkpoints/iteration-1"), ("current", REVIEW)]:
        for row in json.loads((base/"revised/manifest.json").read_text(encoding="utf8")):
            path = base/"revised"/row["path"]
            assert digest(path) == row["sha256"]
            entries[label+"/"+row["path"]] = path.read_bytes()
        entries[label+"/"+PAPER+"paper.pdf"] = (base/"KCOR-clean.pdf").read_bytes()
    def include(rel, path):
        entries["current/"+rel] = path.read_bytes()
    def relocated(rel, source, changes):
        text = source.read_text(encoding="utf8")
        for old, new in changes:
            assert old in text, old
            text = text.replace(old, new)
        entries["current/"+rel] = text.encode()
    studies = REVIEW.parent
    include("analysis/table8/table8_bootstrap.json", studies/"2.4/table8_bootstrap.json")
    # The evaluated runner already loads the bundled workbook. Select the current
    # production joint fit using the same partial as the live wave-entry replay.
    entries["current/analysis/table8/compare.py"] = (
        "from functools import partial\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "import os, sys\nos.environ['DATASET'] = 'Czech'\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))\n"
        "import KCOR\nadapter = SimpleNamespace(K=KCOR)\n"
        "production_fit = KCOR.fit_theta0_gompertz\njoint_fit = partial(production_fit, joint_seed=True)\n"
        "def initialize(weekly_update):\n    assert weekly_update == 'revised'\n").encode()
    relocated("analysis/table8/table8_bootstrap.py", studies/"1.3/table8_bootstrap.py",[
        ('subprocess.check_output(["git", "cat-file", "blob", INPUT_BLOB], cwd=HERE)', '(HERE/"input.xlsx").read_bytes()'),
        ('weekly_update = args.edition or ("revised" if args.check_workbook else "archived")',
         'if args.edition or not args.check_workbook:\n        parser.error("Use replay.py to generate the matching workbook")\n    weekly_update = "revised"')])
    relocated("analysis/quiet_windows/scan.py", studies/"2.5/scan.py",[
        ('REVIEW = HERE.parent / "iterative review"\nROOT = REVIEW.parents[1]', 'ROOT = HERE.parents[1]'),
        ('REVIEW / "revised/', 'ROOT / "'),
        ('subprocess.check_output(["git", "cat-file", "blob", INPUT], cwd=ROOT)',
         '(ROOT/"data/Czech/reference_without_COVID_correction-1-24-26/KCOR_CMR.xlsx").read_bytes()')])
    for name in ["windows.csv","summary.csv","check_workbook.py"]:
        include("analysis/quiet_windows/"+name, studies/"2.5"/name)
    relocated("analysis/cox_intervals/figure_intervals.py", studies/"2.1/figure_intervals.py",[
        ('HERE.parent / "iterative review/revised/documentation', 'HERE.parents[1] / "documentation')])
    include("analysis/cox_intervals/figure_intervals.csv", studies/"2.1/figure_intervals.csv")
    with zipfile.ZipFile(HERE/"edition-overlays.zip","x",zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(entries.items()):
            archive.writestr(name,data)


def run(dest, cache):
    previous = json.loads((PRIOR/"inputs.json").read_text(encoding="utf8"))
    if not cache:
        subprocess.run([sys.executable,str(PRIOR/"freeze.py"),str(dest)],check=True)
        cache = dest
    files = {}
    for version, source in [("original","baseline"),("evaluated","revised")]:
        folder = cache/LABELS[version]
        manifest = previous["editions"][source]
        for rel, expected in manifest.items():
            if digest(folder/rel) != expected:
                raise ValueError(rel)
            target = dest/LABELS[version]/rel
            if target != folder/rel:
                target.parent.mkdir(parents=True,exist_ok=True)
                os.link(folder/rel,target)
        files[version] = manifest
    with zipfile.ZipFile(HERE/"edition-overlays.zip") as z:
        for version in ("checkpoint","current"):
            folder = dest/LABELS[version]
            shutil.copytree(dest/LABELS["evaluated"],folder,copy_function=os.link)
            manifest = dict(files["evaluated"])
            for key in z.namelist():
                prefix,rel=key.split("/",1)
                if prefix != version:
                    continue
                path=folder/rel
                path.parent.mkdir(parents=True,exist_ok=True)
                path.unlink(missing_ok=True)
                path.write_bytes(z.read(key))
                manifest[rel]=digest(path)
            files[version]=manifest
    def record(path, content):
        if path.exists():
            assert path.read_text(encoding="utf8")==content, path
        else:
            path.write_text(content,encoding="utf8")
    record(HERE/"inputs.json",json.dumps({"source":previous["commit"],"labels":LABELS,"files":files},indent=2))
    for source in ("original","evaluated","checkpoint"):
        diff=[]
        for rel in sorted(set(files[source])|set(files["current"])):
            if files[source].get(rel)==files["current"].get(rel):
                continue
            if Path(rel).suffix.lower() in (".pdf",".png",".xlsx"):
                diff.append(f"Binary {rel}: {files[source].get(rel,'absent')} -> {files['current'].get(rel,'absent')}\n")
                continue
            def lines(label):
                p=dest/LABELS[label]/rel
                return p.read_text(encoding="utf8").splitlines(keepends=True) if p.exists() else []
            diff.extend(difflib.unified_diff(lines(source),lines("current"),"a/"+rel,"b/"+rel))
        record(HERE/(source+"-to-current.diff"),"".join(diff))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("destination",type=Path)
    parser.add_argument("--verified-cache",type=Path)
    parser.add_argument("--seal",action="store_true")
    args=parser.parse_args()
    if args.seal:
        make_overlays()
    run(args.destination.resolve(),args.verified_cache.resolve() if args.verified_cache else None)
