from pathlib import Path
import os, json, hashlib, shutil, difflib, zipfile
A=Path(__file__).resolve().parent
D=Path(os.environ["TEMP"])/"kcor-s4-approved"; C=D/"m94"
old=json.loads((A.parent/"3/inputs.json").read_text(encoding="utf8"))
def digest(p):
 with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
labels={"original":"m17","evaluated":"m83","checkpoint":"m61","current":"m94"}
files={"original":old["files"]["original"],"evaluated":old["files"]["current"],"checkpoint":old["files"]["checkpoint"]}
# Include only scientific inputs, source files, figures and analysis results, not cache artifacts.
pp=C/"documentation/preprint/paper.pdf"
pp.unlink(missing_ok=True);shutil.copy2(D/"review-paper.pdf",pp)
r=C/"analysis/README.md"
r.unlink(missing_ok=True)
r.write_text("""# Numerical results

Table 8: `table8/replay.py` reproduces the full ordinary workbook and the person-bootstrap results; the stored 999 draws are in `survivor_validation/table8-draws.npz`. Point estimates and the first matched draw were verified against the integrated implementation. Table 9: `positive_control/replay.py` regenerates the fitted workbook points and the full 999-draw intervals. `quiet_windows/scan.py` regenerates the placement scan. The `survivor_validation`, `information`, `contamination` and `alpha` directories contain numerical results of the specified validation designs and theoretical benchmarks.

The main and supplement Markdown files are authoritative manuscript sources; the combined PDF is a render of both. Historical validation configurations are distinguished from the current method by the manuscript.
""",encoding="utf8")
wanted=set(files["evaluated"])
wanted.update(str(p.relative_to(C)).replace("\\","/") for p in (C/"analysis").rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",))
wanted.add("code/kcor_survivor.py")
# revised files may include figures and new test
rmanifest=json.loads((A.parents[1]/"revised/manifest.json").read_text())
wanted.update(x["path"] for x in rmanifest)
files["current"]={rel:digest(C/rel) for rel in sorted(wanted) if (C/rel).is_file()}
print("Current files",len(files["current"]),flush=True)
(A/"inputs.json").write_text(json.dumps(dict(source=old["source"],labels=labels,files=files),indent=2),encoding="utf8")
for source in ("original","evaluated","checkpoint"):
 changes=[]
 for rel in sorted(set(files[source])|set(files["current"])):
  if files[source].get(rel)==files["current"].get(rel):continue
  left=D/labels[source]/rel;right=C/rel
  if Path(rel).suffix.lower() not in (".py",".md",".yaml",".yml",".txt",".json",".tex",".lua",".cmake") or max((p.stat().st_size for p in (left,right) if p.exists()),default=0)>500000:
   changes.append(f"Data/asset {rel}: {files[source].get(rel,'absent')} -> {files['current'].get(rel,'absent')}\n")
   continue
  def lines(p):
   return p.read_text(encoding="utf-8-sig",errors="replace").splitlines(keepends=True) if p.exists() else []
  changes.extend(difflib.unified_diff(lines(left),lines(right),"a/"+rel,"b/"+rel))
 (A/(source+"-to-current.diff")).write_text("".join(changes),encoding="utf8")
with zipfile.ZipFile(A/"edition-overlay.zip","x",zipfile.ZIP_DEFLATED) as z:
 for rel in files["current"]:
  if files["evaluated"].get(rel)!=files["current"][rel]:
   z.write(C/rel,rel)
print("Sealed overlays and diffs",flush=True)