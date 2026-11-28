from pathlib import Path
import json,hashlib
A=Path(__file__).resolve().parent
stages=["blind","blind-targeted","original-first","original-same","original-handoff","original-same-follow","original-handoff-follow","evaluated-first","evaluated-same","evaluated-handoff","evaluated-same-follow","evaluated-handoff-follow","chain-original","chain-evaluated","chain-current","chain-follow"]
current=["blind","original-same","original-handoff","evaluated-same","evaluated-handoff","chain-current"]
complete=[];missing=[];threads=[]
for rep in range(1,4):
 d=A/f"repeat-{rep}"
 for stage in stages:
  p=d/(stage+".json");q=d/(stage+".md")
  if not p.exists():
   missing.append((rep,stage));continue
  row=json.loads(p.read_text(encoding="utf8"))
  assert hashlib.sha256(q.read_bytes()).hexdigest()==row["sha256"],p
  if row["parent_thread"]:assert row["thread"]==row["parent_thread"],p
  if row["verbatim_handoff"]:
   prev=(d/(row["verbatim_handoff"]+".md")).read_bytes()
   assert hashlib.sha256(prev).hexdigest()==row["handoff_sha256"],p
   prev_thread=json.loads((d/(row["verbatim_handoff"]+".json")).read_text())["thread"]
   assert row["thread"]!=prev_thread,p
  if not row["parent_thread"]:threads.append(row["thread"])
  complete.append((rep,stage))
assert len(threads)==len(set(threads)),"Nonindependent fresh threads"
print(f"Completed {len(complete)}/48 turns; current open reviews {sum(x[1] in current for x in complete)}/18; independent threads {len(threads)}.")
if missing:print("Pending",missing)