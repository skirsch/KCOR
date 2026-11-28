"""Check thread continuity, handoff identity and hashes after all reviews finish."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
THREADS = [
    ("blind","blind-targeted"),
    ("original-first","original-same","original-same-follow"),
    ("original-handoff","original-handoff-follow"),
    ("evaluated-first","evaluated-same","evaluated-same-follow"),
    ("evaluated-handoff","evaluated-handoff-follow"),
    ("chain-original","chain-evaluated","chain-current","chain-follow"),
]


def verify():
    sessions = set()
    count = 0
    for rep in range(1,4):
        folder = HERE/f"repeat-{rep}"
        for path in THREADS:
            branch = None
            for stage in path:
                record = json.loads((folder/(stage+".json")).read_bytes())
                raw = (folder/(stage+".md")).read_bytes()
                assert record["sha256"] == hashlib.sha256(raw).hexdigest(), stage
                assert record["model"] == "gpt-6.2-astra" and record["reasoning_effort"]=="high"
                assert record["parent_thread"] == branch
                assert len(raw.strip())>0
                if not branch:
                    branch=record["thread"]
                    assert branch not in sessions
                    sessions.add(branch)
                assert record["thread"]==branch, (stage,branch)
                if stage.endswith("-handoff"):
                    assert record["verbatim_handoff"]==stage.replace("-handoff","-first")
                    first=folder/(record["verbatim_handoff"]+".md")
                    assert record["handoff_sha256"]==hashlib.sha256(first.read_bytes()).hexdigest()
                count+=1
    assert len(sessions)==18 and count==48
    print(f"Verified {count} archived answers and {len(sessions)} distinct reviewer conversations.")


if __name__=="__main__":
    verify()
