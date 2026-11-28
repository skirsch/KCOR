"""Run the independent, continuing and handed-off reviews on frozen inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
MODEL = "gpt-6.2-astra"
QUESTION = "Please examine KCOR closely. What is wrong, if anything, and how would you fix it?"


def run(root, stage, rep, edition, prior=None, handoff=None, diff=None, follow=False, historical=False):
    archive = HERE/f"repeat-{rep}"
    archive.mkdir(exist_ok=True)
    output = archive/(stage+".md")
    record = archive/(stage+".json")
    if record.exists():
        saved = json.loads(record.read_text(encoding="utf8"))
        if hashlib.sha256(output.read_bytes()).hexdigest() != saved["sha256"]:
            raise ValueError(output)
        return saved["thread"]
    scratch = root/"turns"
    scratch.mkdir(exist_ok=True)
    def copy_for_review(source, name):
        target = scratch/name
        content = source.read_bytes()
        if not target.exists():
            with tempfile.NamedTemporaryFile(dir=scratch,delete=False) as tmp:
                tmp.write(content)
            os.replace(tmp.name,target)
        assert target.read_bytes()==content
        return target.as_posix()
    repo=(root/edition).as_posix()
    prompt = (f"The repository edition under scientific review is {repo}. Inspect it read-only with local tools. "
              "The combined manuscript and its sources are in documentation/preprint. Use only the specified inputs. "
              "No external sources, history search, other reviewers or parent directories. Cite checkable locations. ")
    if prior:
        prompt += "Continue this conversation. The edition specified here is the one to assess now. "
    if handoff:
        path=copy_for_review(archive/(handoff+".md"),f"{rep}-{handoff}-answer.md")
        prompt += f"Read the previous reviewer's full answer verbatim at {path}. "
    if diff:
        path=copy_for_review(HERE/diff if diff!="changes.diff" else HERE.with_name("2")/diff, diff)
        prompt += f"The changes from the previous edition are at {path}. "
    if follow and stage=="blind-targeted":
        prompt += ("Targeted follow-up, separate from your saved blind answer. Does each of these specific "
                   "objections still apply? Cite the current location and distinguish the factual discrepancy "
                   "from any broader untested claim. (1) The multiwave frailty recovery example used expected deaths "
                   "and fixed the true baseline rate without saying so. (2) Figure 1 used an earlier monthly slope "
                   "adjustment and the simulation comparison grid an earlier single-window cumulative-hazard fit "
                   "without identifying them. (3) The historical Poisson pilot and its empty table purported to support "
                   "coverage across independent datasets. Also check N1 known-frailty Cox inputs and plotted intervals, "
                   "N2 integrated weekly fitting, N3 last gap week, N4 first wave week, N5 scan denominator and fitter, "
                   "N6 positive-control intervention, N7 contamination threshold, N8 fixed reference week, N9 Table 8 "
                   "fitting unit and N10 selection of Figure 1. Keep new concerns distinct; report uncertainty when applicable.")
    elif follow:
        prompt += ("Targeted follow-up after your saved open answer. For each earlier objection you omitted or left "
                   "ambiguous, does this specific objection still apply to the revised version? What in the paper "
                   "or code supports your answer? Preserve the open answer separately. ")
        if historical:
            path=copy_for_review(HERE.with_name("2")/f"repeat-{rep}"/"A.md",f"{rep}-historical-blind.md")
            prompt += f"Also check any unaddressed claims in the archived review of the previous edition, {path}. "
    else:
        prompt += QUESTION
    args=[shutil.which("codex"),"exec","-s","read-only","-C",repo,"-m",MODEL,"-c",'model_reasoning_effort="high"']
    if prior:
        args+=["resume",prior]
    args+=["--skip-git-repo-check","--json","-o",str(output),"-"]
    trace=scratch/f"{rep}-{stage}.jsonl"
    print(f"START {rep} {stage}",flush=True)
    with trace.open("wb") as log, trace.with_suffix(".err").open("wb") as err:
        result=subprocess.run(args,input=prompt.encode(),stdout=log,stderr=err,cwd=repo)
    events=[json.loads(line) for line in trace.read_text(encoding="utf8").splitlines() if line.startswith("{")]
    thread=next((e["thread_id"] for e in events if e["type"]=="thread.started"),prior)
    if result.returncode or not output.exists() or not thread or not any(e["type"]=="turn.completed" for e in events):
        raise RuntimeError(f"Failed {rep} {stage}; see {trace}")
    if len(output.read_text(encoding="utf8"))<450 and not follow:
        raise RuntimeError(f"Check incomplete or inaccessible review: {output}")
    if prior and thread != prior:
        raise RuntimeError(f"{stage} resumed a different conversation")
    record.write_text(json.dumps(dict(model=MODEL,reasoning_effort="high",thread=thread,parent_thread=prior,
        edition=edition,diff=diff,verbatim_handoff=handoff,prompt=prompt,
        handoff_sha256=hashlib.sha256((archive/(handoff+".md")).read_bytes()).hexdigest() if handoff else None,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2),encoding="utf8")
    print(f"DONE {rep} {stage}",flush=True)
    return thread


def route(root, job):
    rep,path=job
    turn=partial(run,root,rep=rep)
    if path=="blind":
        turn("blind",edition="m83")
    elif path=="blind-targeted":
        while not (HERE/f"repeat-{rep}/blind.json").is_file():
            time.sleep(5)
        tid=json.loads((HERE/f"repeat-{rep}/blind.json").read_text(encoding="utf8"))["thread"]
        turn("blind-targeted",edition="m83",prior=tid,follow=True)
    elif path=="evaluated":
        thread=turn("evaluated-first",edition="m42")
        with ThreadPoolExecutor(2) as pool:
            def branch(fresh):
                tag="evaluated-handoff" if fresh else "evaluated-same"
                tid=turn(tag,edition="m83",prior=None if fresh else thread,
                         handoff="evaluated-first" if fresh else None,diff="evaluated-to-current.diff")
                turn(tag+"-follow",edition="m83",prior=tid,follow=True,historical=True)
            list(pool.map(branch,[False,True]))
    elif path=="chain":
        thread=turn("chain-original",edition="m17")
        thread=turn("chain-evaluated",edition="m42",prior=thread,diff="changes.diff")
        thread=turn("chain-current",edition="m83",prior=thread,diff="evaluated-to-current.diff")
        turn("chain-follow",edition="m83",prior=thread,follow=True)
    elif path=="original-handoff":
        thread=turn("original-handoff",edition="m83",handoff="original-first",diff="original-to-current.diff")
        turn("original-handoff-follow",edition="m83",prior=thread,follow=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    parser.add_argument("paths",nargs="+",choices=["blind","blind-targeted","evaluated","chain","original-handoff"])
    opt=parser.parse_args()
    jobs=[(rep,path) for path in opt.paths for rep in range(1,4)]
    with ThreadPoolExecutor(min(9,len(jobs))) as pool:
        list(pool.map(partial(route,opt.root.resolve()),jobs))
