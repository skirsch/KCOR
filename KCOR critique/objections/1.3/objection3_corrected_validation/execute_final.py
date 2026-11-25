"""Run the complete fixed experiment and its audits; retain child exit codes."""
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    run = HERE/'final_2026-11-10'
    results = []
    stages = [('run', [str(HERE/'run_study.py'), '--phase','final','--out',str(run),'--workers','6']),
              ('summarize', [str(HERE/'summarize.py'),str(run)]),
              ('audit', [str(HERE/'finish_study.py'),str(run)])]
    for label,args in stages:
        with (HERE/f'final_{label}.log').open('x',encoding='utf-8') as log:
            child = subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(stage=label,exit_code=child.returncode,finished=datetime.now(timezone.utc).isoformat()))
        print(label, 'exit:', child.returncode, flush=True)
        if child.returncode:
            break
    with (HERE/'final_execution.json').open('x',encoding='utf-8') as f:
        json.dump(results,f,indent=2)
    if any(r['exit_code'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
